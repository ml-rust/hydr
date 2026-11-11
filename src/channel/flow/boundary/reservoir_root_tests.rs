use super::*;
use crate::channel::flow::{
    ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
    ChannelTensorBoundary, RectangularChannelCell, TriangularChannelCell,
    boundary::types::EndpointInterior,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use std::sync::Arc;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn context(triangle: bool, bed: f64) -> Result<Context<CpuRuntime>, ChannelError> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let geometry = if triangle {
        ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 1.0,
            right_bank_slope_h_per_v: 1.0,
            bed_elevation_m: bed,
            cells: vec![TriangularChannelCell {
                id: "cell".into(),
                length_m: 128.0,
                maximum_depth_m: 4.0,
                manning_n: 0.0,
            }],
        }
    } else {
        ChannelGeometry::Rectangular {
            cells: vec![RectangularChannelCell {
                id: "cell".into(),
                length_m: 128.0,
                width_m: 1.0,
                bed_elevation_m: bed,
                maximum_depth_m: 4.0,
                manning_n: 0.0,
            }],
        }
    };
    Context::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "exact resting reservoir root fixture".into(),
            geometry,
            lateral_bindings: vec![],
            config: ChannelConfig {
                gravity: 1.0,
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![128.0],
            longitudinal_momentum_m4_per_s: vec![0.0],
        },
    )
}

fn interior(
    ctx: &Context<CpuRuntime>,
    velocity: f64,
    discharge: f64,
) -> Result<Interior<CpuRuntime>, ChannelError> {
    let depth = ctx.tensor(&[1.0])?;
    let velocity = ctx.tensor(&[velocity])?;
    Ok(Interior {
        depth: depth.clone(),
        velocity: velocity.clone(),
        area: depth,
        discharge: ctx.tensor(&[discharge])?,
        depth_bounds: Bounds {
            lo: ctx.tensor(&[1.0 - 2.0 * f64::EPSILON])?,
            hi: ctx.tensor(&[1.0 + 2.0 * f64::EPSILON])?,
        },
        velocity_bounds: Bounds::exact(&velocity),
    })
}

#[test]
fn exact_rest_retains_original_depth_and_section_enclosure() -> TestResult {
    for triangle in [false, true] {
        for bed in [0.0, 16.0] {
            let ctx = context(triangle, bed)?;
            let state = interior(&ctx, 0.0, 0.0)?;
            for loss in [0.0, 1.0] {
                let (request, _) = ctx.boundary_reservoir_prepare(
                    &state,
                    &ctx.tensor(&[4.0])?,
                    &ctx.tensor(&[bed + 1.0])?,
                    &ctx.tensor(&[bed])?,
                    &ctx.tensor(&[loss])?,
                )?;
                let DepthRequest::Known(depth) = request else {
                    return Err("exact resting depth enters numerical solving".into());
                };
                assert_eq!(depth.value.item::<f64>()?.to_bits(), 1.0_f64.to_bits());
                assert_eq!(
                    depth.bounds.lo.item::<f64>()?,
                    state.depth_bounds.lo.item::<f64>()?
                );
                assert_eq!(
                    depth.bounds.hi.item::<f64>()?,
                    state.depth_bounds.hi.item::<f64>()?
                );
                assert_eq!(depth.iterations, 0);
                assert!(!depth.approximate);
                for endpoint in [0, 1] {
                    let inputs = EndpointInterior {
                        depth: ctx.tensor(&[1.0, 1.0])?,
                        area: ctx.tensor(&[1.0, 1.0])?,
                        velocity: ctx.tensor(&[0.0, 0.0])?,
                        discharge: ctx.tensor(&[0.0, 0.0])?,
                    };
                    let policy = ChannelTensorBoundary::QuiescentReservoir {
                        elevation_m: ctx.tensor(&[bed + 1.0])?,
                        entrance_loss_coefficient: ctx.tensor(&[loss])?,
                    };
                    let wall = ChannelTensorBoundary::Wall;
                    let (left, right) = if endpoint == 0 {
                        (&policy, &wall)
                    } else {
                        (&wall, &policy)
                    };
                    let frame = ctx
                        .scalar_boundary_frame(&inputs, left, right)?
                        .ok_or("missing resting reservoir frame")?;
                    assert_eq!(
                        frame.traces.depth_m.item::<f64>()?.to_bits(),
                        1.0_f64.to_bits()
                    );
                    assert_eq!(frame.traces.outward_discharge_m3_per_s.item::<f64>()?, 0.0);
                    assert_eq!(frame.longitudinal_water.item::<f64>()?, 0.0);
                    assert_eq!(
                        frame.traces.momentum_flux_m4_per_s2.item::<f64>()?,
                        if triangle { 1.0 / 3.0 } else { 0.5 }
                    );
                    assert_eq!(frame.traces.root_iterations, vec![0]);
                    assert_eq!(frame.traces.endpoints, vec![endpoint]);
                    assert!(frame.traces.reservoir.is_some());
                }
            }
        }
    }
    Ok(())
}

#[test]
fn unequal_or_uncertain_heads_and_nonzero_motion_retain_numerical_solving() -> TestResult {
    for triangle in [false, true] {
        let ctx = context(triangle, 0.0)?;
        for (head, bed, velocity, discharge) in [
            (f64::from_bits(1.0_f64.to_bits() - 1), 0.0, 0.0, 0.0),
            (f64::from_bits(1.0_f64.to_bits() + 1), 0.0, 0.0, 0.0),
            (1.0, 0.0, 1e-3, 1e-3),
            (1.0, 0.0, 1e-3, 0.0),
            (1.0, 0.0, 0.0, 1e-3),
            (1.0, 2.0_f64.powi(-54), 0.0, 0.0),
        ] {
            let state = interior(&ctx, velocity, discharge)?;
            let (request, reservoir) = ctx.boundary_reservoir_prepare(
                &state,
                &ctx.tensor(&[4.0])?,
                &ctx.tensor(&[head])?,
                &ctx.tensor(&[bed])?,
                &ctx.tensor(&[1.0])?,
            )?;
            assert!(matches!(request, DepthRequest::Reservoir(_)));
            if bed > 0.0 {
                assert_eq!(reservoir.head.item::<f64>()?, 1.0);
                assert!(reservoir.bounds.lo.item::<f64>()? < 1.0);
            }
        }
    }
    Ok(())
}

#[test]
fn resting_root_retains_existing_depth_width_controls() -> TestResult {
    let mut ctx = context(false, 0.0)?;
    Arc::make_mut(&mut ctx.definition)
        .config
        .root_absolute_depth = f64::from_bits(1);
    Arc::make_mut(&mut ctx.definition)
        .config
        .root_relative_depth = 0.0;
    let result = ctx.boundary_reservoir_prepare(
        &interior(&ctx, 0.0, 0.0)?,
        &ctx.tensor(&[4.0])?,
        &ctx.tensor(&[1.0])?,
        &ctx.tensor(&[0.0])?,
        &ctx.tensor(&[0.0])?,
    );
    assert!(matches!(result, Err(ChannelError::Numerical { .. })));
    Ok(())
}
