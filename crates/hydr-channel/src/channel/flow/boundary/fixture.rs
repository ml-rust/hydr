use super::algebra::{Bounds, Interior};
use crate::channel::flow::{
    ChannelConfig, ChannelError, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
    RectangularChannelCell, TriangularChannelCell, context::Context,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub(super) fn context(triangle: bool, bed: f64) -> Result<Context<CpuRuntime>, ChannelError> {
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

pub(super) fn interior(
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
