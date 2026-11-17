use super::*;
use crate::channel::flow::{
    ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
    RectangularChannelCell, TriangularChannelCell,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn context(triangle: bool) -> Result<Context<CpuRuntime>, ChannelError> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let geometry = if triangle {
        ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 1.0,
            right_bank_slope_h_per_v: 1.0,
            bed_elevation_m: 0.0,
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
                bed_elevation_m: 0.0,
                maximum_depth_m: 4.0,
                manning_n: 0.0,
            }],
        }
    };
    Context::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "resolved weak compression wave fixture".into(),
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
            longitudinal_momentum_m4_per_s: vec![16.0],
        },
    )
}

fn check_weak_compression(triangle: bool) -> TestResult {
    let ctx = context(triangle)?;
    let c = &ctx.client;
    let depth = ctx.tensor(&[1.0])?;
    let velocity = ctx.tensor(&[0.125])?;
    let interior = Interior {
        depth: depth.clone(),
        velocity: velocity.clone(),
        area: depth.clone(),
        discharge: velocity.clone(),
        depth_bounds: Bounds::exact(&depth),
        velocity_bounds: Bounds::exact(&velocity),
    };
    let trace = Bounds::exact(&ctx.tensor(&[1.0 + 4.0 * f64::EPSILON])?);
    assert!(ctx.all(&c.gt(&trace.lo, &interior.depth_bounds.hi)?)?);
    let compatible = ctx.boundary_velocity(&trace, &interior)?;
    let sound = ctx.boundary_sound(&trace)?;
    let interior_sound = ctx.boundary_sound(&interior.depth_bounds)?;
    let lambda_i = ctx.bsub(&interior.velocity_bounds, &interior_sound)?;
    let lambda_b = ctx.bsub(&compatible, &sound)?;
    let mut magnitude = ctx.boundary_shock_magnitude(&trace, &interior.depth_bounds)?;
    magnitude.lo = c.maximum(&magnitude.lo, &interior_sound.lo)?;
    let shock = ctx.bsub(&interior.velocity_bounds, &magnitude)?;
    for bounds in [&lambda_i, &lambda_b, &shock] {
        assert!(bounds.hi.item::<f64>()? < 0.0);
    }
    // Independent directed enclosures overlap despite strictly positive compression.
    for characteristic in [&lambda_i, &lambda_b] {
        assert!(shock.lo.item::<f64>()? <= characteristic.hi.item::<f64>()?);
        assert!(characteristic.lo.item::<f64>()? <= shock.hi.item::<f64>()?);
    }
    let speed = ctx.boundary_wave(&trace, &interior, &compatible, &compatible)?;
    let speed = speed.item::<f64>()?;
    assert!(speed.is_finite() && speed > 0.0);
    assert!(speed >= shock.lo.item::<f64>()?.abs());
    Ok(())
}

#[test]
fn rectangular_weak_compression_admits_overlapping_directed_lax_enclosures() -> TestResult {
    check_weak_compression(false)
}

#[test]
fn triangular_weak_compression_admits_overlapping_directed_lax_enclosures() -> TestResult {
    check_weak_compression(true)
}
