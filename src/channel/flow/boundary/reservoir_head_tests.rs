use super::*;
use crate::channel::flow::{
    ChannelConfig, ChannelInitialState, ChannelReachDefinition, RectangularChannelCell,
    geometry::ChannelGeometry,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn context(gravity: f64) -> Result<Context<CpuRuntime>, ChannelError> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    Context::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "test datum".into(),
            provenance: "independent reservoir head arithmetic references".into(),
            geometry: ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 1.0,
                    width_m: 1.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 1.0,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: vec![],
            config: ChannelConfig {
                gravity,
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![0.5],
            longitudinal_momentum_m4_per_s: vec![0.0],
        },
    )
}

fn check_head(bounds: &Bounds<CpuRuntime>, floor_bits: u64, ceiling_bits: u64) -> TestResult {
    let lo = bounds.lo.item::<f64>()?;
    let hi = bounds.hi.item::<f64>()?;
    let floor = f64::from_bits(floor_bits);
    let ceiling = f64::from_bits(ceiling_bits);
    assert!(lo.is_finite() && hi.is_finite());
    assert!(
        lo >= 0.0 && lo <= floor,
        "lower={lo:e}, reference floor={floor:e}"
    );
    assert!(hi >= ceiling, "upper={hi:e}, reference ceiling={ceiling:e}");
    if floor > 0.0 {
        assert!(
            (hi - lo) / ceiling <= 1e-12,
            "head interval [{lo:e}, {hi:e}]"
        );
    }
    if ceiling_bits == 0 {
        assert_eq!(lo.to_bits(), 0);
        assert_eq!(hi.to_bits(), 0);
    }
    Ok(())
}

#[test]
fn independently_weighted_heads_enclose_extreme_exact_input_references() -> TestResult {
    // Independent 150-digit references bracket each exact binary64 input expression.
    let fixtures = [
        (
            -1e-170,
            1.0,
            1e308,
            (0, 1),
            (4123597567603287858, 4123597567603287859),
        ),
        (
            -1e200,
            1e200,
            1.0,
            (7594448965539947098, 7594448965539947098),
            (7594448965539947098, 7594448965539947098),
        ),
        (
            -1e200,
            1e308,
            0.0,
            (5978682040737353251, 5978682040737353252),
            (0, 0),
        ),
    ];
    for (u, gravity, k, velocity_bits, loss_bits) in fixtures {
        let ctx = context(gravity)?;
        let velocity = ctx.tensor(&[u])?;
        let coefficient = ctx.tensor(&[k])?;
        let ReservoirHeads {
            velocity_head: head,
            entrance_loss_head: loss,
            ..
        } = ctx.reservoir_heads(&Bounds::exact(&velocity), &coefficient)?;
        check_head(&head, velocity_bits.0, velocity_bits.1)?;
        check_head(&loss, loss_bits.0, loss_bits.1)?;
    }
    Ok(())
}

#[test]
fn uncertain_velocity_sign_retains_possible_amplified_incoming_loss() -> TestResult {
    let ctx = context(1.0)?;
    let velocity = Bounds {
        lo: ctx.tensor(&[-1e-170])?,
        hi: ctx.tensor(&[1e-170])?,
    };
    let coefficient = ctx.tensor(&[1e308])?;
    let ReservoirHeads {
        entrance_loss_head: loss,
        ..
    } = ctx.reservoir_heads(&velocity, &coefficient)?;
    let lo = loss.lo.item::<f64>()?;
    let hi = loss.hi.item::<f64>()?;
    let reference_ceiling = f64::from_bits(4123597567603287859);
    assert_eq!(lo.to_bits(), 0);
    assert!(hi.is_finite() && hi >= reference_ceiling);
    assert!(hi / reference_ceiling <= 1.0 + 1e-12);
    Ok(())
}

#[test]
fn certified_outflow_selects_exact_zero_incoming_heads_before_large_loss_work() -> TestResult {
    let ctx = context(1e200)?;
    let velocity = ctx.tensor(&[1e200])?;
    let coefficient = ctx.tensor(&[f64::MAX])?;
    let ReservoirHeads {
        velocity_head: head,
        entrance_loss_head: loss,
        dissipation_head: exit,
    } = ctx.reservoir_heads(&Bounds::exact(&velocity), &coefficient)?;
    check_head(&head, 0, 0)?;
    check_head(&loss, 0, 0)?;
    check_head(&exit, 7594448965539947098, 7594448965539947098)?;
    Ok(())
}
