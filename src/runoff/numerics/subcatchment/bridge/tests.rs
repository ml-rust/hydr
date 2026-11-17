use super::*;
use crate::runoff::{ReservoirConfig, ReservoirElement, ReservoirParameters};
use numr::runtime::cpu::CpuRuntime;

fn context(area: f64) -> Result<TensorReservoirContext<CpuRuntime>, ReservoirError> {
    TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        vec![ReservoirElement {
            id: "receiver".into(),
            parameters: ReservoirParameters::from_si(area, area, 0.0, 0.0, 1.0, 0.0)?,
        }],
        ReservoirConfig::default(),
    )
}

#[test]
fn receiver_cartesian_product_admits_the_frozen_representable_inputs() -> Result<(), ReservoirError>
{
    // Every listed input passes the policy's strict positive-intermediate predicates.
    for area in [1.0, 3.0, 1024.0] {
        let ctx = context(area)?;
        for volume in [1e-12, 0.1, 1.0] {
            let declared = ctx.tensor(&[volume])?;
            let received = transfer::merge(&ctx, &[&declared])?;
            for dt in [1.0, 3.0, 64.0] {
                for rain in [0.0, 1e-5] {
                    for old in [0.0, 0.1] {
                        let d0 = ctx.tensor(&[old])?;
                        let rain = ctx.tensor(&[rain])?;
                        let receipt = prepare(&ctx, &d0, &rain, &received, dt)?;
                        check_bridge(&ctx, &d0, &rain, &received, dt, &receipt)?;
                        let residual = receipt.residual.item::<f64>()?;
                        assert!(residual.abs() <= receipt.residual_bound.item::<f64>()?);
                        assert!(receipt.uncertainty.item::<f64>()? > 0.0);
                        assert_eq!(received.volume.item::<f64>()?, volume);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn zero_received_volume_preserves_primitive_source_bits_and_zero_charge()
-> Result<(), ReservoirError> {
    let ctx = context(3.0)?;
    let d0 = ctx.tensor(&[0.1])?;
    let rain = ctx.tensor(&[1e-5])?;
    let received = transfer::merge(&ctx, &[&ctx.zero])?;
    let bridge = prepare(&ctx, &d0, &rain, &received, 3.0)?;
    assert_eq!(
        bridge.input_depth.item::<f64>()?.to_bits(),
        (1e-5f64 * 3.0).to_bits()
    );
    assert_eq!(
        bridge.available_depth.item::<f64>()?.to_bits(),
        (0.1f64 + 1e-5f64 * 3.0).to_bits()
    );
    for tensor in [
        &bridge.received_depth,
        &bridge.received_rate,
        &bridge.depth_increment,
        &bridge.represented_received_volume,
        &bridge.residual,
        &bridge.residual_bound,
        &bridge.uncertainty,
    ] {
        assert_eq!(tensor.item::<f64>()?.to_bits(), 0.0f64.to_bits());
    }
    Ok(())
}

#[test]
fn transfer_underflow_absorption_and_overflow_reject() -> Result<(), ReservoirError> {
    let ctx = context(1024.0)?;
    for (volume, old, dt) in [
        (f64::from_bits(1), 0.0, 1.0),
        (1e-30, 1.0, 1.0),
        (f64::MAX, 0.0, f64::from_bits(1)),
    ] {
        let volume = ctx.tensor(&[volume])?;
        let received = transfer::merge(&ctx, &[&volume])?;
        assert!(prepare(&ctx, &ctx.tensor(&[old])?, &ctx.zero, &received, dt).is_err());
    }
    Ok(())
}

#[test]
fn owned_bridge_import_rejects_altered_uncertainty_and_intermediates() -> Result<(), ReservoirError>
{
    let ctx = context(3.0)?;
    let received = transfer::merge(&ctx, &[&ctx.one])?;
    let original = prepare(&ctx, &ctx.zero, &ctx.zero, &received, 3.0)?;
    for bound in [false, true] {
        let mut altered = original.clone();
        if bound {
            altered.uncertainty = ctx.zero.clone();
        } else {
            altered.received_rate = ctx.one.clone();
        }
        assert!(check_bridge(&ctx, &ctx.zero, &ctx.zero, &received, 3.0, &altered).is_err());
    }
    Ok(())
}

fn cpu_client() -> numr::runtime::cpu::CpuClient {
    numr::runtime::cpu::CpuClient::new(numr::runtime::cpu::CpuDevice::new())
        .with_parallelism(numr::runtime::cpu::ParallelismConfig::new(Some(1), None))
}
