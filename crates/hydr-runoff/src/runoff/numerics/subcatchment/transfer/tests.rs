use super::*;
use crate::runoff::{ReservoirConfig, ReservoirElement, ReservoirParameters};
use numr::runtime::cpu::CpuRuntime;

fn context(count: usize) -> Result<TensorReservoirContext<CpuRuntime>, ReservoirError> {
    let parameters = ReservoirParameters::from_si(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)?;
    TensorReservoirContext::<CpuRuntime>::new(
        cpu_client(),
        (0..count)
            .map(|i| ReservoirElement {
                id: format!("transfer-{i}"),
                parameters,
            })
            .collect(),
        ReservoirConfig::default(),
    )
}

#[test]
fn complementary_partition_preserves_one_represented_donor_volume() -> Result<(), ReservoirError> {
    let ctx = context(5)?;
    let donor = ctx.tensor(&[1.0, 1.0, 1.0, 0.1, 0.0])?;
    let fractions = ctx.tensor(&[0.0, 1.0, 0.5, 0.3, 0.5])?;
    let receipt = partition(&ctx, &donor, &fractions)?;
    check_partition(&ctx, &donor, &fractions, &receipt)?;
    assert_eq!(
        receipt.routed.try_to_vec::<f64>()?,
        vec![0.0, 1.0, 0.5, 0.03, 0.0]
    );
    assert_eq!(
        ctx.client
            .add(&receipt.routed, &receipt.outlet)?
            .try_to_vec::<f64>()?,
        donor.try_to_vec::<f64>()?
    );
    for (residual, bound) in receipt
        .residual
        .try_to_vec::<f64>()?
        .into_iter()
        .zip(receipt.bound.try_to_vec::<f64>()?)
    {
        assert!(residual.abs() <= bound);
    }
    Ok(())
}

#[test]
fn rounded_partition_identity_retains_nonzero_mathematical_discrepancy()
-> Result<(), ReservoirError> {
    let ctx = context(1)?;
    let fraction = ctx.tensor(&[0.1])?;
    let receipt = partition(&ctx, &ctx.one, &fraction)?;
    assert_eq!(
        ctx.client
            .add(&receipt.routed, &receipt.outlet)?
            .item::<f64>()?,
        1.0
    );
    assert_eq!(receipt.residual.item::<f64>()?, 2f64.powi(-55));
    assert!(receipt.residual.item::<f64>()? <= receipt.bound.item::<f64>()?);
    check_partition(&ctx, &ctx.one, &fraction, &receipt)?;
    let mut altered = receipt;
    altered.residual = ctx.zero.clone();
    assert!(check_partition(&ctx, &ctx.one, &fraction, &altered).is_err());
    Ok(())
}

#[test]
fn unrepresentable_positive_partition_rejects_without_clipping() -> Result<(), ReservoirError> {
    let ctx = context(1)?;
    for (donor, fraction) in [
        (f64::from_bits(1), 0.5),
        (f64::from_bits(1), f64::from_bits(1.0f64.to_bits() - 1)),
    ] {
        assert!(partition(&ctx, &ctx.tensor(&[donor])?, &ctx.tensor(&[fraction])?).is_err());
    }
    for fraction in [-1.0, 1.1, f64::NAN] {
        assert!(partition(&ctx, &ctx.one, &ctx.tensor(&[fraction])?).is_err());
    }
    Ok(())
}

#[test]
fn ordered_merge_charges_only_actual_positive_additions() -> Result<(), ReservoirError> {
    let ctx = context(3)?;
    let first = ctx.tensor(&[0.0, 0.1, 1.0])?;
    let second = ctx.tensor(&[1.0, 0.2, 0.0])?;
    let merged = merge(&ctx, &[&ctx.zero, &first, &second, &ctx.zero])?;
    check_merge(&ctx, &[&ctx.zero, &first, &second, &ctx.zero], &merged)?;
    let bound = merged.bound.try_to_vec::<f64>()?;
    assert_eq!(bound[0], 0.0);
    assert!(bound[1] > 0.0);
    assert_eq!(bound[2], 0.0);
    assert_eq!(
        merged.volume.try_to_vec::<f64>()?,
        vec![1.0, 0.1 + 0.2, 1.0]
    );
    Ok(())
}

#[test]
fn absorbed_positive_merge_and_forged_receipts_reject() -> Result<(), ReservoirError> {
    let ctx = context(1)?;
    let tiny = ctx.tensor(&[f64::from_bits(1)])?;
    assert!(merge(&ctx, &[&ctx.one, &tiny]).is_err());
    let mut merged = merge(&ctx, &[&ctx.one])?;
    merged.bound = ctx.one.clone();
    assert!(check_merge(&ctx, &[&ctx.one], &merged).is_err());
    let mut divided = partition(&ctx, &ctx.one, &ctx.tensor(&[0.5])?)?;
    divided.outlet = ctx.one.clone();
    assert!(check_partition(&ctx, &ctx.one, &ctx.tensor(&[0.5])?, &divided).is_err());
    Ok(())
}

fn cpu_client() -> numr::runtime::cpu::CpuClient {
    numr::runtime::cpu::CpuClient::new(numr::runtime::cpu::CpuDevice::new())
        .with_parallelism(numr::runtime::cpu::ParallelismConfig::new(Some(1), None))
}
