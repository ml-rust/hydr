//! One authoritative routed volume and complementary outlet partition.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::bounds;
use crate::runoff::ReservoirError;
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;

#[derive(Debug, Clone)]
pub(crate) struct Partition<R: Runtime<DType = DType>> {
    pub routed: Tensor<R>,
    pub outlet: Tensor<R>,
    pub residual: Tensor<R>,
    pub bound: Tensor<R>,
}

#[derive(Debug, Clone)]
pub(crate) struct Merged<R: Runtime<DType = DType>> {
    pub volume: Tensor<R>,
    pub bound: Tensor<R>,
}

pub(crate) fn partition<R>(
    ctx: &TensorReservoirContext<R>,
    donor: &Tensor<R>,
    fraction: &Tensor<R>,
) -> Result<Partition<R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>,
{
    bounds::check_nonnegative(ctx, donor, "donor export volume")?;
    bounds::check_nonnegative(ctx, fraction, "routed fraction")?;
    let c = &ctx.client;
    ctx.reject_mask(
        &c.gt(fraction, &ctx.one)?,
        "routed fraction",
        &[("fraction", fraction)],
        &[],
        "Require a fraction within [0,1]",
    )?;
    let routed = c.where_cond(
        &c.eq(fraction, &ctx.zero)?,
        &ctx.zero,
        &c.where_cond(&c.eq(fraction, &ctx.one)?, donor, &c.mul(fraction, donor)?)?,
    )?;
    let outlet = c.sub(donor, &routed)?;
    bounds::check_nonnegative(ctx, &routed, "routed volume")?;
    bounds::check_nonnegative(ctx, &outlet, "direct outlet volume")?;
    let interior = c.mul(
        &c.gt(donor, &ctx.zero)?,
        &c.mul(&c.gt(fraction, &ctx.zero)?, &c.lt(fraction, &ctx.one)?)?,
    )?;
    ctx.reject_mask(
        &c.mul(
            &interior,
            &c.maximum(&c.le(&routed, &ctx.zero)?, &c.le(&outlet, &ctx.zero)?)?,
        )?,
        "positive transfer partition",
        &[
            ("donor", donor),
            ("fraction", fraction),
            ("routed", &routed),
            ("outlet", &outlet),
        ],
        &[],
        "Positive routed and outlet portions must remain representable",
    )?;
    let (sum, residual) = c.two_sum(&routed, &outlet)?;
    ctx.finite(&sum, "partition reconciliation")?;
    ctx.reject_mask(
        &c.ne(&sum, donor)?,
        "partition reconciliation",
        &[("donor", donor), ("sum", &sum)],
        &[],
        "Require routed plus complementary outlet to equal donor volume",
    )?;
    // The rounded high component equals donor. The low component retains the exact discrepancy.
    ctx.finite(&residual, "partition mathematical residual")?;
    let raw_bound = bounds::upward_add(
        ctx,
        &bounds::rounding(ctx, &outlet)?,
        &bounds::rounding(ctx, &sum)?,
    )?;
    let bound = c.where_cond(&interior, &raw_bound, &ctx.zero)?;
    ctx.reject_mask(
        &c.gt(&c.abs(&residual)?, &bound)?,
        "partition mathematical envelope",
        &[("residual", &residual), ("bound", &bound)],
        &[],
        "Require the represented partition discrepancy within its arithmetic bound",
    )?;
    Ok(Partition {
        routed,
        outlet,
        residual,
        bound,
    })
}

pub(crate) fn merge<R>(
    ctx: &TensorReservoirContext<R>,
    sources: &[&Tensor<R>],
) -> Result<Merged<R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>,
{
    let c = &ctx.client;
    let mut volume = ctx.zero.clone();
    let mut bound = ctx.zero.clone();
    for source in sources {
        bounds::check_nonnegative(ctx, source, "received source volume")?;
        let positive = c.gt(source, &ctx.zero)?;
        let existing = c.gt(&volume, &ctx.zero)?;
        let addition = c.mul(&positive, &existing)?;
        let sum = c.add(&volume, source)?;
        ctx.finite(&sum, "received volume merge")?;
        ctx.reject_mask(
            &c.mul(&positive, &c.le(&sum, &volume)?)?,
            "positive received volume merge",
            &[("existing", &volume), ("source", source), ("sum", &sum)],
            &[],
            "A positive interface volume is absorbed. Rescale volume support",
        )?;
        let next_bound = bounds::upward_add(ctx, &bound, &bounds::rounding(ctx, &sum)?)?;
        bound = c.where_cond(&addition, &next_bound, &bound)?;
        volume = c.where_cond(&positive, &c.where_cond(&existing, &sum, source)?, &volume)?;
    }
    Ok(Merged { volume, bound })
}

pub(crate) fn equal<R>(
    ctx: &TensorReservoirContext<R>,
    actual: &Tensor<R>,
    expected: &Tensor<R>,
    name: &str,
) -> Result<(), ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>,
{
    crate::runoff::numerics::reservoir::drainage::check_metadata(ctx, actual, name)?;
    ctx.finite(actual, name)?;
    ctx.reject_mask(
        &ctx.client.ne(actual, expected)?,
        name,
        &[("actual", actual), ("expected", expected)],
        &[],
        "Receipt differs from admitted arithmetic",
    )
}

pub(crate) fn check_partition<R>(
    ctx: &TensorReservoirContext<R>,
    donor: &Tensor<R>,
    fraction: &Tensor<R>,
    receipt: &Partition<R>,
) -> Result<(), ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>,
{
    let expected = partition(ctx, donor, fraction)?;
    for (name, actual, wanted) in [
        ("routed volume", &receipt.routed, &expected.routed),
        ("outlet volume", &receipt.outlet, &expected.outlet),
        ("partition residual", &receipt.residual, &expected.residual),
        ("partition bound", &receipt.bound, &expected.bound),
    ] {
        equal(ctx, actual, wanted, name)?;
    }
    Ok(())
}

pub(crate) fn check_merge<R>(
    ctx: &TensorReservoirContext<R>,
    sources: &[&Tensor<R>],
    receipt: &Merged<R>,
) -> Result<(), ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>,
{
    let expected = merge(ctx, sources)?;
    equal(
        ctx,
        &receipt.volume,
        &expected.volume,
        "merged received volume",
    )?;
    equal(ctx, &receipt.bound, &expected.bound, "merge uncertainty")
}

#[cfg(test)]
mod tests {
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
    fn complementary_partition_preserves_one_represented_donor_volume() -> Result<(), ReservoirError>
    {
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
}
