//! Source-volume adjustment admission, separate from conservation tolerances.

use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::SoilError;
use numr::dtype::DType;

/// One stored-F64 binade spacing, including the smallest subnormal quantum.
pub(crate) fn rounding_bound(value: f64) -> Result<f64, SoilError> {
    finite(value, "rounding-bound input")?;
    finite(
        f64::EPSILON * value.abs() + f64::from_bits(1),
        "rounding bound",
    )
}

pub(crate) fn upward_add(a: f64, b: f64) -> Result<f64, SoilError> {
    nonnegative(a, "outward addition first operand")?;
    nonnegative(b, "outward addition second operand")?;
    let result = finite(a + b, "outward addition intermediate")?;
    finite(result + rounding_bound(result)?, "outward addition")
}

pub(crate) fn upward_mul(a: f64, b: f64) -> Result<f64, SoilError> {
    nonnegative(a, "outward product first operand")?;
    nonnegative(b, "outward product second operand")?;
    let result = finite(a * b, "outward product intermediate")?;
    finite(result + rounding_bound(result)?, "outward product")
}

/// Zero charges preserve the accepted accumulator exactly, including signed zero.
pub(crate) fn add_adjustment_charge(current: f64, step: f64) -> Result<f64, SoilError> {
    nonnegative(current, "accepted adjustment charge")?;
    nonnegative(step, "step adjustment charge")?;
    if step == 0.0 {
        Ok(current)
    } else {
        upward_add(current, step)
    }
}

/// Count-dependent reduction bound, cached by the surface runtime.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AdjustmentReduction {
    count: usize,
    denominator: f64,
}

impl AdjustmentReduction {
    pub(crate) fn new(count: usize) -> Result<Self, SoilError> {
        let additions = count
            .checked_add(7)
            .ok_or_else(|| SoilError::input("correction reduction count overflows usize"))?;
        if additions as u128 > (1u128 << 53) {
            return Err(SoilError::input(
                "correction reduction requires N+7 <= 2^53 entries",
            ));
        }
        let x = additions as f64 * f64::EPSILON;
        let d0 = 1.0 - x;
        let denominator = d0 - rounding_bound(d0)?;
        if !(0.0..1.0).contains(&x) || !denominator.is_finite() || denominator <= 0.0 {
            return Err(SoilError::input(
                "correction reduction denominator is not positive finite. Reduce the grid size",
            ));
        }
        Ok(Self { count, denominator })
    }

    pub(crate) fn totals<R>(
        self,
        client: &R::Client,
        depths: &Tensor<R>,
        area: f64,
    ) -> Result<(f64, f64), SoilError>
    where
        R: Runtime<DType = DType>,
        R::Client: BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + TypeConversionOps<R>
            + UnaryOps<R>,
    {
        if depths.shape() != [self.count]
            || depths.dtype() != DType::F64
            || !depths.device().is_same(client.device())
            || !area.is_finite()
            || area <= 0.0
        {
            return Err(SoilError::input(format!(
                "correction shape={:?}, area={area}: require {} flat F64 entries and positive finite area",
                depths.shape(),
                self.count
            )));
        }
        self.totals_with_product(client, depths, || client.mul_scalar(depths, area))
    }

    /// Reduce corrections with checked positive per-lane F64 areas, without broadcasting.
    pub(crate) fn totals_with_areas<R>(
        self,
        client: &R::Client,
        depths: &Tensor<R>,
        areas: &Tensor<R>,
    ) -> Result<(f64, f64), SoilError>
    where
        R: Runtime<DType = DType>,
        R::Client: BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + TypeConversionOps<R>
            + UnaryOps<R>,
    {
        if depths.shape() != [self.count]
            || areas.shape() != [self.count]
            || depths.dtype() != DType::F64
            || !depths.device().is_same(client.device())
            || areas.dtype() != DType::F64
            || !areas.device().is_same(client.device())
        {
            return Err(SoilError::input(format!(
                "correction shape={:?}, dtype={:?}, area shape={:?}, dtype={:?}: require {} matching flat F64 entries",
                depths.shape(),
                depths.dtype(),
                areas.shape(),
                areas.dtype(),
                self.count
            )));
        }
        if self.count == 0 {
            return Ok((0.0, 0.0));
        }
        let zero = Tensor::<R>::zeros(areas.shape(), DType::F64, areas.device())?;
        let invalid = client.maximum(
            &client.cast(&client.isnan(areas)?, DType::F64)?,
            &client.cast(&client.isinf(areas)?, DType::F64)?,
        )?;
        let invalid = client.maximum(&invalid, &client.le(areas, &zero)?)?;
        if let Some(index) = rejected_lane(client, &invalid)? {
            return Err(SoilError::input(
                "per-lane area must be positive finite F64. Rescale or correct area support",
            )
            .at_cell(index));
        }
        self.totals_with_product(client, depths, || client.mul(depths, areas))
    }

    fn totals_with_product<R>(
        self,
        client: &R::Client,
        depths: &Tensor<R>,
        multiply: impl FnOnce() -> numr::error::Result<Tensor<R>>,
    ) -> Result<(f64, f64), SoilError>
    where
        R: Runtime<DType = DType>,
        R::Client: BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + TypeConversionOps<R>
            + UnaryOps<R>,
    {
        if self.count == 0 {
            return Ok((0.0, 0.0));
        }
        let zero = Tensor::<R>::zeros(depths.shape(), DType::F64, depths.device())?;
        check_tensor_finite(client, depths, "adjustment depth")?;
        reject(
            client,
            &client.lt(depths, &zero)?,
            "adjustment depth is negative",
        )?;
        // Keep the scalar path's original dispatch and operation order.
        let volumes = multiply()?;
        check_tensor_finite(client, &volumes, "adjustment volume")?;
        let positive = client.gt(depths, &zero)?;
        reject(
            client,
            &client.mul(&positive, &client.eq(&volumes, &zero)?)?,
            "positive adjustment volume underflows. Rescale cell area or source inputs",
        )?;
        let bounds = client.add_scalar(
            &client.mul_scalar(&client.abs(&volumes)?, f64::EPSILON)?,
            f64::from_bits(1),
        )?;
        let upper = client.where_cond(&positive, &client.add(&volumes, &bounds)?, &zero)?;
        check_tensor_finite(client, &upper, "outward adjustment volume")?;
        let reported = nonnegative(
            client.sum(&volumes, &[], false)?.item::<f64>()?,
            "reported adjustment sum",
        )?;
        let sum = nonnegative(
            client.sum(&upper, &[], false)?.item::<f64>()?,
            "outward adjustment sum",
        )?;
        if sum == 0.0 {
            return Ok((reported, 0.0));
        }
        let quotient = finite(sum / self.denominator, "adjustment reduction quotient")?;
        let charge = finite(
            quotient + rounding_bound(quotient)?,
            "adjustment reduction charge",
        )?;
        Ok((reported, charge))
    }
}

pub(crate) fn adjustment_volume_totals<R>(
    client: &R::Client,
    depths: &Tensor<R>,
    area: f64,
) -> Result<(f64, f64), SoilError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    AdjustmentReduction::new(depths.numel())?.totals(client, depths, area)
}

fn finite(value: f64, quantity: &str) -> Result<f64, SoilError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SoilError::numerical(format!(
            "{quantity}={value}: require finite source-admission arithmetic. Rescale inputs"
        )))
    }
}

fn nonnegative(value: f64, quantity: &str) -> Result<f64, SoilError> {
    finite(value, quantity)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(SoilError::numerical(format!(
            "{quantity}={value}: require nonnegative source-admission arithmetic"
        )))
    }
}

fn check_tensor_finite<R>(
    client: &R::Client,
    value: &Tensor<R>,
    quantity: &str,
) -> Result<(), SoilError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let invalid = client.maximum(
        &client.cast(&client.isnan(value)?, value.dtype())?,
        &client.cast(&client.isinf(value)?, value.dtype())?,
    )?;
    reject(
        client,
        &invalid,
        &format!("{quantity} contains nonfinite values. Rescale source inputs"),
    )
}

fn reject<R>(client: &R::Client, mask: &Tensor<R>, context: &str) -> Result<(), SoilError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    if let Some(index) = rejected_lane(client, mask)? {
        return Err(SoilError::numerical(context).at_cell(index));
    }
    Ok(())
}

fn rejected_lane<R>(client: &R::Client, mask: &Tensor<R>) -> Result<Option<usize>, SoilError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    if client.any(mask, &[], false)?.item::<f64>()? == 0.0 {
        return Ok(None);
    }
    let flags = mask.try_to_vec::<f64>()?;
    let index = flags
        .iter()
        .position(|flag| *flag != 0.0)
        .ok_or_else(|| SoilError::numerical("source rejection mask contradicts its reduction"))?;
    Ok(Some(index))
}

#[cfg(test)]
#[path = "admission/variable_areas_tests.rs"]
mod variable_areas_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuClient;
    use numr::runtime::cpu::{CpuDevice, ParallelismConfig};

    #[test]
    fn zero_charge_preserves_bits_and_tiny_charge_advances() -> Result<(), SoilError> {
        assert_eq!(
            add_adjustment_charge(-0.0, 0.0)?.to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(1.0 + f64::from_bits(1), 1.0);
        assert!(add_adjustment_charge(1.0, f64::from_bits(1))? > 1.0);
        assert!(add_adjustment_charge(f64::MAX, 1.0).is_err());
        Ok(())
    }

    #[test]
    fn reduction_charges_positive_volumes_and_preserves_zero() -> Result<(), SoilError> {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        for count in [1, 31, 32, 33, 63, 64, 65] {
            let depths = Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
                &vec![2f64.powi(-70); count],
                &[count],
                &device,
            )?;
            let (reported, charge) = adjustment_volume_totals(&client, &depths, 1.0)?;
            assert_eq!(reported, count as f64 * 2f64.powi(-70));
            assert!(charge > reported);
            let zero =
                Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(&[count], DType::F64, &device)?;
            assert_eq!(adjustment_volume_totals(&client, &zero, 1.0)?, (0.0, 0.0));
        }
        let tiny = Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(
            &[f64::from_bits(1)],
            &[1],
            &device,
        )?;
        assert!(adjustment_volume_totals(&client, &tiny, 0.5).is_err());
        let huge =
            Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(&[f64::MAX], &[1], &device)?;
        assert!(adjustment_volume_totals(&client, &huge, 2.0).is_err());
        Ok(())
    }
}
