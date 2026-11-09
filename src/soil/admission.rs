//! Source-volume adjustment admission, separate from conservation tolerances.

use super::SoilError;
use crate::cpu::CpuTensor;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::cpu::CpuClient;

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

    pub(crate) fn totals(
        self,
        client: &CpuClient,
        depths: &CpuTensor,
        area: f64,
    ) -> Result<(f64, f64), SoilError> {
        if depths.shape() != [self.count] || !area.is_finite() || area <= 0.0 {
            return Err(SoilError::input(format!(
                "correction shape={:?}, area={area}: require {} flat entries and positive finite area",
                depths.shape(),
                self.count
            )));
        }
        if self.count == 0 {
            return Ok((0.0, 0.0));
        }
        let zero = CpuTensor::zeros(depths.shape(), DType::F64, depths.device())?;
        check_tensor_finite(client, depths, "adjustment depth")?;
        reject(
            client,
            &client.lt(depths, &zero)?,
            "adjustment depth is negative",
        )?;
        let volumes = client.mul_scalar(depths, area)?;
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

pub(crate) fn adjustment_volume_totals(
    client: &CpuClient,
    depths: &CpuTensor,
    area: f64,
) -> Result<(f64, f64), SoilError> {
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

fn check_tensor_finite(
    client: &CpuClient,
    value: &CpuTensor,
    quantity: &str,
) -> Result<(), SoilError> {
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

fn reject(client: &CpuClient, mask: &CpuTensor, context: &str) -> Result<(), SoilError> {
    if client.any(mask, &[], false)?.item::<f64>()? == 0.0 {
        return Ok(());
    }
    let flags = mask.try_to_vec::<f64>()?;
    let index = flags
        .iter()
        .position(|flag| *flag != 0.0)
        .ok_or_else(|| SoilError::numerical("source rejection mask contradicts its reduction"))?;
    Err(SoilError::numerical(context).at_cell(index))
}

#[cfg(test)]
mod tests {
    use super::*;
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
            let depths = CpuTensor::from_slice(&vec![2f64.powi(-70); count], &[count], &device)?;
            let (reported, charge) = adjustment_volume_totals(&client, &depths, 1.0)?;
            assert_eq!(reported, count as f64 * 2f64.powi(-70));
            assert!(charge > reported);
            let zero = CpuTensor::zeros(&[count], DType::F64, &device)?;
            assert_eq!(adjustment_volume_totals(&client, &zero, 1.0)?, (0.0, 0.0));
        }
        let tiny = CpuTensor::from_slice(&[f64::from_bits(1)], &[1], &device)?;
        assert!(adjustment_volume_totals(&client, &tiny, 0.5).is_err());
        let huge = CpuTensor::from_slice(&[f64::MAX], &[1], &device)?;
        assert!(adjustment_volume_totals(&client, &huge, 2.0).is_err());
        Ok(())
    }
}
