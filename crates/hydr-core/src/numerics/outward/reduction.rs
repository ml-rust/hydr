use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::error::{OutwardError, finite, nonnegative};
use super::lanes::{check_tensor_finite, reject};
use super::rounding_bound;

mod areas;
mod unit_sum;

/// Count-dependent reduction bound, cached by the surface runtime.
#[derive(Debug, Clone, Copy)]
pub struct AdjustmentReduction {
    count: usize,
    denominator: f64,
}

impl AdjustmentReduction {
    #[inline]
    pub fn new(count: usize) -> Result<Self, OutwardError> {
        let additions = count
            .checked_add(7)
            .ok_or_else(|| OutwardError::input("correction reduction count overflows usize"))?;
        if additions as u128 > (1u128 << 53) {
            return Err(OutwardError::input(
                "correction reduction requires N+7 <= 2^53 entries",
            ));
        }
        let x = additions as f64 * f64::EPSILON;
        let d0 = 1.0 - x;
        let denominator = d0 - rounding_bound(d0)?;
        if !(0.0..1.0).contains(&x) || !denominator.is_finite() || denominator <= 0.0 {
            return Err(OutwardError::input(
                "correction reduction denominator is not positive finite. Reduce the grid size",
            ));
        }
        Ok(Self { count, denominator })
    }

    pub fn totals<R>(
        self,
        client: &R::Client,
        depths: &Tensor<R>,
        area: f64,
    ) -> Result<(f64, f64), OutwardError>
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
            return Err(OutwardError::input(format!(
                "correction shape={:?}, area={area}: require {} flat F64 entries and positive finite area",
                depths.shape(),
                self.count
            )));
        }
        self.totals_with_product(client, depths, || client.mul_scalar(depths, area))
    }

    fn totals_with_product<R>(
        self,
        client: &R::Client,
        depths: &Tensor<R>,
        multiply: impl FnOnce() -> numr::error::Result<Tensor<R>>,
    ) -> Result<(f64, f64), OutwardError>
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
        // Multiply only after the depth checks pass, so an invalid depth rejects first.
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

pub fn adjustment_volume_totals<R>(
    client: &R::Client,
    depths: &Tensor<R>,
    area: f64,
) -> Result<(f64, f64), OutwardError>
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

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuClient;
    use numr::runtime::cpu::{CpuDevice, ParallelismConfig};

    #[test]
    fn reduction_charges_positive_volumes_and_preserves_zero() -> Result<(), OutwardError> {
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
