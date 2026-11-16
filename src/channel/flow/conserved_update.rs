//! Two-part conserved updates through certified expansion arithmetic.

use super::{
    ChannelError, conserved_point::ConservedPoint, surveyed_transport::error::arithmetic_error,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{
    ArithmeticResult, CertifiedExpansion, IntervalContext, with_interval_arithmetic,
};

/// Client operations that certified expansion arithmetic requires.
pub(crate) trait ExpansionClient<R: Runtime<DType = DType>>:
    TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + ScalarOps<R> + TwoSumOps<R>
{
}

impl<R, C> ExpansionClient<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + ScalarOps<R> + TwoSumOps<R>,
{
}

pub(super) type Arithmetic<'a, R> = IntervalContext<'a, R, <R as Runtime>::Client>;

/// Expansion center `(high, low)` and its absolute error radius.
#[derive(Clone, Debug)]
pub(super) struct Parts<R: Runtime<DType = DType>> {
    pub high: Tensor<R>,
    pub low: Tensor<R>,
    pub bound: Tensor<R>,
}

/// Shared face points and the cell transfers formed from them.
///
/// Face `bound` holds the product radius. Transfer `bound` holds the subtraction radius.
pub(super) struct SurveyedFaces<R: Runtime<DType = DType>> {
    pub volume: Parts<R>,
    pub momentum: Parts<R>,
    pub water_delta: Parts<R>,
    pub momentum_delta: Parts<R>,
}

/// Run one checked expansion block and map its arithmetic error.
pub(super) fn expanded<R, T>(
    client: &R::Client,
    stage: &'static str,
    build: impl for<'a> FnOnce(&mut Arithmetic<'a, R>) -> ArithmeticResult<T>,
) -> Result<T, ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    with_interval_arithmetic::<R, _, _, _>(client, DType::F64, stage, build)
        .map_err(arithmetic_error)
}

/// Exact expansion of a normalized `(high, low)` pair with zero radius.
pub(super) fn exact<R>(
    a: &mut Arithmetic<'_, R>,
    high: &Tensor<R>,
    low: &Tensor<R>,
) -> ArithmeticResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let zero = Tensor::zeros(high.shape(), DType::F64, a.client().device())?;
    exact_over(a, high, low, &zero)
}

/// Exact expansion that reuses a caller-owned zero radius of the same shape.
pub(super) fn exact_over<R>(
    a: &mut Arithmetic<'_, R>,
    high: &Tensor<R>,
    low: &Tensor<R>,
    zero: &Tensor<R>,
) -> ArithmeticResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    a.expansion_bounds(high, low, zero)
}

pub(super) fn bounded<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Parts<R>,
) -> ArithmeticResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    a.expansion_bounds(&value.high, &value.low, &value.bound)
}

/// Selected center of an expansion. Its radius is recorded apart.
pub(super) fn center<R: Runtime<DType = DType>>(
    value: &CertifiedExpansion<R>,
) -> ConservedPoint<R> {
    ConservedPoint {
        high: value.high().clone(),
        low: value.low().clone(),
    }
}

fn parts<R: Runtime<DType = DType>>(value: &CertifiedExpansion<R>) -> Parts<R> {
    Parts {
        high: value.high().clone(),
        low: value.low().clone(),
        bound: value.abs_error().clone(),
    }
}

/// Upper bound on the magnitude of every value inside the expansion.
pub(super) fn magnitude<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
) -> ArithmeticResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let interval = a.expansion_interval(value)?;
    let c = a.client();
    Ok(c.maximum(&c.abs(interval.lower())?, &c.abs(interval.upper())?)?)
}

/// Difference of the left and right face points of every cell.
pub(super) fn transfer<R>(
    a: &mut Arithmetic<'_, R>,
    face: &CertifiedExpansion<R>,
    count: usize,
) -> ArithmeticResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let zero = Tensor::zeros(&[count], DType::F64, a.client().device())?;
    let incoming = exact_over(
        a,
        &face.high().narrow(0, 0, count)?,
        &face.low().narrow(0, 0, count)?,
        &zero,
    )?;
    let outgoing = exact_over(
        a,
        &face.high().narrow(0, 1, count)?,
        &face.low().narrow(0, 1, count)?,
        &zero,
    )?;
    a.expansion_sub(&incoming, &outgoing)
}

/// Integrate face rates over `dt` and form cell transfers from the selected face points.
pub(super) fn surveyed_faces<R>(
    client: &R::Client,
    water_rate: &Tensor<R>,
    momentum_rate: &Tensor<R>,
    dt: f64,
    count: usize,
) -> Result<SurveyedFaces<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    expanded(client, "surveyed shared face impulses", |a| {
        let duration = Tensor::full_scalar(water_rate.shape(), DType::F64, dt, client.device())?;
        let duration = a.expansion_point(&duration)?;
        let mut product = |rate: &Tensor<R>| -> ArithmeticResult<_> {
            let rate = a.expansion_point(rate)?;
            let raw = a.expansion_mul(&rate, &duration)?;
            // The center becomes the shared face point. Its radius is charged separately.
            let point = exact(a, raw.high(), raw.low())?;
            let transfer = transfer(a, &point, count)?;
            Ok((parts(&raw), parts(&transfer)))
        };
        let (volume, water_delta) = product(water_rate)?;
        let (momentum, momentum_delta) = product(momentum_rate)?;
        Ok(SurveyedFaces {
            volume,
            momentum,
            water_delta,
            momentum_delta,
        })
    })
}

/// Add two expansions and select the center once.
///
/// The returned bound is the radius of this sum. The selected point has radius zero.
pub(super) fn select_sum<R>(
    client: &R::Client,
    first: &Parts<R>,
    second: &Parts<R>,
    stage: &'static str,
) -> Result<(ConservedPoint<R>, Tensor<R>), ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    expanded(client, stage, |a| {
        let first = bounded(a, first)?;
        let second = bounded(a, second)?;
        let sum = a.expansion_add(&first, &second)?;
        Ok((center(&sum), sum.abs_error().clone()))
    })
}

/// Lower bound of each normalized cell value.
///
/// A negative low part lowers the bound below `high`, so it never adds donor water.
pub(super) fn lower_bound<R>(
    client: &R::Client,
    point: &ConservedPoint<R>,
) -> Result<Tensor<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    expanded(client, "surveyed donor water lower bound", |a| {
        let value = exact(a, &point.high, &point.low)?;
        Ok(a.expansion_interval(&value)?.lower().clone())
    })
}

/// Upper bound of the water each cell sends through its two faces.
pub(super) fn outgoing_upper<R>(
    client: &R::Client,
    face_high: &Tensor<R>,
    face_low: &Tensor<R>,
    count: usize,
) -> Result<Tensor<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let c = client;
    expanded(client, "surveyed outgoing shared face volumes", |a| {
        let zero = Tensor::zeros(&[count], DType::F64, c.device())?;
        let right_high = face_high.narrow(0, 1, count)?;
        let right_low = face_low.narrow(0, 1, count)?;
        let left_high = face_high.narrow(0, 0, count)?;
        let left_low = face_low.narrow(0, 0, count)?;
        // A normalized point has the sign of its high part.
        let right_out = c.gt(&right_high, &zero)?;
        let left_out = c.lt(&left_high, &zero)?;
        let right = exact_over(
            a,
            &c.where_cond(&right_out, &right_high, &zero)?,
            &c.where_cond(&right_out, &right_low, &zero)?,
            &zero,
        )?;
        let left = exact_over(
            a,
            &c.where_cond(&left_out, &c.neg(&left_high)?, &zero)?,
            &c.where_cond(&left_out, &c.neg(&left_low)?, &zero)?,
            &zero,
        )?;
        let sum = a.expansion_add(&right, &left)?;
        Ok(a.expansion_interval(&sum)?.upper().clone())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    fn tensor<R>(client: &R::Client, values: &[f64]) -> TestResult<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        Ok(Tensor::from_slice(
            values,
            &[values.len()],
            client.device(),
        )?)
    }

    fn values<R: Runtime<DType = DType>>(value: &Tensor<R>) -> TestResult<Vec<f64>> {
        Ok(value.try_to_vec::<f64>()?)
    }

    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let sum = a + b;
        let virtual_b = sum - a;
        (sum, (a - (sum - virtual_b)) + (b - virtual_b))
    }

    fn sub_ulp_impulse<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        let tensor = |values: &[f64]| tensor::<R>(&client, values);
        let impulse = -4791380824461455.0 * 2f64.powi(-104);
        // The outgoing right face carries the impulse away from the cell.
        let faces = surveyed_faces::<R>(
            &client,
            &tensor(&[0.0, -impulse])?,
            &tensor(&[0.0, 0.0])?,
            1.0,
            1,
        )?;
        assert_eq!(values(&faces.volume.high)?, vec![0.0, -impulse]);
        assert_eq!(values(&faces.volume.bound)?, vec![0.0, 0.0]);
        assert_eq!(values(&faces.water_delta.high)?, vec![impulse]);
        assert_eq!(35.0 + impulse, 35.0);
        let state = Parts {
            high: tensor(&[35.0])?,
            low: tensor(&[0.0])?,
            bound: tensor(&[0.0])?,
        };
        let (point, bound) = select_sum::<R>(&client, &state, &faces.water_delta, "test")?;
        assert_eq!(values(&point.high)?[0].to_bits(), 35f64.to_bits());
        assert_eq!(values(&point.low)?[0].to_bits(), impulse.to_bits());
        assert_eq!(values(&bound)?, vec![0.0]);
        let lower = lower_bound::<R>(&client, &point)?;
        assert_eq!(values(&lower)?, vec![f64::from_bits(35f64.to_bits() - 1)]);
        let outgoing = outgoing_upper::<R>(&client, &faces.volume.high, &faces.volume.low, 1)?;
        assert_eq!(values(&outgoing)?, vec![-impulse]);
        Ok(())
    }

    fn fused_residual<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        let tensor = |values: &[f64]| tensor::<R>(&client, values);
        let rate = 1.0 + f64::EPSILON;
        let faces = surveyed_faces::<R>(
            &client,
            &tensor(&[rate, 0.0])?,
            &tensor(&[0.0, 0.0])?,
            rate,
            1,
        )?;
        let (high, low) = (
            values(&faces.volume.high)?[0],
            values(&faces.volume.low)?[0],
        );
        assert_eq!(high, rate * rate);
        assert_eq!(low, rate.mul_add(rate, -(rate * rate)));
        assert!(values(&faces.volume.bound)?[0] >= 0.0);
        Ok(())
    }

    fn selection_bound<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        let tensor = |values: &[f64]| tensor::<R>(&client, values);
        // Four exactly separated terms cannot fit in two parts.
        let (a, b, c, d) = (1.0, 2f64.powi(-60), 2f64.powi(-120), 2f64.powi(-180));
        let first = Parts {
            high: tensor(&[a])?,
            low: tensor(&[c])?,
            bound: tensor(&[0.0])?,
        };
        let second = Parts {
            high: tensor(&[b])?,
            low: tensor(&[d])?,
            bound: tensor(&[0.0])?,
        };
        let (point, bound) = select_sum::<R>(&client, &first, &second, "test")?;
        let (high, low) = (values(&point.high)?[0], values(&point.low)?[0]);
        let (oracle_high, oracle_low) = two_sum(a, b);
        assert_eq!(two_sum(high, low), (high, low));
        assert_eq!((high, low), (oracle_high, oracle_low));
        let bound = values(&bound)?[0];
        assert!(bound >= c + d && bound <= 2.0 * (c + d), "bound={bound:e}");
        // The selected point carries no radius into the next update.
        let next = Parts {
            high: point.high.clone(),
            low: point.low.clone(),
            bound: tensor(&[0.0])?,
        };
        let zero = Parts {
            high: tensor(&[0.0])?,
            low: tensor(&[0.0])?,
            bound: tensor(&[0.0])?,
        };
        let (again, again_bound) = select_sum::<R>(&client, &next, &zero, "test")?;
        assert_eq!(values(&again.high)?, vec![high]);
        assert_eq!(values(&again.low)?, vec![low]);
        assert_eq!(values(&again_bound)?, vec![0.0]);
        Ok(())
    }

    fn cpu() -> CpuClient {
        CpuClient::new(CpuDevice::new())
    }

    #[test]
    fn cpu_cap_cell_retains_a_sub_ulp_shared_impulse() -> TestResult {
        sub_ulp_impulse::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_face_product_keeps_its_fused_residual() -> TestResult {
        fused_residual::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_compress_discards_a_term_into_one_selection_bound() -> TestResult {
        selection_bound::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_cap_cell_retains_a_sub_ulp_shared_impulse() -> TestResult {
            sub_ulp_impulse::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_face_product_keeps_its_fused_residual() -> TestResult {
            fused_residual::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_compress_discards_a_term_into_one_selection_bound() -> TestResult {
            selection_bound::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }
    }
}
