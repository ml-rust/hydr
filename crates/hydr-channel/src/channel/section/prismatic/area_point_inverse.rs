use super::{
    area_point_input::{Inputs, difference},
    area_point_query::PrismaticAreaPointBandCoordinates,
    area_query::inverse::{self as area_query_inverse, LocalInverse},
    band_geometry::{BandRows, masked},
    context::Context,
    query_input::{Arithmetic, QueryResult, reject},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ReduceOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use solvr::common::arithmetic::Interval;

#[cfg(test)]
struct FormulaCase {
    id: &'static str,
    slope: u64,
    area: u64,
    depth: [u64; 2],
}

#[cfg(test)]
#[path = "../../../../tests/prismatic_area_points/formula_data.rs"]
mod formula_data;

pub(super) fn recover<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    inputs: &Inputs<R>,
) -> QueryResult<PrismaticAreaPointBandCoordinates<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = &context.source.zero;
    let count = inputs.owners.numel();
    let valid = c.mul(&rows.valid, &inputs.wet.reshape(&[count, 1])?)?;
    let high = c.where_cond(&valid, &inputs.areas.high().reshape(&[count, 1])?, zero)?;
    let low = c.where_cond(&valid, &inputs.areas.low().reshape(&[count, 1])?, zero)?;
    let radius = c.mul_scalar(&valid, 0.0)?;
    let point = a.expansion_bounds(&high, &low, &radius)?;
    let start_area = masked(a, &valid, &rows.start_area, zero)?;
    let end_area = masked(a, &valid, &rows.end_area, zero)?;
    let lower_delta = difference(a, &point, &start_area, false)?;
    let upper_delta = difference(a, &point, &end_area, true)?;
    // A candidate represents a physical state inside this band's closed support.
    let candidate = c.mul(
        &valid,
        &c.mul(
            &c.ge(lower_delta.upper(), zero)?,
            &c.ge(upper_delta.upper(), zero)?,
        )?,
    )?;
    let LocalInverse {
        candidate,
        area_above_start: lower_delta,
        area_below_end: upper_delta,
        depth_above_start: h,
        depth_below_end: e,
    } = area_query_inverse::recover(a, context, rows, &candidate, &lower_delta, &upper_delta)?;
    let missing = c.mul(&inputs.wet, &c.eq(&c.max(&candidate, &[1], false)?, zero)?)?;
    reject(
        context,
        &inputs.owners,
        &missing,
        true,
        "no band encloses the requested area point. Check profile conditioning",
    )?;
    let dry_candidate = c.mul(
        &rows.valid,
        &c.mul(
            &inputs.dry.reshape(&[count, 1])?,
            &c.mul(
                &c.eq(rows.start_depth.lower(), zero)?,
                &c.eq(rows.start_depth.upper(), zero)?,
            )?,
        )?,
    )?;
    // Dry geometry uses depth zero directly, without an inverse width quotient.
    let area_above_start = masked(a, &candidate, &lower_delta, zero)?;
    let depth_above_start = masked(a, &candidate, &h, zero)?;
    let wet_area_below_end = masked(a, &candidate, &upper_delta, zero)?;
    let wet_depth_below_end = masked(a, &candidate, &e, zero)?;
    let area_below_end = a.bounds(
        &c.where_cond(
            &dry_candidate,
            rows.end_area.lower(),
            wet_area_below_end.lower(),
        )?,
        &c.where_cond(
            &dry_candidate,
            rows.end_area.upper(),
            wet_area_below_end.upper(),
        )?,
    )?;
    let depth_below_end = a.bounds(
        &c.where_cond(
            &dry_candidate,
            rows.end_depth.lower(),
            wet_depth_below_end.lower(),
        )?,
        &c.where_cond(
            &dry_candidate,
            rows.end_depth.upper(),
            wet_depth_below_end.upper(),
        )?,
    )?;
    Ok(PrismaticAreaPointBandCoordinates {
        area_above_start,
        area_below_end,
        depth_above_start,
        depth_below_end,
        candidate: c.maximum(&candidate, &dry_candidate)?,
    })
}

pub(super) fn absolute_depth<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    coordinates: &PrismaticAreaPointBandCoordinates<R>,
) -> QueryResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = &context.source.zero;
    let start = masked(a, &coordinates.candidate, &rows.start_depth, zero)?;
    let end = masked(a, &coordinates.candidate, &rows.end_depth, zero)?;
    let lower = a.add(&start, &coordinates.depth_above_start)?;
    let upper = a.sub(&end, &coordinates.depth_below_end)?;
    Ok(a.bounds(
        &c.maximum(lower.lower(), upper.lower())?,
        &c.minimum(lower.upper(), upper.upper())?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::super::area_query::inverse::pointed_depth;
    use super::formula_data;
    use super::*;
    use crate::channel::section::prismatic::query_input::QueryError;
    use numr::tensor::Tensor;
    use solvr::common::arithmetic::with_interval_arithmetic;

    type Result = std::result::Result<(), Box<dyn std::error::Error>>;

    fn run<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        for case in formula_data::FORMULAS {
            let area =
                Tensor::<R>::from_slice(&[f64::from_bits(case.area)], &[1], client.device())?;
            let slope =
                Tensor::<R>::from_slice(&[f64::from_bits(case.slope)], &[1], client.device())?;
            let depth = with_interval_arithmetic::<R, _, _, _>(
                &client,
                DType::F64,
                "pointed inverse range control",
                |a| {
                    let area = a.point(&area)?;
                    let slope = a.point(&slope)?;
                    Ok(pointed_depth(a, &area, &slope))
                },
            )?;
            let depth = match depth {
                Ok(depth) => depth,
                Err(QueryError::Geometry(error)) => return Err(error.into()),
                Err(QueryError::Arithmetic(error)) => return Err(error.into()),
            };
            let lower = depth.lower().item::<f64>()?;
            let upper = depth.upper().item::<f64>()?;
            let reference = case.depth.map(f64::from_bits);
            assert!(lower.is_finite() && upper.is_finite(), "{}", case.id);
            assert!(
                0.0 < lower && lower <= reference[0] && reference[1] <= upper,
                "{}",
                case.id
            );
            assert!(upper - lower <= reference[0] * 1e-10, "{}", case.id);
        }
        Ok(())
    }

    #[test]
    fn cpu_pointed_inverse_retains_finite_tiny_slope_depth() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        run::<CpuRuntime>(
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_pointed_inverse_retains_finite_tiny_slope_depth() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
