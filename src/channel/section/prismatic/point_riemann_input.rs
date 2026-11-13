use super::{batch::PrismaticSectionBatch, context::Context, query_input};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult, checked};
use crate::channel::flow::surveyed_points::types::CapturedPointInput;
use crate::channel::section::CrossSectionError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

impl<R> PrismaticSectionBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn capture_riemann_points(
        &self,
        owners: &Tensor<R>,
        left_area: &CertifiedExpansion<R>,
        left_discharge: &CertifiedExpansion<R>,
        right_area: &CertifiedExpansion<R>,
        right_discharge: &CertifiedExpansion<R>,
    ) -> HydraulicResult<Arc<CapturedPointInput<R>>> {
        for (name, point) in [
            ("left area_m2", left_area),
            ("left discharge_m3_per_s", left_discharge),
            ("right area_m2", right_area),
            ("right discharge_m3_per_s", right_discharge),
        ] {
            for (component, value) in [
                ("high", point.high()),
                ("low", point.low()),
                ("radius", point.abs_error()),
            ] {
                query_input::metadata(&self.context, owners, value)
                    .map_err(|error| input_error(format!("{name} {component}: {error}")))?;
            }
        }
        let count = owners.numel();
        u32::try_from(count).map_err(|_| {
            input_error(format!(
                "input count={count} exceeds U32 indexing. Split the batch"
            ))
        })?;
        count
            .checked_mul(15)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                input_error(format!(
                    "input count={count} overflows captured source bytes. Split the batch"
                ))
            })?;
        let context = &self.context;
        checked::<R, _>(
            &context.source.client,
            "surveyed point input capture",
            |a| {
                let c = a.client();
                let captured_owners = Tensor::empty(owners.shape(), DType::I64, c.device())?;
                c.copy_into(&captured_owners, owners)?;
                check_owners(context, &captured_owners)?;
                let left_area =
                    capture_point(a, context, &captured_owners, left_area, "left area_m2")?;
                let left_discharge = capture_point(
                    a,
                    context,
                    &captured_owners,
                    left_discharge,
                    "left discharge_m3_per_s",
                )?;
                let right_area =
                    capture_point(a, context, &captured_owners, right_area, "right area_m2")?;
                let right_discharge = capture_point(
                    a,
                    context,
                    &captured_owners,
                    right_discharge,
                    "right discharge_m3_per_s",
                )?;
                let left_dry = check_state(
                    context,
                    &captured_owners,
                    &left_area,
                    &left_discharge,
                    "left",
                )?;
                let right_dry = check_state(
                    context,
                    &captured_owners,
                    &right_area,
                    &right_discharge,
                    "right",
                )?;
                Ok(Arc::new(CapturedPointInput {
                    owners: captured_owners,
                    left_area,
                    left_discharge,
                    right_area,
                    right_discharge,
                    left_dry,
                    right_dry,
                }))
            },
        )
    }
}

fn check_owners<R>(context: &Context<R>, owners: &Tensor<R>) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if owners.numel() == 0 {
        return Ok(());
    }
    let c = &context.source.client;
    let count = i64::try_from(context.source.definitions.len())
        .map_err(|_| input_error("section count exceeds I64 owners. Split the section batch"))?;
    let zero = Tensor::from_slice(&[0_i64], &[], c.device())?;
    let limit = Tensor::from_slice(&[count], &[], c.device())?;
    let invalid = c.maximum(&c.lt(owners, &zero)?, &c.ge(owners, &limit)?)?;
    if let Some(query) = context
        .source
        .rejected_lane(&c.cast(&invalid, DType::F64)?)?
    {
        let owner = owners.narrow(0, query, 1)?.item::<i64>()?;
        return Err(input_error(format!(
            "owner={owner}, query={query}, section_count={count}: supply an owner in 0..{count}"
        )));
    }
    Ok(())
}

fn capture_point<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    owners: &Tensor<R>,
    point: &CertifiedExpansion<R>,
    name: &str,
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if owners.numel() != 0 {
        let c = a.client();
        let mut invalid = Tensor::zeros(owners.shape(), DType::F64, c.device())?;
        for value in [point.high(), point.low(), point.abs_error()] {
            let nonfinite = c.maximum(
                &c.cast(&c.isnan(value)?, DType::F64)?,
                &c.cast(&c.isinf(value)?, DType::F64)?,
            )?;
            invalid = c.maximum(&invalid, &nonfinite)?;
        }
        let invalid = c.maximum(&invalid, &c.ne(point.abs_error(), &context.source.zero)?)?;
        reject_point(
            context,
            owners,
            &invalid,
            point,
            name,
            "components must be finite and radius must be zero. Supply an exact finite conserved point",
        )?;
    }
    Ok(a.expansion_bounds(point.high(), point.low(), point.abs_error())?)
}

fn check_state<R>(
    context: &Context<R>,
    owners: &Tensor<R>,
    area: &CertifiedExpansion<R>,
    discharge: &CertifiedExpansion<R>,
    side: &str,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = &context.source.client;
    let zero = &context.source.zero;
    let negative = c.maximum(
        &c.lt(area.high(), zero)?,
        &c.mul(&c.eq(area.high(), zero)?, &c.lt(area.low(), zero)?)?,
    )?;
    let dry = c.mul(&c.eq(area.high(), zero)?, &c.eq(area.low(), zero)?)?;
    if owners.numel() != 0 {
        reject_point(
            context,
            owners,
            &negative,
            area,
            &format!("{side} area_m2"),
            "complete area is negative. Supply a nonnegative area point",
        )?;
        let discharge_nonzero = c.maximum(
            &c.ne(discharge.high(), zero)?,
            &c.ne(discharge.low(), zero)?,
        )?;
        reject_point(
            context,
            owners,
            &c.mul(&dry, &discharge_nonzero)?,
            discharge,
            &format!("{side} discharge_m3_per_s"),
            "dry area has nonzero discharge. Supply exact zero discharge at zero area",
        )?;
    }
    Ok(dry)
}

fn reject_point<R>(
    context: &Context<R>,
    owners: &Tensor<R>,
    invalid: &Tensor<R>,
    point: &CertifiedExpansion<R>,
    name: &str,
    requirement: &str,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if let Some(query) = context.source.rejected_lane(invalid)? {
        let owner = owners.narrow(0, query, 1)?.item::<i64>()?;
        let high = point.high().narrow(0, query, 1)?.item::<f64>()?;
        let low = point.low().narrow(0, query, 1)?.item::<f64>()?;
        let radius = point.abs_error().narrow(0, query, 1)?.item::<f64>()?;
        return Err(input_error(format!(
            "{name} high={high:?}, low={low:?}, radius={radius:?}, query={query}, section_owner={owner}: {requirement}"
        )));
    }
    Ok(())
}

fn input_error(context: impl Into<String>) -> HydraulicError {
    CrossSectionError::InvalidInput {
        section: None,
        context: context.into(),
    }
    .into()
}
