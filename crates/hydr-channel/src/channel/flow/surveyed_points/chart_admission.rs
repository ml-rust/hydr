use super::paired_sources::SelectedRows;
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig,
        error::{Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection},
    },
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn budget(
    config: &PrismaticRiemannConfig,
    rows: usize,
    max_points: usize,
    operation: &str,
) -> HydraulicResult<()> {
    if max_points > config.root.max_points {
        return Err(work_limit(format!(
            "{operation} max_points={max_points} exceeds config.root.max_points={}. Supply the remaining configured budget",
            config.root.max_points,
        )));
    }
    let admission = rows.checked_mul(2).ok_or_else(|| {
        work_limit(format!(
            "{operation} admission count 2*rows={rows} overflows usize. Split the invocation",
        ))
    })?;
    if admission > max_points {
        return Err(work_limit(format!(
            "{operation} admission needs {admission} points, max_points={max_points}. Supply sufficient remaining points or split the invocation",
        )));
    }
    Ok(())
}

pub(super) fn supported<R>(
    a: &mut Arithmetic<'_, R>,
    selected: &SelectedRows<R>,
    chart: &SupportedAreaQuery<R>,
) -> HydraulicResult<SupportedAreaQuery<R>>
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
    let zero = Tensor::zeros(&[], a.dtype(), c.device())?;
    if c.nonzero(&c.le(&selected.velocity.both_wet, &zero)?)?
        .numel()
        != 0
    {
        return Err(unresolved(
            "Chart root requires two wet sources. Classify dry sources before solving",
        ));
    }
    let chart = chart.with_parameter_bounds(a, &selected.bands, &chart.query.parameter)?;
    let scale = a.expansion_interval(&chart.query.scale)?;
    if c.nonzero(&c.le(scale.lower(), &zero)?)?.numel() != 0 {
        return Err(unresolved(
            "Chart root scale lacks a strictly positive complete lower bound. Choose a resolved affine chart",
        ));
    }
    Ok(chart)
}

pub(super) fn unresolved(context: &str) -> HydraulicError {
    HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::GeometryUnresolved,
        query: None,
        section_owner: None,
        context: context.into(),
    }
}

fn work_limit(context: String) -> HydraulicError {
    HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::WorkLimit,
        query: None,
        section_owner: None,
        context,
    }
}
