use std::cell::{Cell, RefCell};

use super::paired_sources::{PairedSources, SelectedRows};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult, checked},
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::Interval,
    optimize::{CertifiedRootDomainQuery, CertifiedRootMappedValues, CertifiedRootTolerance},
};

pub(super) fn certificates<R>(
    parent: &PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
) -> HydraulicResult<(Interval<R>, Interval<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let selected = parent.select_rows(rows, query)?;
    checked::<R, _>(
        parent.geometry.client(),
        "wet root physical depth certificates",
        |a| physical(a, parent, &selected, query),
    )
}

fn physical<R>(
    a: &mut Arithmetic<'_, R>,
    parent: &PairedSources<'_, R>,
    selected: &SelectedRows<R>,
    query: &SupportedAreaQuery<R>,
) -> HydraulicResult<(Interval<R>, Interval<R>)>
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
    let state = parent
        .geometry
        .recover_query(a, &selected.bands, query)?
        .state;
    let normalized = a.div_positive_interval(&state.depth, &selected.bands.cap_depth)?;
    let zero = Tensor::zeros(normalized.lower().shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(normalized.lower().shape(), a.dtype(), c.device())?;
    let cap_band = c
        .sub_scalar(selected.boundaries.cap_indices(), 1.0)?
        .unsqueeze(1)?;
    let deficit_lower = c.gather(query.below_end.lower(), 1, &cap_band)?.flatten()?;
    let deficit_upper = c.gather(query.below_end.upper(), 1, &cap_band)?.flatten()?;
    let exact_cap = c.mul(&c.eq(&deficit_lower, &zero)?, &c.eq(&deficit_upper, &zero)?)?;
    // Whole-image support bounds normalized depth by [0,1]. Only proved zero deficit establishes one exactly.
    let normalized = a.bounds(
        &c.where_cond(&exact_cap, &one, &c.maximum(normalized.lower(), &zero)?)?,
        &c.where_cond(&exact_cap, &one, &c.minimum(normalized.upper(), &one)?)?,
    )?;
    Ok((state.depth, normalized))
}

/// Share Solvr's checked inward tolerance predicate without opening another arithmetic scope.
pub(super) fn satisfied<R>(
    a: &mut Arithmetic<'_, R>,
    normalized: &Interval<R>,
    tolerance: &CertifiedRootTolerance,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    crate::channel::flow::surveyed::root_calls::resolve_root(
        tolerance.satisfied(a, normalized.lower(), normalized.upper()),
        None,
    )
}

pub(super) fn evaluate<R>(
    sources: &RefCell<&mut PairedSources<'_, R>>,
    invocation_rows: &Tensor<R>,
    chart: &SupportedAreaQuery<R>,
    query: &CertifiedRootDomainQuery<R>,
    cause: &Cell<Option<HydraulicError>>,
) -> numr::error::Result<CertifiedRootMappedValues<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let calculation = (|| -> HydraulicResult<_> {
        let pair = sources
            .try_borrow()
            .map_err(|_| HydraulicError::Numerical {
                context:
                    "Wet depth mapping overlaps another source evaluation. Discard the root result"
                        .into(),
            })?;
        let c = pair.geometry.client();
        let (rows, target) = checked::<R, _>(c, "wet depth mapped domain", |a| {
            let rows = c.index_select(invocation_rows, 0, &query.owner)?;
            let bands = pair.bands.select(a, &rows)?;
            let chart = chart.select(a, &query.owner)?;
            let parameter = a.bounds(&query.lower, &query.upper)?;
            Ok((rows, chart.with_parameter_bounds(a, &bands, &parameter)?))
        })?;
        let (_, normalized) = certificates(&pair, &rows, &target)?;
        Ok(CertifiedRootMappedValues {
            lower: normalized.lower().clone(),
            upper: normalized.upper().clone(),
        })
    })();
    calculation.map_err(|error| {
        let first = cause.take().unwrap_or(error);
        cause.set(Some(first));
        numr::error::Error::InvalidArgument {
            arg: "wet depth mapping",
            reason: "Hydraulic mapping rejected. Read the retained typed cause".into(),
        }
    })
}
