use std::cell::{Cell, RefCell};

use super::paired_sources::PairedSources;
use crate::channel::{
    flow::surveyed::{
        error::{HydraulicError, HydraulicResult, checked},
        work::QueryWork,
    },
    section::{AreaDirection, PhysicalEndpoint, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedRootQuery, CertifiedRootValues},
};

pub(super) fn evaluate<R>(
    sources: &RefCell<&mut PairedSources<'_, R>>,
    invocation_rows: &Tensor<R>,
    chart: &SupportedAreaQuery<R>,
    query: &CertifiedRootQuery<R>,
    cause: &Cell<Option<HydraulicError>>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let calculation = (|| -> HydraulicResult<_> {
        let mut pair = sources.try_borrow_mut().map_err(|_| HydraulicError::Numerical {
            context: "Chart root callback source borrow overlaps another evaluation. Discard the root result".into(),
        })?;
        pair.evaluation
            .work
            .charge(QueryWork::Residual, query.point.numel())?;
        let c = pair.geometry.client();
        let (rows, target) = checked::<R, _>(c, "point chart root callback", |a| {
            // Solvr owners index this fixed invocation, never the parent cache directly.
            let rows = c.index_select(invocation_rows, 0, &query.owner)?;
            let bands = pair.bands.select(a, &rows)?;
            let chart = chart.select(a, &query.owner)?;
            let parameter = a.point(&query.point)?;
            let target = chart.with_parameter_bounds(a, &bands, &parameter)?;
            Ok((rows, target))
        })?;
        let value = pair.endpoint_on_rows(&rows, &target, PhysicalEndpoint::Lower)?;
        checked::<R, _>(pair.geometry.client(), "point chart root residual", |a| {
            let zero = Tensor::zeros(&[], a.dtype(), a.client().device())?;
            if a.client()
                .nonzero(&a.client().le(&value.available, &zero)?)?
                .numel()
                != 0
            {
                return Err(super::chart_admission::unresolved(
                    "Chart root residual is unavailable. Supply two wet sources and a supported chart",
                ));
            }
            match chart.query.direction {
                AreaDirection::Add => Ok(CertifiedRootValues {
                    lower: value.physical.lower().clone(),
                    upper: value.physical.upper().clone(),
                }),
                AreaDirection::Subtract => Ok(CertifiedRootValues {
                    lower: a.client().mul_scalar(value.physical.upper(), -1.0)?,
                    upper: a.client().mul_scalar(value.physical.lower(), -1.0)?,
                }),
            }
        })
    })();
    calculation.map_err(|error| {
        let first = match cause.take() {
            Some(first) => first,
            None => error,
        };
        cause.set(Some(first));
        numr::error::Error::InvalidArgument {
            arg: "point chart root residual",
            reason: "Hydraulic callback rejected. Read the retained typed cause".into(),
        }
    })
}
