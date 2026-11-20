use super::{
    paired_sources::PairedSources, source_sonic_numerical::SourceSonicNumericalEvidence,
    source_sonic_types::SonicSide,
};
use crate::channel::{
    flow::surveyed::{
        error::{HydraulicError, HydraulicResult, checked},
        work::QueryWork,
    },
    section::{PhysicalEndpoint, SupportedAreaQuery},
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
use std::cell::{Cell, RefCell};

pub(super) fn evaluate<R>(
    sources: &RefCell<&mut PairedSources<'_, R>>,
    rows: &Tensor<R>,
    chart: &SupportedAreaQuery<R>,
    numerical: &SourceSonicNumericalEvidence<R>,
    side: SonicSide,
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
        let mut sources = sources.try_borrow_mut().map_err(|_| {
            super::point_root_session::invalid("source sonic callback overlaps another evaluation")
        })?;
        sources
            .evaluation
            .work
            .charge(QueryWork::Residual, query.point.numel())?;
        let c = sources.geometry.client();
        let (parents, target, numerical) =
            checked::<R, _>(c, "source sonic callback bound query", |a| {
                let parents = c.index_select(rows, 0, &query.owner)?;
                let selected = sources.select_sources(a, &parents)?;
                let chart = chart.select(a, &query.owner)?;
                Ok((
                    parents,
                    chart.with_parameter_point(a, &selected.bands, &query.point)?,
                    numerical.selected(a, &query.owner)?,
                ))
            })?;
        let value = super::source_sonic_policy::residual(
            &mut sources,
            &parents,
            &target,
            PhysicalEndpoint::Lower,
            side,
            &numerical,
        )?;
        let zero = Tensor::zeros(value.available.shape(), value.available.dtype(), c.device())?;
        if c.nonzero(&c.eq(&value.available, &zero)?)?.numel() != 0 {
            return Err(super::point_root_session::invalid(
                "source sonic residual is unavailable",
            ));
        }
        Ok(CertifiedRootValues {
            lower: value.physical.lower().clone(),
            upper: value.physical.upper().clone(),
        })
    })();
    calculation.map_err(|error| {
        let first = match cause.take() {
            Some(first) => first,
            None => error,
        };
        cause.set(Some(first));
        numr::error::Error::InvalidArgument {
            arg: "source sonic residual",
            reason: "Hydraulic callback rejected. Read the retained typed cause".into(),
        }
    })
}
