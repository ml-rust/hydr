use super::{
    point_root_session::{PointRootSession, invalid},
    wet_depth_types::DepthRefinementState,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn check<R>(session: &PointRootSession<'_, '_, R>) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(
        session.sources.geometry.client(),
        "point candidate complete wet binding admission",
        |a| {
            for domain in &session.wet.domains {
                let (_, expected) =
                    super::dry_vacuum_evidence::owners(a, session, &domain.invocation_owner)?;
                equal(a, &expected, &domain.parent_rows)?;
                if let DepthRefinementState::Deferred { original } = &domain.refinement {
                    equal(a, &domain.invocation_owner, &original.invocation_owner)?;
                    equal(a, &domain.parent_rows, &original.parent_rows)?;
                }
            }
            for domain in &session.wet.domains {
                session
                    .sources
                    .select_rows(&domain.parent_rows, &domain.query)?;
                match &domain.refinement {
                    DepthRefinementState::Deferred { original } => {
                        session
                            .sources
                            .select_rows(&domain.parent_rows, &original.query)?;
                    }
                    DepthRefinementState::Initialized { state } => {
                        session
                            .sources
                            .select_rows(&domain.parent_rows, &state.chart)?;
                    }
                }
            }
            Ok(())
        },
    )
}
fn equal<R>(a: &Arithmetic<'_, R>, actual: &Tensor<R>, expected: &Tensor<R>) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if actual.shape() != expected.shape()
        || actual.dtype() != expected.dtype()
        || a.client()
            .nonzero(&a.client().ne(actual, expected)?)?
            .numel()
            != 0
    {
        return Err(invalid(
            "point candidate wet owner or nested source row binding changed",
        ));
    }
    Ok(())
}
