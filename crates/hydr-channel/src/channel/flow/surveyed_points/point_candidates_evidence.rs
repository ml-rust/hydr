use super::{point_root_session::PointRootSession, wet_root_types::WetRootEvidence};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn snapshot<R>(
    session: &PointRootSession<'_, '_, R>,
    demand: &Tensor<R>,
) -> HydraulicResult<(Tensor<R>, Vec<WetRootEvidence<R>>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = session.sources.geometry.client();
    checked::<R, _>(c, "point candidate immutable wet root evidence", |a| {
        let mut depth = Tensor::full_scalar(
            session.invocation_rows.shape(),
            DType::I64,
            super::wet_depth_types::WetDepthStatus::Unresolved as i64 as f64,
            c.device(),
        )?;
        for domain in &session.wet.domains {
            depth = c.index_put(&depth, 0, &domain.invocation_owner, &domain.status)?;
        }
        let mut evidence = Vec::new();
        for item in &session.wet.evidence {
            let rows = c.nonzero(&c.index_select(demand, 0, &item.invocation_owner)?)?;
            let rows = rows.reshape(&[rows.numel()])?;
            if rows.numel() == 0 {
                continue;
            }
            evidence.push(WetRootEvidence {
                invocation_owner: c.index_select(&item.invocation_owner, 0, &rows)?,
                parent_rows: c.index_select(&item.parent_rows, 0, &rows)?,
                kind: item.kind,
                value: super::wet_root_state::selected_value(a, &item.value, &rows)?,
            });
        }
        Ok((depth, evidence))
    })
}
