use super::super::{
    paired_sources::PairedSources, source_binding::SourcePanelLevel, wave_curve_types::CurveValue,
};
use super::SourcePanelValues;
use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};

impl<R: Runtime<DType = DType>> SourcePanelValues<R> {
    pub(in crate::channel::flow::surveyed_points) fn checked_side(
        &self,
        sources: &PairedSources<'_, R>,
        side: usize,
        parents: &Tensor<R>,
    ) -> HydraulicResult<&CurveValue<R>>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        sources.check_root_witness(self.level.witness())?;
        let Some(rows) = self.parent_rows.get(side) else {
            return Err(HydraulicError::Numerical {
                context: format!("Source panel side={side} is absent. Supply side 0 or 1"),
            });
        };
        if self.side.get(side) != Some(&side)
            || parents.shape() != rows.shape()
            || parents.dtype() != DType::I64
            || !parents.device().is_same(rows.device())
        {
            return Err(HydraulicError::Numerical { context: "Source panel parent/side provenance differs. Preserve the checked source request".into() });
        }
        if sources
            .geometry
            .client()
            .nonzero(&sources.geometry.client().ne(parents, rows)?)?
            .numel()
            != 0
        {
            return Err(HydraulicError::Numerical {
                context:
                    "Source panel parent identities differ. Preserve the checked source request"
                        .into(),
            });
        }
        self.values
            .get(side)
            .ok_or_else(|| HydraulicError::Numerical {
                context: "Source panel value side is absent. Preserve the checked source pair"
                    .into(),
            })
    }
}

pub(super) fn owned_values<R>(
    c: &R::Client,
    level: &SourcePanelLevel,
    rows: [&Tensor<R>; 2],
    values: [CurveValue<R>; 2],
) -> HydraulicResult<SourcePanelValues<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + RuntimeClient<R>,
{
    let copy = |rows: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        let owned = Tensor::empty(rows.shape(), DType::I64, c.device())?;
        c.copy_into(&owned, rows)?;
        Ok(owned)
    };
    Ok(SourcePanelValues {
        values,
        level: level.clone(),
        parent_rows: [copy(rows[0])?, copy(rows[1])?],
        side: [0, 1],
    })
}
