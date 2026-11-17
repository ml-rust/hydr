//! Reuse the existing unit-area reduction envelope for external-volume totals.

use super::super::CoupledLevelPoolError;
use super::context::map_error;
use super::{InflowContext, InflowState};
use crate::numerics::outward::AdjustmentReduction;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct InflowIntervalSummary {
    pub volume_m3: f64,
    pub reduction_bound_m3: f64,
    pub assembly_bound_m3: f64,
    pub counter_roundoff_bound_m3: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct InflowCumulativeSummary {
    pub volume_m3: f64,
    pub roundoff_bound_m3: f64,
    pub reduction_bound_m3: f64,
}

pub(super) fn totals<R>(
    client: &R::Client,
    reduction: AdjustmentReduction,
    value: &Tensor<R>,
) -> Result<(f64, f64), CoupledLevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let (reported, _) = reduction.totals(client, value, 1.0).map_err(map_error)?;
    let bound = reduction
        .unit_sum_error_bound(reported)
        .map_err(map_error)?;
    Ok((reported, bound))
}

pub(super) fn charge<R>(
    client: &R::Client,
    reduction: AdjustmentReduction,
    value: &Tensor<R>,
) -> Result<f64, CoupledLevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    Ok(reduction.totals(client, value, 1.0).map_err(map_error)?.1)
}

impl<R: Runtime<DType = DType>> InflowContext<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(crate) fn state_summary(
        &self,
        client: &R::Client,
        state: &InflowState<R>,
    ) -> Result<InflowCumulativeSummary, CoupledLevelPoolError> {
        let (volume_m3, reduction_bound_m3) =
            totals(client, self.boundary_reduction, &state.volume)?;
        Ok(InflowCumulativeSummary {
            volume_m3,
            reduction_bound_m3,
            roundoff_bound_m3: charge(client, self.boundary_reduction, &state.bound)?,
        })
    }
}
