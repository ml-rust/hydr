//! Scalar physical-source summaries without per-element host projection.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;

use super::LevelPoolError;
use crate::tensor::level_pool::{TensorPoolContext, TensorPoolSourceStep};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LevelPoolIntervalSources {
    pub rain_volume_m3: f64,
    pub bottom_loss_volume_m3: f64,
    pub evaporation_volume_m3: f64,
    pub reduction_bound_m3: f64,
    pub roundoff_bound_m3: f64,
    pub counter_roundoff_bound_m3: f64,
}

pub(super) fn interval<R>(
    context: &TensorPoolContext<R>,
    source: Option<&TensorPoolSourceStep<R>>,
) -> Result<LevelPoolIntervalSources, LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    let Some(source) = source else {
        return Ok(LevelPoolIntervalSources::default());
    };
    let physical = super::source_projection::physical_totals(context, Some(source))?;
    let mut roundoff_bound_m3 = 0.0;
    for witness in [
        &source.rain_witness_m3,
        &source.bottom_loss_witness_m3,
        &source.evaporation_witness_m3,
    ] {
        roundoff_bound_m3 = super::accounting::outward_add(
            roundoff_bound_m3,
            super::accounting::sum_bound(
                context,
                &context.client.abs(witness)?,
                "source storage addition bound",
            )?,
        )?;
    }
    let mut counter_roundoff_bound_m3 = 0.0;
    for counter in [
        &source.rain_counter,
        &source.bottom_loss_counter,
        &source.evaporation_counter,
    ] {
        counter_roundoff_bound_m3 = super::accounting::outward_add(
            counter_roundoff_bound_m3,
            super::accounting::sum_bound(
                context,
                &context.client.abs(&counter.witness_m3)?,
                "source counter addition bound",
            )?,
        )?;
    }
    Ok(LevelPoolIntervalSources {
        rain_volume_m3: physical.rain_volume_m3,
        bottom_loss_volume_m3: physical.bottom_loss_volume_m3,
        evaporation_volume_m3: physical.evaporation_volume_m3,
        reduction_bound_m3: super::accounting::reduction_bound(
            context,
            &[
                physical.rain_volume_m3,
                physical.bottom_loss_volume_m3,
                physical.evaporation_volume_m3,
            ],
        )?,
        roundoff_bound_m3,
        counter_roundoff_bound_m3,
    })
}
