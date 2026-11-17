//! Explicit cumulative source projections and outward aggregate diagnostic bounds.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;

use crate::storage::accounting::{read, sum_bound};
use crate::storage::tensor::{TensorPoolContext, TensorPoolSourceState, TensorPoolSourceStep};
use crate::storage::{LevelPoolError, LevelPoolSourceLedger};

pub(crate) fn ledgers<R>(
    context: &TensorPoolContext<R>,
    state: Option<&TensorPoolSourceState<R>>,
) -> Result<Option<Vec<LevelPoolSourceLedger>>, LevelPoolError>
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
    let Some(state) = state else { return Ok(None) };
    let n = context.elements.len();
    let rain = read(&state.rain, n)?;
    let evaporation = read(&state.evaporation, n)?;
    let bottom_loss = read(&state.bottom_loss, n)?;
    let bottom_loss_demand = read(&state.bottom_loss_demand, n)?;
    let rain_bound = read(&state.rain_bound, n)?;
    let evaporation_bound = read(&state.evaporation_bound, n)?;
    let bottom_loss_bound = read(&state.bottom_loss_bound, n)?;
    let bottom_loss_demand_bound = read(&state.bottom_loss_demand_bound, n)?;
    Ok(Some(
        (0..n)
            .map(|i| LevelPoolSourceLedger {
                cumulative_rain_volume_m3: rain[i],
                cumulative_evaporation_volume_m3: evaporation[i],
                cumulative_bottom_loss_volume_m3: bottom_loss[i],
                cumulative_bottom_loss_demand_m3: bottom_loss_demand[i],
                rain_roundoff_bound_m3: rain_bound[i],
                evaporation_roundoff_bound_m3: evaporation_bound[i],
                bottom_loss_roundoff_bound_m3: bottom_loss_bound[i],
                bottom_loss_demand_roundoff_bound_m3: bottom_loss_demand_bound[i],
            })
            .collect(),
    ))
}
pub(crate) fn totals<R>(
    context: &TensorPoolContext<R>,
    state: Option<&TensorPoolSourceState<R>>,
) -> Result<Option<LevelPoolSourceLedger>, LevelPoolError>
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
    let Some(state) = state else { return Ok(None) };
    Ok(Some(LevelPoolSourceLedger {
        cumulative_rain_volume_m3: context.sum(&state.rain, "aggregate source rain")?,
        cumulative_evaporation_volume_m3: context
            .sum(&state.evaporation, "aggregate source evaporation")?,
        cumulative_bottom_loss_volume_m3: context
            .sum(&state.bottom_loss, "aggregate source bottom_loss")?,
        cumulative_bottom_loss_demand_m3: context.sum(
            &state.bottom_loss_demand,
            "aggregate source bottom_loss_demand",
        )?,
        rain_roundoff_bound_m3: sum_bound(context, &state.rain_bound, "aggregate rain_bound")?,
        evaporation_roundoff_bound_m3: sum_bound(
            context,
            &state.evaporation_bound,
            "aggregate evaporation_bound",
        )?,
        bottom_loss_roundoff_bound_m3: sum_bound(
            context,
            &state.bottom_loss_bound,
            "aggregate bottom_loss_bound",
        )?,
        bottom_loss_demand_roundoff_bound_m3: sum_bound(
            context,
            &state.bottom_loss_demand_bound,
            "aggregate bottom_loss_demand_bound",
        )?,
    }))
}
#[derive(Default)]
pub(crate) struct SourceTotals {
    pub rain_volume_m3: f64,
    pub evaporation_volume_m3: f64,
    pub bottom_loss_volume_m3: f64,
    pub bottom_loss_demand_m3: f64,
    pub unmet_bottom_loss_m3: f64,
    pub potential_evaporation_volume_m3: f64,
    pub unmet_evaporation_volume_m3: f64,
}
pub(crate) fn step_totals<R>(
    context: &TensorPoolContext<R>,
    source: Option<&TensorPoolSourceStep<R>>,
) -> Result<SourceTotals, LevelPoolError>
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
    let Some(s) = source else {
        return Ok(SourceTotals::default());
    };
    let physical = physical_totals(context, Some(s))?;
    Ok(SourceTotals {
        rain_volume_m3: physical.rain_volume_m3,
        evaporation_volume_m3: physical.evaporation_volume_m3,
        bottom_loss_volume_m3: physical.bottom_loss_volume_m3,
        bottom_loss_demand_m3: context.sum(
            &s.bottom_loss_cap.potential_volume_m3,
            "aggregate interval bottom_loss_demand_m3",
        )?,
        unmet_bottom_loss_m3: context.sum(
            &s.bottom_loss_cap.unmet_volume_m3,
            "aggregate interval unmet_bottom_loss_m3",
        )?,
        potential_evaporation_volume_m3: context.sum(
            &s.evaporation_cap.potential_volume_m3,
            "aggregate interval potential_evaporation_volume_m3",
        )?,
        unmet_evaporation_volume_m3: context.sum(
            &s.evaporation_cap.unmet_volume_m3,
            "aggregate interval unmet_evaporation_volume_m3",
        )?,
    })
}

#[derive(Default)]
pub(crate) struct PhysicalSourceTotals {
    pub rain_volume_m3: f64,
    pub evaporation_volume_m3: f64,
    pub bottom_loss_volume_m3: f64,
}
pub(crate) fn physical_totals<R>(
    context: &TensorPoolContext<R>,
    source: Option<&TensorPoolSourceStep<R>>,
) -> Result<PhysicalSourceTotals, LevelPoolError>
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
    let Some(s) = source else {
        return Ok(PhysicalSourceTotals::default());
    };
    Ok(PhysicalSourceTotals {
        rain_volume_m3: context.sum(&s.rain_volume_m3, "aggregate interval rain_volume_m3")?,
        evaporation_volume_m3: context.sum(
            &s.evaporation_cap.actual_volume_m3,
            "aggregate interval evaporation_volume_m3",
        )?,
        bottom_loss_volume_m3: context.sum(
            &s.bottom_loss_cap.actual_volume_m3,
            "aggregate interval bottom_loss_volume_m3",
        )?,
    })
}
