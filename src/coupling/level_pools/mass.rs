use super::inflows::{InflowCumulativeSummary, InflowIntervalSummary};
use super::{CoupledLevelPoolError, context::Context};
use crate::storage::{LevelPoolBatch, LevelPoolTotals};
use hydr_storage::internal::LevelPoolIntervalSources;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;

pub(super) struct CheckedPoolMass {
    pub residual: f64,
    pub before: LevelPoolTotals,
    pub after: LevelPoolTotals,
}

pub(super) struct InflowMass<'a> {
    pub before: &'a InflowCumulativeSummary,
    pub interval: &'a InflowIntervalSummary,
    pub cumulative: &'a InflowCumulativeSummary,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    pub fn mass_check(
        &self,
        before: &LevelPoolBatch<R>,
        after: &LevelPoolBatch<R>,
        assembly: f64,
        sources: Option<&LevelPoolIntervalSources>,
        external: InflowMass<'_>,
    ) -> Result<CheckedPoolMass, CoupledLevelPoolError> {
        let old = before.totals()?;
        let new = after.totals()?;
        let delta = new.volume_m3 - old.volume_m3;
        let cumulative_delta = new.volume_m3 - new.initial_volume_m3;
        let mut prior = old.volume_m3 - old.initial_volume_m3 - external.before.volume_m3;
        let mut prior_extra = charge(
            external.before.roundoff_bound_m3,
            external.before.reduction_bound_m3,
        )?;
        if let Some(counters) = &old.sources {
            prior = old.volume_m3
                - old.initial_volume_m3
                - counters.cumulative_rain_volume_m3
                - external.before.volume_m3
                + counters.cumulative_bottom_loss_volume_m3
                + counters.cumulative_evaporation_volume_m3;
            prior_extra = charge(
                charge(
                    charge(prior_extra, counters.rain_roundoff_bound_m3)?,
                    counters.bottom_loss_roundoff_bound_m3,
                )?,
                counters.evaporation_roundoff_bound_m3,
            )?;
        }
        let mut interval = delta - external.interval.volume_m3;
        let mut cumulative = cumulative_delta - external.cumulative.volume_m3;
        let mut interval_extra = 0.0;
        let mut cumulative_extra = 0.0;
        match (sources, &new.sources) {
            (None, None) => {}
            (Some(source), Some(counters)) => {
                interval = delta - source.rain_volume_m3 - external.interval.volume_m3
                    + source.bottom_loss_volume_m3
                    + source.evaporation_volume_m3;
                cumulative = cumulative_delta
                    - counters.cumulative_rain_volume_m3
                    - external.cumulative.volume_m3
                    + counters.cumulative_bottom_loss_volume_m3
                    + counters.cumulative_evaporation_volume_m3;
                interval_extra = charge(
                    charge(source.reduction_bound_m3, source.roundoff_bound_m3)?,
                    source.counter_roundoff_bound_m3,
                )?;
                cumulative_extra = charge(
                    charge(
                        counters.rain_roundoff_bound_m3,
                        counters.bottom_loss_roundoff_bound_m3,
                    )?,
                    counters.evaporation_roundoff_bound_m3,
                )?;
            }
            _ => {
                return Err(CoupledLevelPoolError::input(
                    "mass accounting source summary differs from native source policy. Supply complete original interval forcing",
                ));
            }
        }
        interval_extra = charge(
            charge(interval_extra, external.interval.reduction_bound_m3)?,
            external.interval.counter_roundoff_bound_m3,
        )?;
        cumulative_extra = charge(
            charge(cumulative_extra, external.cumulative.roundoff_bound_m3)?,
            external.cumulative.reduction_bound_m3,
        )?;
        let scale = old.physical_scale_m3.max(new.physical_scale_m3);
        let budget = self.setup.config.mass_absolute_m3 + self.setup.config.mass_relative * scale;
        let interval_charge = charge(
            charge(
                charge(
                    charge(interval.abs(), old.reduction_bound_m3)?,
                    new.reduction_bound_m3,
                )?,
                assembly,
            )?,
            interval_extra,
        )?;
        let cumulative_charge = charge(
            charge(charge(cumulative.abs(), new.reduction_bound_m3)?, assembly)?,
            cumulative_extra,
        )?;
        let prior_charge = charge(charge(prior.abs(), old.reduction_bound_m3)?, prior_extra)?;
        if !budget.is_finite()
            || !interval_charge.is_finite()
            || !cumulative_charge.is_finite()
            || !prior_charge.is_finite()
            || prior_charge > budget
            || interval_charge > budget
            || cumulative_charge > budget
        {
            return Err(CoupledLevelPoolError::numerical(format!(
                "mass drift start={prior}, interval={interval}, cumulative={cumulative}, assembly={assembly}, budget={budget}: shorten the interval or inspect physical scales"
            )));
        }
        Ok(CheckedPoolMass {
            residual: interval,
            before: old,
            after: new,
        })
    }
}
pub(super) fn charge(a: f64, b: f64) -> Result<f64, CoupledLevelPoolError> {
    crate::numerics::outward::add_adjustment_charge(a, b)
        .map_err(|error| CoupledLevelPoolError::numerical(error.to_string()))
}
