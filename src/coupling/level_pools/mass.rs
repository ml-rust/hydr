use super::{CoupledLevelPoolError, context::Context};
use crate::storage::{LevelPoolBatch, LevelPoolIntervalSources};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;

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
    ) -> Result<f64, CoupledLevelPoolError> {
        let old = before.totals()?;
        let new = after.totals()?;
        let mut interval = new.volume_m3 - old.volume_m3;
        let mut cumulative = new.volume_m3 - new.initial_volume_m3;
        let mut interval_extra = 0.0;
        let mut cumulative_extra = 0.0;
        match (sources, &new.sources) {
            (None, None) => {}
            (Some(source), Some(counters)) => {
                interval = interval - source.rain_volume_m3
                    + source.bottom_loss_volume_m3
                    + source.evaporation_volume_m3;
                cumulative = cumulative - counters.cumulative_rain_volume_m3
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
        if !budget.is_finite()
            || !interval_charge.is_finite()
            || !cumulative_charge.is_finite()
            || interval_charge > budget
            || cumulative_charge > budget
        {
            return Err(CoupledLevelPoolError::numerical(format!(
                "mass drift interval={interval}, cumulative={cumulative}, assembly={assembly}, budget={budget}: shorten the interval or inspect physical scales"
            )));
        }
        Ok(interval)
    }
}
fn charge(a: f64, b: f64) -> Result<f64, CoupledLevelPoolError> {
    crate::soil::admission::add_adjustment_charge(a, b)
        .map_err(|error| CoupledLevelPoolError::numerical(error.to_string()))
}
