//! Scalar physical-source summaries without per-element host projection.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;

use super::LevelPoolError;
use crate::storage::tensor::{TensorPoolContext, TensorPoolSourceStep};

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LevelPoolIntervalSources {
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

#[cfg(test)]
mod tests {
    use crate::storage::LevelPoolSources;
    use crate::storage::fixture::batch;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn source_summary_retains_actual_processes_when_counters_absorb_increments() -> TestResult {
        let mut batch = batch(true)?;
        let film = batch.context.tensor(&[1e-20, 0.0])?;
        batch.state.volume_m3 = film.clone();
        batch.state.initial_volume_m3 = film;
        let counter = batch.context.tensor(&[1.0, 0.0])?;
        let source_state = batch
            .state
            .sources
            .as_mut()
            .ok_or("missing source counters")?;
        source_state.rain = counter.clone();
        source_state.bottom_loss = counter.clone();
        source_state.bottom_loss_demand = counter;
        let rate = f64::EPSILON / 4.0;
        let plan = batch.bind_source_rates(&[
            LevelPoolSources {
                rain_rate_m_per_s: rate,
                potential_evaporation_rate_m_per_s: 0.0,
                bottom_loss_rate_m3_per_s: rate,
            },
            LevelPoolSources {
                rain_rate_m_per_s: 0.0,
                potential_evaporation_rate_m_per_s: 0.0,
                bottom_loss_rate_m3_per_s: 0.0,
            },
        ])?;
        let prepared = plan.prepare(1.0)?;
        let totals = prepared.source_totals()?;
        assert_eq!(totals.rain_volume_m3, rate);
        assert_eq!(totals.bottom_loss_volume_m3, rate);
        assert!(totals.counter_roundoff_bound_m3 >= 2.0 * rate);
        let zero = batch.context.tensor(&[0.0, 0.0])?;
        let next = prepared.trial_to_tensors(&zero, &zero)?.into_next();
        let before = batch
            .totals()?
            .sources
            .ok_or("missing original source totals")?;
        let after = next.totals()?.sources.ok_or("missing next source totals")?;
        assert_eq!(
            after.cumulative_rain_volume_m3 - before.cumulative_rain_volume_m3,
            0.0
        );
        assert_eq!(
            after.cumulative_bottom_loss_volume_m3 - before.cumulative_bottom_loss_volume_m3,
            0.0
        );
        Ok(())
    }
}
