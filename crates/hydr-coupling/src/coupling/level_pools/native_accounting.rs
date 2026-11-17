//! Scalar physical summaries for atomic composition with another checked engine.

use super::{
    CoupledLevelPoolBatch, CoupledLevelPoolError,
    context::Context,
    mass::{InflowMass, charge},
};
use crate::numerics::outward::AdjustmentReduction;
use crate::storage::LevelPoolTotals;
use hydr_storage::internal::LevelPoolIntervalSources;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PoolCouplingTotals {
    pub initial_volume_m3: f64,
    pub volume_m3: f64,
    pub cumulative_rain_volume_m3: f64,
    pub cumulative_evaporation_volume_m3: f64,
    pub cumulative_bottom_loss_volume_m3: f64,
    pub cumulative_external_volume_m3: f64,
    pub reduction_bound_m3: f64,
    pub roundoff_bound_m3: f64,
    pub external_reduction_bound_m3: f64,
    pub external_roundoff_bound_m3: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PoolCouplingInterval {
    pub start_s: f64,
    pub end_s: f64,
    pub volume_before_m3: f64,
    pub volume_after_m3: f64,
    pub rain_volume_m3: f64,
    pub evaporation_volume_m3: f64,
    pub bottom_loss_volume_m3: f64,
    pub reduction_bound_m3: f64,
    pub roundoff_bound_m3: f64,
}

fn reduction_bound(count: usize, values: &[f64]) -> Result<f64, CoupledLevelPoolError> {
    let reduction = AdjustmentReduction::new(count)
        .map_err(|error| CoupledLevelPoolError::numerical(error.to_string()))?;
    values.iter().try_fold(0.0, |bound, &value| {
        charge(
            bound,
            reduction
                .unit_sum_error_bound(value)
                .map_err(|error| CoupledLevelPoolError::numerical(error.to_string()))?,
        )
    })
}

fn historical_addition_bound(totals: &LevelPoolTotals) -> Result<f64, CoupledLevelPoolError> {
    let mut bound = charge(
        totals.incoming_roundoff_bound_m3,
        totals.outgoing_roundoff_bound_m3,
    )?;
    if let Some(source) = &totals.sources {
        for value in [
            source.rain_roundoff_bound_m3,
            source.evaporation_roundoff_bound_m3,
            source.bottom_loss_roundoff_bound_m3,
        ] {
            bound = charge(bound, value)?;
        }
    }
    Ok(bound)
}

impl<R: Runtime<DType = DType>> CoupledLevelPoolBatch<R>
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
    pub(crate) fn coupling_totals(&self) -> Result<PoolCouplingTotals, CoupledLevelPoolError> {
        let native = self.pools.totals()?;
        let external = self
            .context
            .inflows
            .state_summary(&self.context.client, &self.external)?;
        let (rain, evaporation, bottom_loss) =
            native.sources.as_ref().map_or((0.0, 0.0, 0.0), |source| {
                (
                    source.cumulative_rain_volume_m3,
                    source.cumulative_evaporation_volume_m3,
                    source.cumulative_bottom_loss_volume_m3,
                )
            });
        // Physical counters exclude internal transfers. Their historical addition
        // uncertainty remains charged through the independently checked child ledger.
        let reduction = reduction_bound(
            self.context.pool_count(),
            &[
                native.initial_volume_m3,
                native.volume_m3,
                native.cumulative_incoming_volume_m3,
                native.cumulative_outgoing_volume_m3,
                rain,
                evaporation,
                bottom_loss,
            ],
        )?;
        Ok(PoolCouplingTotals {
            initial_volume_m3: native.initial_volume_m3,
            volume_m3: native.volume_m3,
            cumulative_rain_volume_m3: rain,
            cumulative_evaporation_volume_m3: evaporation,
            cumulative_bottom_loss_volume_m3: bottom_loss,
            cumulative_external_volume_m3: external.volume_m3,
            reduction_bound_m3: charge(reduction, external.reduction_bound_m3)?,
            roundoff_bound_m3: charge(
                historical_addition_bound(&native)?,
                external.roundoff_bound_m3,
            )?,
            external_reduction_bound_m3: external.reduction_bound_m3,
            external_roundoff_bound_m3: external.roundoff_bound_m3,
        })
    }

    pub(crate) fn coupling_last_step(
        &self,
    ) -> Result<Option<PoolCouplingInterval>, CoupledLevelPoolError> {
        Ok(self.last.as_ref().map(|step| step.coupling))
    }

    pub(crate) fn original_inflow_volumes(
        &self,
    ) -> Result<Option<Tensor<R>>, CoupledLevelPoolError> {
        self.last.as_ref().map(|step| {
            if !step.external.all_interval_volumes() {
                return Err(CoupledLevelPoolError::input("native pairing requires original integrated-volume inputs at every boundary"));
            }
            let volumes = step.external.original_volumes();
            self.context.inflows.check_tensor(&self.context.client, &volumes)?;
            Ok(volumes)
        }).transpose()
    }
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
    pub(super) fn interval_summary(
        &self,
        before: &LevelPoolTotals,
        after: &LevelPoolTotals,
        times: (f64, f64),
        assembly: f64,
        sources: Option<&LevelPoolIntervalSources>,
        external: InflowMass<'_>,
    ) -> Result<PoolCouplingInterval, CoupledLevelPoolError> {
        let old = before;
        let new = after;
        let source = match sources {
            Some(source) => *source,
            None => LevelPoolIntervalSources::default(),
        };
        let reduction = reduction_bound(self.pool_count(), &[old.volume_m3, new.volume_m3])?;
        let reduction = charge(
            charge(reduction, source.reduction_bound_m3)?,
            external.interval.reduction_bound_m3,
        )?;
        let mut roundoff = charge(
            historical_addition_bound(old)?,
            historical_addition_bound(new)?,
        )?;
        for value in [
            assembly,
            source.roundoff_bound_m3,
            source.counter_roundoff_bound_m3,
            external.before.roundoff_bound_m3,
            external.cumulative.roundoff_bound_m3,
            external.interval.counter_roundoff_bound_m3,
        ] {
            roundoff = charge(roundoff, value)?;
        }
        Ok(PoolCouplingInterval {
            start_s: times.0,
            end_s: times.1,
            volume_before_m3: old.volume_m3,
            volume_after_m3: new.volume_m3,
            rain_volume_m3: source.rain_volume_m3,
            evaporation_volume_m3: source.evaporation_volume_m3,
            bottom_loss_volume_m3: source.bottom_loss_volume_m3,
            reduction_bound_m3: reduction,
            roundoff_bound_m3: roundoff,
        })
    }
}
