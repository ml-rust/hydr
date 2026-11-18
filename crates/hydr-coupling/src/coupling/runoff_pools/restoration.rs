use super::identity::{same_balance, same_sources};
use super::{RunoffPoolBatch, RunoffPoolError, RunoffPoolSnapshotData, RunoffPoolStepDiagnostics};
use crate::coupling::CoupledPoolInflowValue;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    MidpointOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::sync::Arc;

impl<R: Runtime<DType = DType>> RunoffPoolBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
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
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Check complete owned child proofs and their pairing before replacing either accepted engine.
    pub fn restore_data(&mut self, data: &RunoffPoolSnapshotData) -> Result<(), RunoffPoolError> {
        data.check_identities().map_err(RunoffPoolError::input)?;
        self.check_identity(&data.setup, data.initial_time_s)?;
        let mut runoff = self.runoff.clone();
        runoff.restore_data(&data.runoff)?;
        let mut pools = self.pools.clone();
        pools.restore_data(&data.pools)?;
        let candidate = Self {
            context: Arc::clone(&self.context),
            runoff,
            pools,
            last: data.last_step.clone().map(Arc::new),
        };
        candidate.check_continuation(true)?;
        match (
            &data.last_step,
            &data.runoff.last_step,
            &data.pools.last_step,
        ) {
            (None, None, None) => {}
            (Some(common), Some(native), Some(pool)) => {
                if native.receipts.len() != common.forcing.subcatchments.len() {
                    return Err(RunoffPoolError::input(
                        "parent meteorology count differs from complete native originals",
                    ));
                }
                for (index, (receipt, forcing)) in native
                    .receipts
                    .iter()
                    .zip(&common.forcing.subcatchments)
                    .enumerate()
                {
                    if receipt.catchment != index
                        || receipt.input.rain_rate_m_per_s.to_bits()
                            != forcing.rain_rate_m_per_s.to_bits()
                        || receipt.input.potential_evaporation_m_per_s.to_bits()
                            != forcing.potential_evaporation_m_per_s.to_bits()
                        || receipt
                            .input
                            .external_runon_volumes_m3
                            .iter()
                            .any(|v| *v != 0.0)
                    {
                        return Err(RunoffPoolError::input(
                            "parent meteorology or excluded runon differs from native original inputs",
                        ));
                    }
                }
                let forcing = pool.forcing_interval.as_ref().ok_or_else(|| {
                    RunoffPoolError::input("pool receipt lacks retained native-volume forcing")
                })?;
                if !same_sources(
                    common.forcing.pool_sources.as_deref(),
                    forcing.sources.as_deref(),
                ) || forcing.start_s.to_bits() != common.start_s.to_bits()
                    || forcing.end_s.to_bits() != common.end_s.to_bits()
                    || pool.requested_end_s.to_bits() != common.end_s.to_bits()
                    || pool.retries != 0
                    || forcing
                        .inflows
                        .iter()
                        .any(|i| !matches!(i.value, CoupledPoolInflowValue::IntervalVolume { .. }))
                {
                    return Err(RunoffPoolError::input(
                        "parent sources, exact native-volume span, or child retry metadata differ",
                    ));
                }
            }
            _ => {
                return Err(RunoffPoolError::input(
                    "common and native final receipt presence differs",
                ));
            }
        }
        *self = candidate;
        Ok(())
    }
    pub(super) fn check_continuation(&self, owned: bool) -> Result<(), RunoffPoolError> {
        self.check_epochs()?;
        self.ledger()?;
        let count = self.runoff.accepted_steps();
        match &self.last {
            None => {
                if count != 0
                    || self.time().to_bits() != self.initial_time().to_bits()
                    || self.pools.coupling_last_step()?.is_some()
                {
                    return Err(RunoffPoolError::input(
                        "zero-history composite requires the original epoch and no final child intervals",
                    ));
                }
            }
            Some(common) => {
                if count == 0 {
                    return Err(RunoffPoolError::input(
                        "initial composite contains an advanced common receipt",
                    ));
                }
                self.check_common(common)?;
                let runoff = self.runoff.interval_summary()?;
                let pools = self
                    .pools
                    .coupling_last_step()?
                    .ok_or_else(|| RunoffPoolError::input("advanced pools lack common interval"))?;
                if runoff.start_s.to_bits() != common.start_s.to_bits()
                    || runoff.end_s.to_bits() != common.end_s.to_bits()
                    || pools.start_s.to_bits() != common.start_s.to_bits()
                    || pools.end_s.to_bits() != common.end_s.to_bits()
                    || runoff.totals.external != 0.0
                {
                    return Err(RunoffPoolError::input(
                        "common final interval endpoints or excluded runon differ",
                    ));
                }
                let inflows = self.pools.original_inflow_volumes()?.ok_or_else(|| {
                    RunoffPoolError::input("advanced pools lack original native input volumes")
                })?;
                self.context
                    .check_pair(self.runoff.original_outlet()?, &inflows, owned)?;
                if !same_balance(&self.interval_balance()?, &common.balance) {
                    return Err(RunoffPoolError::input(
                        "common water receipt differs from checked original child totals and bounds",
                    ));
                }
            }
        }
        Ok(())
    }
    fn check_common(&self, step: &RunoffPoolStepDiagnostics) -> Result<(), RunoffPoolError> {
        self.context
            .check_forcing(&step.forcing, step.start_s, step.requested_end_s)?;
        if !step.start_s.is_finite()
            || step.start_s < self.initial_time()
            || step.end_s.to_bits() != self.time().to_bits()
            || step.end_s <= step.start_s
            || step.end_s > step.requested_end_s
            || step.end_s - step.start_s < self.context.minimum
            || step.outlets != self.setup().outlets
            || step.retries > self.setup().config.retry_limit
            || (self.runoff.accepted_steps() == 1
                && step.start_s.to_bits() != self.initial_time().to_bits())
        {
            return Err(RunoffPoolError::input(
                "common interval, retry limits, mapping, or original epoch are inconsistent",
            ));
        }
        let mut expected = step
            .requested_end_s
            .min(step.start_s + self.context.maximum);
        for _ in 0..step.retries {
            let next = step.start_s + (expected - step.start_s) * 0.5;
            if next <= step.start_s || next >= expected {
                return Err(RunoffPoolError::input(
                    "common retry sequence makes no represented progress",
                ));
            }
            expected = next;
        }
        if expected.to_bits() != step.end_s.to_bits() {
            return Err(RunoffPoolError::input(
                "common endpoint differs from complete bounded recalculation sequence",
            ));
        }
        Ok(())
    }
}
