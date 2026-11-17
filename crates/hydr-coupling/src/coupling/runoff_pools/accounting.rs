use super::{RunoffPoolBatch, RunoffPoolError, RunoffPoolWaterBalance};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};

impl<R: Runtime<DType = DType>> RunoffPoolBatch<R>
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
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Derive lifetime physical accounting from checked child ledgers without duplicate counters.
    pub fn ledger(&self) -> Result<RunoffPoolWaterBalance, RunoffPoolError> {
        self.check_epochs()?;
        let runoff = self.runoff.ledger();
        let bounds = self.runoff.accounting_bounds()?;
        let pools = self.pools.coupling_totals()?;
        if runoff.external_runon_volume_m3 != 0.0 {
            return Err(RunoffPoolError::input(
                "native lifetime external runon is outside the composite model",
            ));
        }
        let pairing_error = (runoff.outlet_volume_m3 - pools.cumulative_external_volume_m3).abs();
        let pairing_bound = charge(
            charge(bounds.outlet_error_m3, pools.external_roundoff_bound_m3)?,
            pools.external_reduction_bound_m3,
        )?;
        let pairing_used = if pairing_error == 0.0 {
            0.0
        } else {
            charge(
                pairing_error,
                crate::numerics::outward::rounding_bound(pairing_error)
                    .map_err(|e| RunoffPoolError::numerical(e.to_string()))?,
            )?
        };
        if !pairing_error.is_finite() || pairing_used > pairing_bound {
            return Err(RunoffPoolError::numerical(format!(
                "cumulative runoff export and receiving imports differ by {pairing_error}, bound={pairing_bound}. Supply paired child histories"
            )));
        }
        let bound = charge(
            charge(
                charge(
                    bounds.combined_error_m3,
                    runoff.transfer_uncertainty_charge_m3,
                )?,
                pools.reduction_bound_m3,
            )?,
            pools.roundoff_bound_m3,
        )?;
        let balance = combine(Parts {
            before: runoff.initial_storage_m3 + pools.initial_volume_m3,
            after: runoff.storage_m3 + pools.volume_m3,
            rain: runoff.applied_rain_volume_m3 + pools.cumulative_rain_volume_m3,
            infiltration: runoff.infiltration_volume_m3,
            evaporation: runoff.evaporation_volume_m3 + pools.cumulative_evaporation_volume_m3,
            bottom: pools.cumulative_bottom_loss_volume_m3,
            adjustment: runoff.soil_water_adjustment_volume_m3,
            outlet: runoff.outlet_volume_m3,
            bound,
        })?;
        self.admit(&balance)?;
        Ok(balance)
    }
    pub(super) fn interval_balance(&self) -> Result<RunoffPoolWaterBalance, RunoffPoolError> {
        self.check_epochs()?;
        let runoff = self.runoff.interval_summary()?;
        let pools = self
            .pools
            .coupling_last_step()?
            .ok_or_else(|| RunoffPoolError::input("paired pool interval receipt is missing"))?;
        if runoff.start_s.to_bits() != pools.start_s.to_bits()
            || runoff.end_s.to_bits() != pools.end_s.to_bits()
            || runoff.totals.external != 0.0
        {
            return Err(RunoffPoolError::input(
                "native child interval endpoints or excluded runon differ",
            ));
        }
        let bound = charge(
            charge(
                charge(
                    runoff.bounds.combined_error_m3,
                    runoff.transfer_uncertainty_charge_m3,
                )?,
                pools.reduction_bound_m3,
            )?,
            pools.roundoff_bound_m3,
        )?;
        let t = runoff.totals;
        let balance = combine(Parts {
            before: t.before + pools.volume_before_m3,
            after: t.after + pools.volume_after_m3,
            rain: t.rain + pools.rain_volume_m3,
            infiltration: t.infiltration,
            evaporation: t.evaporation + pools.evaporation_volume_m3,
            bottom: pools.bottom_loss_volume_m3,
            adjustment: t.adjustment,
            outlet: t.outlet,
            bound,
        })?;
        self.admit(&balance)?;
        Ok(balance)
    }
    pub(super) fn check_epochs(&self) -> Result<(), RunoffPoolError> {
        if self.runoff.time().to_bits() != self.pools.time().to_bits()
            || self.runoff.accepted_steps() != self.pools.pools().accepted_steps()
            || self.time() < self.initial_time()
        {
            return Err(RunoffPoolError::input(
                "child accepted times, counts, or common original epoch differ",
            ));
        }
        Ok(())
    }
    fn admit(&self, balance: &RunoffPoolWaterBalance) -> Result<(), RunoffPoolError> {
        let c = &self.context.setup.config;
        let budget = c.mass_absolute_m3 + c.mass_relative * balance.physical_scale_m3;
        let used = charge(balance.water_residual_m3.abs(), balance.roundoff_bound_m3)?;
        if !budget.is_finite() || used > budget {
            return Err(RunoffPoolError::numerical(format!(
                "parent water residual={}, bound={}, budget={budget}: refine the common interval or inspect physical support",
                balance.water_residual_m3, balance.roundoff_bound_m3
            )));
        }
        Ok(())
    }
}
struct Parts {
    before: f64,
    after: f64,
    rain: f64,
    infiltration: f64,
    evaporation: f64,
    bottom: f64,
    adjustment: f64,
    outlet: f64,
    bound: f64,
}
fn combine(p: Parts) -> Result<RunoffPoolWaterBalance, RunoffPoolError> {
    let delta = p.after - p.before;
    let less_infiltration = p.rain - p.infiltration;
    let less_evaporation = less_infiltration - p.evaporation;
    let less_bottom = less_evaporation - p.bottom;
    let corrected = less_bottom + p.adjustment;
    let residual = delta - corrected;
    let terms = [
        p.before,
        p.after,
        p.rain,
        p.infiltration,
        p.evaporation,
        p.bottom,
        p.adjustment,
    ];
    let scale = [
        p.before,
        p.after,
        p.rain,
        p.infiltration,
        p.evaporation,
        p.bottom,
    ]
    .into_iter()
    .fold(0.0, f64::max);
    let mut bound = p.bound;
    for value in terms.into_iter().chain([
        delta,
        less_infiltration,
        less_evaporation,
        less_bottom,
        corrected,
        residual,
    ]) {
        let rounding = crate::numerics::outward::rounding_bound(value)
            .map_err(|e| RunoffPoolError::numerical(e.to_string()))?;
        bound = charge(bound, rounding)?;
    }
    if !residual.is_finite() || !scale.is_finite() || !p.outlet.is_finite() {
        return Err(RunoffPoolError::numerical(
            "composite physical accounting overflows. Rescale represented support",
        ));
    }
    Ok(RunoffPoolWaterBalance {
        storage_before_m3: p.before,
        storage_after_m3: p.after,
        rain_volume_m3: p.rain,
        infiltration_volume_m3: p.infiltration,
        evaporation_volume_m3: p.evaporation,
        bottom_loss_volume_m3: p.bottom,
        soil_water_adjustment_volume_m3: p.adjustment,
        internal_outlet_volume_m3: p.outlet,
        water_residual_m3: residual,
        roundoff_bound_m3: bound,
        physical_scale_m3: scale,
    })
}
fn charge(a: f64, b: f64) -> Result<f64, RunoffPoolError> {
    crate::numerics::outward::add_adjustment_charge(a, b)
        .map_err(|e| RunoffPoolError::numerical(e.to_string()))
}
