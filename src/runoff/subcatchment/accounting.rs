//! Complete scalar lifetime counters and interval metadata.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::{SubcatchmentError, SubcatchmentReceipt, SubcatchmentTransfer};
use crate::numerics::subcatchment::{TensorSubcatchmentStep, bounds};
use crate::soil::admission::add_adjustment_charge;
use crate::tensor::subcatchment::TensorSubcatchmentContext;

/// Independent physical mass counters and separate numerical admission budgets.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentLedger {
    /// Original represented storage in m³.
    pub initial_storage_m3: f64,
    /// Current represented storage in m³.
    pub storage_m3: f64,
    /// Lifetime actual applied rain in m³.
    pub applied_rain_volume_m3: f64,
    /// Lifetime original external runon in m³.
    pub external_runon_volume_m3: f64,
    /// Lifetime original native-model infiltration in m³.
    pub infiltration_volume_m3: f64,
    /// Lifetime capped evaporation in m³.
    pub evaporation_volume_m3: f64,
    /// Lifetime component outlet export in m³.
    pub outlet_volume_m3: f64,
    /// Signed reported soil numerical water correction in m³.
    pub soil_water_adjustment_volume_m3: f64,
    /// Conservative lifetime absolute soil correction consumption in m³.
    pub soil_water_adjustment_abs_bound_m3: f64,
    /// Lifetime internal interface traffic counted once, in m³.
    pub internal_transfer_volume_m3: f64,
    /// Conservative lifetime transfer/bridge uncertainty consumption in m³.
    pub transfer_uncertainty_charge_m3: f64,
    /// Monotone lower bound for cumulative physical uncertainty scale, in m³.
    pub physical_scale_lower_m3: f64,
    /// Independent signed cumulative water residual in m³.
    pub residual_m3: f64,
}

/// One static topological phase's work and conservative scalar admission evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentPhaseDiagnostics {
    /// Batched excess-root iterations for this phase.
    pub root_iterations: usize,
    /// Conservative absolute soil correction charge in m³.
    pub soil_water_adjustment_abs_charge_m3: f64,
    /// Conservative transfer/bridge uncertainty charge in m³.
    pub transfer_uncertainty_charge_m3: f64,
    /// Phase physical-scale lower increment in m³.
    pub physical_scale_lower_m3: f64,
}

/// Complete accepted interval and ordered source/transfer witnesses.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentStepDiagnostics {
    /// Accepted starting time in seconds.
    pub start: f64,
    /// Accepted ending time in seconds.
    pub end: f64,
    /// Exact represented interval in seconds.
    pub dt: f64,
    /// Total batched root work across both phases.
    pub root_iterations: usize,
    /// Aggregate original storage in m³.
    pub storage_before_m3: f64,
    /// Aggregate accepted storage in m³.
    pub storage_after_m3: f64,
    /// Actual applied rain in m³.
    pub applied_rain_volume_m3: f64,
    /// Original external runon in m³.
    pub external_runon_volume_m3: f64,
    /// Original native-model infiltration in m³.
    pub infiltration_volume_m3: f64,
    /// Capped evaporation in m³.
    pub evaporation_volume_m3: f64,
    /// Component outlet export in m³.
    pub outlet_volume_m3: f64,
    /// Reported explicit soil correction in m³.
    pub soil_water_adjustment_volume_m3: f64,
    /// Conservative absolute soil correction charge in m³.
    pub soil_water_adjustment_abs_charge_m3: f64,
    /// Internal interface traffic counted once, in m³.
    pub internal_transfer_volume_m3: f64,
    /// Conservative transfer/bridge uncertainty charge in m³.
    pub transfer_uncertainty_charge_m3: f64,
    /// Step physical-scale lower increment in m³.
    pub physical_scale_lower_m3: f64,
    /// Signed aggregate physical water residual in m³.
    pub water_residual_m3: f64,
    /// Independent aggregate physical tolerance scale in m³.
    pub water_scale_m3: f64,
    /// Applied-minus-total-area rainfall diagnostic in m³.
    pub rain_area_difference_m3: f64,
    /// Outward rain-area comparison envelope in m³.
    pub rain_area_bound_m3: f64,
    /// Static first-phase work and budget evidence.
    pub phase1: Option<SubcatchmentPhaseDiagnostics>,
    /// Static receiver-phase work and budget evidence.
    pub phase2: Option<SubcatchmentPhaseDiagnostics>,
    /// Ordered complete catchment support and source receipts.
    pub receipts: Vec<SubcatchmentReceipt>,
    /// Ordered explicit internal interface witnesses.
    pub transfers: Vec<SubcatchmentTransfer>,
}

pub(super) struct StepTotals {
    pub before: f64,
    pub after: f64,
    pub rain: f64,
    pub external: f64,
    pub infiltration: f64,
    pub evaporation: f64,
    pub outlet: f64,
    pub adjustment: f64,
    pub internal: f64,
    pub residual: f64,
    pub scale: f64,
    pub rain_difference: f64,
    pub rain_bound: f64,
}

pub(super) fn aggregate_step<R>(
    context: &TensorSubcatchmentContext<R>,
    step: &TensorSubcatchmentStep<R>,
) -> Result<StepTotals, SubcatchmentError>
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
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let t = &step.totals;
    let before = context.total(&t.storage_before, "original storage")?;
    let after = context.total(&t.storage_after, "accepted storage")?;
    let rain = context.total(&t.applied_rain, "applied rainfall")?;
    let external = context.total(&t.external_runon, "external runon")?;
    let infiltration = context.total(&t.infiltration, "infiltration")?;
    let evaporation = context.total(&t.evaporation, "evaporation")?;
    let outlet = context.total(&t.outlet, "component outlet")?;
    let adjustment = context.total(&t.soil_adjustment, "soil adjustment")?;
    let internal = context.total(&t.internal_received, "internal traffic")?;
    let residual =
        (after - before) - (rain + external - infiltration - evaporation - outlet + adjustment);
    let scale = [
        before,
        after,
        rain,
        external,
        infiltration,
        evaporation,
        outlet,
        adjustment,
    ]
    .into_iter()
    .fold(0.0, f64::max);
    check_residual(context, residual, scale, "aggregate interval")?;
    let rain_difference =
        context.total(&t.catchment_rain_area_difference, "rain-area difference")?;
    let rain_bound = context.total(&t.catchment_rain_area_bound, "rain-area envelope")?;
    Ok(StepTotals {
        before,
        after,
        rain,
        external,
        infiltration,
        evaporation,
        outlet,
        adjustment,
        internal,
        residual,
        scale,
        rain_difference,
        rain_bound,
    })
}

impl SubcatchmentLedger {
    pub(super) fn new(storage: f64, lower: f64) -> Self {
        Self {
            initial_storage_m3: storage,
            storage_m3: storage,
            applied_rain_volume_m3: 0.0,
            external_runon_volume_m3: 0.0,
            infiltration_volume_m3: 0.0,
            evaporation_volume_m3: 0.0,
            outlet_volume_m3: 0.0,
            soil_water_adjustment_volume_m3: 0.0,
            soil_water_adjustment_abs_bound_m3: 0.0,
            internal_transfer_volume_m3: 0.0,
            transfer_uncertainty_charge_m3: 0.0,
            physical_scale_lower_m3: lower,
            residual_m3: 0.0,
        }
    }
    pub(super) fn advance<R>(
        &self,
        context: &TensorSubcatchmentContext<R>,
        step: &TensorSubcatchmentStep<R>,
        totals: &StepTotals,
        steps: usize,
    ) -> Result<Self, SubcatchmentError>
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
            + BatchedNewtonAlgorithms<R>
            + BatchedBisectionAlgorithms<R>,
    {
        let mut next = Self {
            initial_storage_m3: self.initial_storage_m3,
            storage_m3: totals.after,
            applied_rain_volume_m3: self.applied_rain_volume_m3 + totals.rain,
            external_runon_volume_m3: self.external_runon_volume_m3 + totals.external,
            infiltration_volume_m3: self.infiltration_volume_m3 + totals.infiltration,
            evaporation_volume_m3: self.evaporation_volume_m3 + totals.evaporation,
            outlet_volume_m3: self.outlet_volume_m3 + totals.outlet,
            soil_water_adjustment_volume_m3: self.soil_water_adjustment_volume_m3
                + totals.adjustment,
            soil_water_adjustment_abs_bound_m3: add_adjustment_charge(
                self.soil_water_adjustment_abs_bound_m3,
                step.totals.adjustment_charge,
            )?,
            internal_transfer_volume_m3: self.internal_transfer_volume_m3 + totals.internal,
            transfer_uncertainty_charge_m3: bounds::add_charge(
                self.transfer_uncertainty_charge_m3,
                step.totals.uncertainty_charge,
            )?,
            physical_scale_lower_m3: bounds::add_scale_lower(
                self.physical_scale_lower_m3,
                step.totals.physical_scale_lower,
            )?,
            residual_m3: 0.0,
        };
        next.residual_m3 = (next.storage_m3 - next.initial_storage_m3)
            - (next.applied_rain_volume_m3 + next.external_runon_volume_m3
                - next.infiltration_volume_m3
                - next.evaporation_volume_m3
                - next.outlet_volume_m3
                + next.soil_water_adjustment_volume_m3);
        check_ledger(context, &next, steps)?;
        Ok(next)
    }
}

pub(super) fn check_ledger<R>(
    context: &TensorSubcatchmentContext<R>,
    ledger: &SubcatchmentLedger,
    steps: usize,
) -> Result<(), SubcatchmentError>
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
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let counters = [
        ledger.initial_storage_m3,
        ledger.storage_m3,
        ledger.applied_rain_volume_m3,
        ledger.external_runon_volume_m3,
        ledger.infiltration_volume_m3,
        ledger.evaporation_volume_m3,
        ledger.outlet_volume_m3,
        ledger.soil_water_adjustment_volume_m3,
        ledger.soil_water_adjustment_abs_bound_m3,
        ledger.internal_transfer_volume_m3,
        ledger.transfer_uncertainty_charge_m3,
        ledger.physical_scale_lower_m3,
    ];
    if counters.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(SubcatchmentError::numerical(
            "cumulative counters must remain finite and nonnegative",
        ));
    }
    let ceiling = super::accounting_bounds::physical_scale_ceiling(context, ledger, steps)?;
    if ledger.physical_scale_lower_m3 > ceiling {
        return Err(SubcatchmentError::numerical(format!(
            "physical scale lower={} exceeds independent counter ceiling={ceiling}",
            ledger.physical_scale_lower_m3
        )));
    }
    let permitted = bounds::permitted_budget(
        context.config.reservoir.volume_absolute,
        context.config.reservoir.volume_relative,
        ledger.physical_scale_lower_m3,
    )?;
    let traffic_lower = super::accounting_bounds::transfer_charge_lower(context, ledger, steps)?;
    let positive_traffic =
        ledger.external_runon_volume_m3 > 0.0 || ledger.internal_transfer_volume_m3 > 0.0;
    if ledger.transfer_uncertainty_charge_m3 < traffic_lower
        || (positive_traffic && ledger.transfer_uncertainty_charge_m3 == 0.0)
    {
        return Err(SubcatchmentError::numerical(
            "cumulative transfer uncertainty charge cannot support declared historical traffic",
        ));
    }
    if ledger.transfer_uncertainty_charge_m3 > permitted {
        return Err(SubcatchmentError::numerical(format!(
            "transfer uncertainty charge={} exceeds conservative permitted budget={permitted}",
            ledger.transfer_uncertainty_charge_m3
        )));
    }
    let adjustment_lower = super::accounting_bounds::adjustment_lower(
        context,
        ledger.soil_water_adjustment_volume_m3,
        steps,
    )?;
    if ledger.soil_water_adjustment_abs_bound_m3 > context.config.max_cumulative_water_adjustment_m3
        || ledger.soil_water_adjustment_abs_bound_m3 < adjustment_lower
        || (ledger.soil_water_adjustment_abs_bound_m3 == 0.0
            && ledger.soil_water_adjustment_volume_m3 > 0.0)
    {
        return Err(SubcatchmentError::numerical(
            "reported soil corrections exceed their independent cumulative admission budget",
        ));
    }
    let calculated = (ledger.storage_m3 - ledger.initial_storage_m3)
        - (ledger.applied_rain_volume_m3 + ledger.external_runon_volume_m3
            - ledger.infiltration_volume_m3
            - ledger.evaporation_volume_m3
            - ledger.outlet_volume_m3
            + ledger.soil_water_adjustment_volume_m3);
    if ledger.residual_m3 != calculated || !ledger.residual_m3.is_finite() {
        return Err(SubcatchmentError::numerical(
            "stored cumulative residual differs from complete represented accounting",
        ));
    }
    check_residual(
        context,
        calculated,
        counters[..8].iter().copied().fold(0.0, f64::max),
        "cumulative",
    )
}

fn check_residual<R>(
    context: &TensorSubcatchmentContext<R>,
    residual: f64,
    scale: f64,
    quantity: &str,
) -> Result<(), SubcatchmentError>
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
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let tolerance =
        context.config.reservoir.volume_absolute + context.config.reservoir.volume_relative * scale;
    if !residual.is_finite()
        || !scale.is_finite()
        || !tolerance.is_finite()
        || residual.abs() > tolerance
    {
        return Err(SubcatchmentError::numerical(format!(
            "{quantity} residual={residual}, scale={scale}, tolerance={tolerance}: refine interval or rescale support"
        )));
    }
    Ok(())
}
