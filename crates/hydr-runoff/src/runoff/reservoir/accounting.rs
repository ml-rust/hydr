//! Complete receipts and independent interval/cumulative volume admission.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

use super::ReservoirError;
use crate::runoff::numerics::reservoir::TensorReservoirStep;
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;

/// Lifetime represented reservoir water accounting, without numerical adjustments.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirLedger {
    /// Original represented storage in m³.
    pub initial_storage_m3: f64,
    /// Current represented storage in m³.
    pub storage_m3: f64,
    /// Lifetime incoming volume in m³.
    pub incoming_volume_m3: f64,
    /// Lifetime exported volume in m³.
    pub export_volume_m3: f64,
    /// Signed cumulative storage residual in m³.
    pub residual_m3: f64,
}

/// Ordered element evidence for one accepted interval. Native endpoint quadrature is not used.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirReceipt {
    /// Ordered element index.
    pub element: usize,
    /// Interval-start depth in metres.
    pub old_depth_m: f64,
    /// Constant incoming rate in m/s.
    pub incoming_rate_m_per_s: f64,
    /// Incoming rate times the original interval in metres.
    pub incoming_depth_m: f64,
    /// Original available depth in metres.
    pub available_depth_m: f64,
    /// Active drainage duration in seconds. Non-draining branches retain zero.
    pub active_duration_s: f64,
    /// Retained lower excess-root bracket in metres.
    pub root_lower_m: f64,
    /// Retained upper excess-root bracket in metres.
    pub root_upper_m: f64,
    /// Accepted represented depth in metres.
    pub depth_m: f64,
    /// Original available depth minus accepted depth, in metres.
    pub export_depth_m: f64,
    /// Interval-start storage in m³.
    pub storage_before_m3: f64,
    /// Accepted storage in m³.
    pub storage_after_m3: f64,
    /// Actual incoming volume in m³.
    pub incoming_volume_m3: f64,
    /// Conservatively defined export volume in m³.
    pub export_volume_m3: f64,
    /// Finite-roughness endpoint depth-law rate in m/s. Instantaneous overflow is undefined.
    pub endpoint_rate_m_per_s: Option<f64>,
    /// Finite-roughness endpoint discharge in m³/s. Instantaneous overflow is undefined.
    pub endpoint_discharge_m3_per_s: Option<f64>,
    /// Active duration times endpoint rate, in metres. Zero when the law gate is inactive.
    pub law_depth_m: f64,
    /// Represented active endpoint-law volume in m³. Zero when the law gate is inactive.
    pub law_volume_m3: f64,
    /// Export minus active endpoint-law volume in m³. Zero for instantaneous overflow.
    pub constitutive_residual_m3: f64,
    /// Constitutive tolerance scale in m³. Zero for instantaneous overflow.
    pub constitutive_scale_m3: f64,
    /// Independent represented storage residual in m³.
    pub water_residual_m3: f64,
    /// Independent storage tolerance scale in m³, without a unit floor.
    pub water_scale_m3: f64,
}

/// Accepted interval, batched root work, aggregate volumes, and complete ordered receipts.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirStepDiagnostics {
    /// Starting accepted time in seconds.
    pub start: f64,
    /// Ending accepted time in seconds.
    pub end: f64,
    /// Exact represented duration `end-start`, in seconds.
    pub dt: f64,
    /// Batched bisection rounds, without upper-bracket expansion.
    pub root_iterations: usize,
    /// Aggregate original storage in m³.
    pub storage_before_m3: f64,
    /// Aggregate accepted storage in m³.
    pub storage_after_m3: f64,
    /// Aggregate incoming volume in m³.
    pub incoming_volume_m3: f64,
    /// Aggregate exported volume in m³.
    pub export_volume_m3: f64,
    /// Sum of admitted finite-roughness constitutive residuals, in m³.
    pub constitutive_residual_m3: f64,
    /// Sum of active constitutive scales, in m³.
    pub constitutive_scale_m3: f64,
    /// Independent aggregate storage residual, in m³.
    pub water_residual_m3: f64,
    /// Aggregate storage tolerance scale, in m³.
    pub water_scale_m3: f64,
    /// Complete ordered per-element witnesses.
    pub receipts: Vec<ReservoirReceipt>,
}

pub(super) struct StepTotals {
    pub before: f64,
    pub after: f64,
    pub incoming: f64,
    pub export: f64,
    pub constitutive: f64,
    pub constitutive_scale: f64,
    pub water: f64,
    pub water_scale: f64,
}

pub(super) fn aggregate_step<R>(
    context: &TensorReservoirContext<R>,
    step: &TensorReservoirStep<R>,
) -> Result<StepTotals, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let before = context.total(&step.storage_before, "aggregate original storage")?;
    let after = context.total(&step.storage_after, "aggregate accepted storage")?;
    let incoming = context.total(&step.incoming_volume, "aggregate incoming volume")?;
    let export = context.total(&step.export_volume, "aggregate export volume")?;
    let constitutive = context.total(
        &step.constitutive_residual,
        "aggregate constitutive residual",
    )?;
    let constitutive_scale =
        context.total(&step.constitutive_scale, "aggregate constitutive scale")?;
    let water = (after - before) - (incoming - export);
    let water_scale = before.max(after).max(incoming).max(export);
    residual(
        context,
        "aggregate constitutive",
        constitutive,
        constitutive_scale,
    )?;
    residual(context, "aggregate water", water, water_scale)?;
    Ok(StepTotals {
        before,
        after,
        incoming,
        export,
        constitutive,
        constitutive_scale,
        water,
        water_scale,
    })
}

impl ReservoirLedger {
    pub(super) fn new(storage: f64) -> Self {
        Self {
            initial_storage_m3: storage,
            storage_m3: storage,
            incoming_volume_m3: 0.0,
            export_volume_m3: 0.0,
            residual_m3: 0.0,
        }
    }
    pub(super) fn advance<R>(
        &self,
        context: &TensorReservoirContext<R>,
        step: &StepTotals,
    ) -> Result<Self, ReservoirError>
    where
        R: Runtime<DType = DType>,
        R::Client: BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + TypeConversionOps<R>
            + UnaryOps<R>
            + BatchedBisectionAlgorithms<R>,
    {
        let mut next = Self {
            initial_storage_m3: self.initial_storage_m3,
            storage_m3: step.after,
            incoming_volume_m3: self.incoming_volume_m3 + step.incoming,
            export_volume_m3: self.export_volume_m3 + step.export,
            residual_m3: 0.0,
        };
        next.residual_m3 = (next.storage_m3 - next.initial_storage_m3)
            - (next.incoming_volume_m3 - next.export_volume_m3);
        check_ledger(context, &next)?;
        Ok(next)
    }
}

pub(super) fn check_ledger<R>(
    context: &TensorReservoirContext<R>,
    ledger: &ReservoirLedger,
) -> Result<(), ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let values = [
        ledger.initial_storage_m3,
        ledger.storage_m3,
        ledger.incoming_volume_m3,
        ledger.export_volume_m3,
    ];
    if values.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(ReservoirError::numerical(
            "cumulative volumes are negative or nonfinite. Rescale support or horizon",
        ));
    }
    let calculated = (ledger.storage_m3 - ledger.initial_storage_m3)
        - (ledger.incoming_volume_m3 - ledger.export_volume_m3);
    if !ledger.residual_m3.is_finite() || ledger.residual_m3 != calculated {
        return Err(ReservoirError::numerical(
            "stored cumulative residual differs from complete represented accounting",
        ));
    }
    residual(
        context,
        "cumulative water",
        calculated,
        values.into_iter().fold(0.0, f64::max),
    )
}

fn residual<R>(
    context: &TensorReservoirContext<R>,
    quantity: &str,
    value: f64,
    scale: f64,
) -> Result<(), ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let tolerance = context.config.volume_absolute + context.config.volume_relative * scale;
    if !value.is_finite()
        || !scale.is_finite()
        || scale < 0.0
        || !tolerance.is_finite()
        || value.abs() > tolerance
    {
        return Err(ReservoirError::numerical(format!(
            "{quantity} residual={value}, scale={scale}, tolerance={tolerance}: refine interval or rescale the component"
        )));
    }
    Ok(())
}
