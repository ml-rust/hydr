mod forcing_identity;
mod receipt_identity;

use super::super::{
    ChannelError, ChannelInitialState, ChannelLateralReceipt, ChannelLateralStateData,
    ChannelReachDefinition, ChannelStepDiagnostics, ChannelWaterLedger,
};
pub(super) use forcing_identity::forcing;
use receipt_identity::{boundary_trace, friction_geometry, riemann};

pub(super) fn require(condition: bool, detail: &str) -> Result<(), ChannelError> {
    if condition {
        Ok(())
    } else {
        Err(ChannelError::input(format!(
            "channel checkpoint {detail}: supply an unchanged complete same-build package"
        )))
    }
}

pub(super) fn values(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Compare both parts of a two-part state bit for bit.
pub(super) fn point(high: &[f64], low: &[f64], other_high: &[f64], other_low: &[f64]) -> bool {
    values(high, other_high) && values(low, other_low)
}

pub(super) fn initial(a: &ChannelInitialState, b: &ChannelInitialState) -> bool {
    a.time_s.to_bits() == b.time_s.to_bits()
        && values(&a.volume_m3, &b.volume_m3)
        && values(
            &a.longitudinal_momentum_m4_per_s,
            &b.longitudinal_momentum_m4_per_s,
        )
}

pub(super) fn definition(a: &ChannelReachDefinition, b: &ChannelReachDefinition) -> bool {
    a.same_bits(b)
}

pub(super) fn lateral_state(a: &ChannelLateralStateData, b: &ChannelLateralStateData) -> bool {
    values(&a.cumulative_volume_m3, &b.cumulative_volume_m3)
        && values(&a.counter_roundoff_bound_m3, &b.counter_roundoff_bound_m3)
        && a.cumulative_assembly_bound_m3.to_bits() == b.cumulative_assembly_bound_m3.to_bits()
}

fn optional(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
        (None, None) => true,
        _ => false,
    }
}

fn source_receipt(a: &ChannelLateralReceipt, b: &ChannelLateralReceipt) -> bool {
    a.input.same_bits(&b.input)
        && a.cell_id == b.cell_id
        && optional(
            a.applied_longitudinal_velocity_m_per_s,
            b.applied_longitudinal_velocity_m_per_s,
        )
        && values(
            &[
                a.volume_m3,
                a.momentum_m4_per_s,
                a.cumulative_volume_before_m3,
                a.cumulative_volume_after_m3,
                a.counter_addition_low_m3,
                a.counter_bound_before_m3,
                a.counter_bound_after_m3,
            ],
            &[
                b.volume_m3,
                b.momentum_m4_per_s,
                b.cumulative_volume_before_m3,
                b.cumulative_volume_after_m3,
                b.counter_addition_low_m3,
                b.counter_bound_before_m3,
                b.counter_bound_after_m3,
            ],
        )
}

pub(super) fn ledger(a: &ChannelWaterLedger, b: &ChannelWaterLedger) -> bool {
    values(
        &[
            a.initial_storage_m3,
            a.storage_m3,
            a.boundary_in_m3,
            a.boundary_out_m3,
            a.lateral_in_m3,
            a.lateral_out_m3,
            a.lateral_counter_abs_bound_m3,
            a.lateral_assembly_abs_bound_m3,
            a.residual_m3,
            a.numerical_transfer_underflow_bound_m3,
            a.storage_selection_abs_bound_m3,
        ],
        &[
            b.initial_storage_m3,
            b.storage_m3,
            b.boundary_in_m3,
            b.boundary_out_m3,
            b.lateral_in_m3,
            b.lateral_out_m3,
            b.lateral_counter_abs_bound_m3,
            b.lateral_assembly_abs_bound_m3,
            b.residual_m3,
            b.numerical_transfer_underflow_bound_m3,
            b.storage_selection_abs_bound_m3,
        ],
    )
}

pub(super) fn diagnostics(a: &ChannelStepDiagnostics, b: &ChannelStepDiagnostics) -> bool {
    a.retries == b.retries
        && a.boundary_traces
            .iter()
            .zip(&b.boundary_traces)
            .all(|(a, b)| boundary_trace(a.as_ref(), b.as_ref()))
        && riemann(a.riemann.as_ref(), b.riemann.as_ref())
        && friction_geometry(a.friction_geometry.as_ref(), b.friction_geometry.as_ref())
        && optional(a.candidate_cfl_limit_s, b.candidate_cfl_limit_s)
        && values(
            &[
                a.start_s,
                a.end_s,
                a.cfl_limit_s,
                a.accepted_start_cfl_limit_s,
                a.depletion_limit_s,
                a.storage_before_m3,
                a.storage_after_m3,
                a.boundary_in_m3,
                a.boundary_out_m3,
                a.lateral_in_m3,
                a.lateral_out_m3,
                a.lateral_assembly_abs_bound_m3,
                a.lateral_counter_addition_abs_bound_m3,
                a.residual_m3,
                a.max_local_residual_m3,
                a.max_local_momentum_residual_m4_per_s,
                a.max_momentum_underflow_bound_m4_per_s,
                a.transfer_underflow_bound_m3,
            ],
            &[
                b.start_s,
                b.end_s,
                b.cfl_limit_s,
                b.accepted_start_cfl_limit_s,
                b.depletion_limit_s,
                b.storage_before_m3,
                b.storage_after_m3,
                b.boundary_in_m3,
                b.boundary_out_m3,
                b.lateral_in_m3,
                b.lateral_out_m3,
                b.lateral_assembly_abs_bound_m3,
                b.lateral_counter_addition_abs_bound_m3,
                b.residual_m3,
                b.max_local_residual_m3,
                b.max_local_momentum_residual_m4_per_s,
                b.max_momentum_underflow_bound_m4_per_s,
                b.transfer_underflow_bound_m3,
            ],
        )
        && values(&a.face_volume_m3, &b.face_volume_m3)
        && values(&a.face_volume_low_m3, &b.face_volume_low_m3)
        && values(&a.storage_selection_bound_m3, &b.storage_selection_bound_m3)
        && values(
            &a.recovery_selection_bound_m2,
            &b.recovery_selection_bound_m2,
        )
        && values(
            &a.discharge_recovery_selection_bound_m3_per_s,
            &b.discharge_recovery_selection_bound_m3_per_s,
        )
        && values(&a.face_momentum_low_m4_per_s, &b.face_momentum_low_m4_per_s)
        && values(
            &a.face_transfer_underflow_bound_m3,
            &b.face_transfer_underflow_bound_m3,
        )
        && values(&a.face_momentum_m4_per_s, &b.face_momentum_m4_per_s)
        && values(
            &a.left_pressure_impulse_m4_per_s,
            &b.left_pressure_impulse_m4_per_s,
        )
        && values(
            &a.right_pressure_impulse_m4_per_s,
            &b.right_pressure_impulse_m4_per_s,
        )
        && values(&a.friction_impulse_m4_per_s, &b.friction_impulse_m4_per_s)
        && values(
            &a.friction_impulse_low_m4_per_s,
            &b.friction_impulse_low_m4_per_s,
        )
        && values(
            &a.momentum_underflow_bound_m4_per_s,
            &b.momentum_underflow_bound_m4_per_s,
        )
        && values(
            &a.lateral_momentum_abs_bound_m4_per_s,
            &b.lateral_momentum_abs_bound_m4_per_s,
        )
        && a.lateral_receipts.len() == b.lateral_receipts.len()
        && a.lateral_receipts
            .iter()
            .zip(&b.lateral_receipts)
            .all(|(a, b)| source_receipt(a, b))
}

pub(super) fn ledger_admission(
    ledger: &ChannelWaterLedger,
    storage: f64,
    original_storage: f64,
    definition: &ChannelReachDefinition,
) -> Result<(), ChannelError> {
    require(
        [
            ledger.initial_storage_m3,
            ledger.storage_m3,
            ledger.boundary_in_m3,
            ledger.boundary_out_m3,
            ledger.lateral_in_m3,
            ledger.lateral_out_m3,
            ledger.lateral_counter_abs_bound_m3,
            ledger.lateral_assembly_abs_bound_m3,
            ledger.residual_m3,
            ledger.numerical_transfer_underflow_bound_m3,
            ledger.storage_selection_abs_bound_m3,
        ]
        .iter()
        .all(|x| x.is_finite())
            && ledger.initial_storage_m3.to_bits() == original_storage.to_bits()
            && ledger.storage_m3.to_bits() == storage.to_bits()
            && ledger.boundary_in_m3 >= 0.0
            && ledger.boundary_out_m3 >= 0.0
            && ledger.lateral_in_m3 >= 0.0
            && ledger.lateral_out_m3 >= 0.0
            && ledger.lateral_counter_abs_bound_m3 >= 0.0
            && ledger.lateral_assembly_abs_bound_m3 >= 0.0
            && ledger.numerical_transfer_underflow_bound_m3 >= 0.0
            && ledger.storage_selection_abs_bound_m3 >= 0.0
            && (matches!(
                definition.geometry,
                super::super::ChannelGeometry::PrismaticSurveyed { .. }
            ) || ledger.storage_selection_abs_bound_m3.to_bits() == 0),
        "ledger storage, original epoch, or external counters disagree",
    )?;
    let residual = (storage - original_storage)
        - ((ledger.boundary_in_m3 - ledger.boundary_out_m3)
            + (ledger.lateral_in_m3 - ledger.lateral_out_m3));
    let scale = original_storage
        .max(storage)
        .max(ledger.boundary_in_m3)
        .max(ledger.boundary_out_m3)
        .max(ledger.lateral_in_m3)
        .max(ledger.lateral_out_m3);
    let tolerance = definition.config.volume_absolute + definition.config.volume_relative * scale;
    let mut charged_residual = residual.abs();
    for charge in [
        ledger.numerical_transfer_underflow_bound_m3,
        ledger.lateral_counter_abs_bound_m3,
        ledger.lateral_assembly_abs_bound_m3,
        ledger.storage_selection_abs_bound_m3,
    ] {
        charged_residual = crate::numerics::outward::add_adjustment_charge(charged_residual, charge)
            .map_err(|error| ChannelError::input(format!(
                "channel checkpoint charged water residual rejects: {error}. Supply an unchanged complete same-build package"
            )))?;
    }
    require(
        residual.to_bits() == ledger.residual_m3.to_bits()
            && tolerance.is_finite()
            && charged_residual <= tolerance,
        "cumulative residual differs from physical water accounting",
    )
}
