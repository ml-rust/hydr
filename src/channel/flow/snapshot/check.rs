use super::super::{
    ChannelError, ChannelInitialState, ChannelReachDefinition, ChannelStepDiagnostics,
    ChannelWaterLedger,
};

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

pub(super) fn initial(a: &ChannelInitialState, b: &ChannelInitialState) -> bool {
    a.time_s.to_bits() == b.time_s.to_bits()
        && values(&a.volume_m3, &b.volume_m3)
        && values(
            &a.longitudinal_momentum_m4_per_s,
            &b.longitudinal_momentum_m4_per_s,
        )
}

pub(super) fn definition(a: &ChannelReachDefinition, b: &ChannelReachDefinition) -> bool {
    let ac = &a.config;
    let bc = &b.config;
    a.vertical_datum == b.vertical_datum
        && a.provenance == b.provenance
        && a.cells.len() == b.cells.len()
        && a.cells.iter().zip(&b.cells).all(|(a, b)| {
            a.id == b.id
                && values(
                    &[
                        a.length_m,
                        a.width_m,
                        a.bed_elevation_m,
                        a.maximum_depth_m,
                        a.manning_n,
                    ],
                    &[
                        b.length_m,
                        b.width_m,
                        b.bed_elevation_m,
                        b.maximum_depth_m,
                        b.manning_n,
                    ],
                )
        })
        && ac.max_retries == bc.max_retries
        && ac.root_iterations == bc.root_iterations
        && values(
            &[
                ac.gravity,
                ac.cfl,
                ac.max_step,
                ac.min_step,
                ac.root_absolute_depth,
                ac.root_relative_depth,
                ac.volume_absolute,
                ac.volume_relative,
                ac.momentum_absolute,
                ac.momentum_relative,
            ],
            &[
                bc.gravity,
                bc.cfl,
                bc.max_step,
                bc.min_step,
                bc.root_absolute_depth,
                bc.root_relative_depth,
                bc.volume_absolute,
                bc.volume_relative,
                bc.momentum_absolute,
                bc.momentum_relative,
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
            a.residual_m3,
            a.numerical_transfer_underflow_bound_m3,
        ],
        &[
            b.initial_storage_m3,
            b.storage_m3,
            b.boundary_in_m3,
            b.boundary_out_m3,
            b.residual_m3,
            b.numerical_transfer_underflow_bound_m3,
        ],
    )
}

pub(super) fn diagnostics(a: &ChannelStepDiagnostics, b: &ChannelStepDiagnostics) -> bool {
    a.retries == b.retries
        && values(
            &[
                a.start_s,
                a.end_s,
                a.cfl_limit_s,
                a.depletion_limit_s,
                a.storage_before_m3,
                a.storage_after_m3,
                a.boundary_in_m3,
                a.boundary_out_m3,
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
                b.depletion_limit_s,
                b.storage_before_m3,
                b.storage_after_m3,
                b.boundary_in_m3,
                b.boundary_out_m3,
                b.residual_m3,
                b.max_local_residual_m3,
                b.max_local_momentum_residual_m4_per_s,
                b.max_momentum_underflow_bound_m4_per_s,
                b.transfer_underflow_bound_m3,
            ],
        )
        && values(&a.face_volume_m3, &b.face_volume_m3)
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
            &a.momentum_underflow_bound_m4_per_s,
            &b.momentum_underflow_bound_m4_per_s,
        )
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
            ledger.residual_m3,
            ledger.numerical_transfer_underflow_bound_m3,
        ]
        .iter()
        .all(|x| x.is_finite())
            && ledger.initial_storage_m3.to_bits() == original_storage.to_bits()
            && ledger.storage_m3.to_bits() == storage.to_bits()
            && ledger.boundary_in_m3 >= 0.0
            && ledger.boundary_out_m3 >= 0.0
            && ledger.numerical_transfer_underflow_bound_m3 >= 0.0,
        "ledger storage, original epoch, or external counters disagree",
    )?;
    let residual = (storage - original_storage) - (ledger.boundary_in_m3 - ledger.boundary_out_m3);
    let scale = original_storage
        .max(storage)
        .max(ledger.boundary_in_m3)
        .max(ledger.boundary_out_m3);
    let tolerance = definition.config.volume_absolute + definition.config.volume_relative * scale;
    let charged_residual = crate::soil::admission::add_adjustment_charge(
        residual.abs(),
        ledger.numerical_transfer_underflow_bound_m3,
    )
    .map_err(|error| {
        ChannelError::input(format!(
            "channel checkpoint charged water residual rejects: {error}. Supply an unchanged complete same-build package"
        ))
    })?;
    require(
        residual.to_bits() == ledger.residual_m3.to_bits()
            && tolerance.is_finite()
            && charged_residual <= tolerance,
        "cumulative residual differs from physical water accounting",
    )
}
