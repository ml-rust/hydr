use super::{CoupledLevelPoolSetup, CoupledLevelPoolStepDiagnostics};
use crate::storage::LevelPoolSnapshotData;
use crate::structures::StructureOperatingPoint;
use hydr_structures::internal::{bits, same_elements};

pub(super) fn same_setup(a: &CoupledLevelPoolSetup, b: &CoupledLevelPoolSetup) -> bool {
    let ac = &a.config;
    let bc = &b.config;
    a.source_policies == b.source_policies
        && a.external_inflows == b.external_inflows
        && a.pools.len() == b.pools.len()
        && a.pools.iter().zip(&b.pools).all(|(a, b)| a.same_bits(b))
        && same_elements(&a.structures, &b.structures)
        && ac.storage.same_bits(&bc.storage)
        && bits(
            &[
                ac.head_absolute_m,
                ac.transfer_absolute_m3,
                ac.mass_absolute_m3,
                ac.mass_relative,
                ac.minimum_step_s,
                ac.maximum_step_s,
            ],
            &[
                bc.head_absolute_m,
                bc.transfer_absolute_m3,
                bc.mass_absolute_m3,
                bc.mass_relative,
                bc.minimum_step_s,
                bc.maximum_step_s,
            ],
        )
        && ac.iteration_limit == bc.iteration_limit
        && ac.retry_limit == bc.retry_limit
        && a.pool_support.len() == b.pool_support.len()
        && a.pool_support.iter().zip(&b.pool_support).all(|(a, b)| {
            a.pool_id == b.pool_id
                && a.zero_volume_stage_m.to_bits() == b.zero_volume_stage_m.to_bits()
                && a.zero_volume_limit_assertion == b.zero_volume_limit_assertion
        })
        && a.connections.len() == b.connections.len()
        && a.connections.iter().zip(&b.connections).all(|(a, b)| {
            a.structure_id == b.structure_id
                && [
                    (&a.endpoint_a, &b.endpoint_a),
                    (&a.endpoint_b, &b.endpoint_b),
                ]
                .iter()
                .all(|(a, b)| {
                    a.pool_id == b.pool_id
                        && a.bed_elevation_m.to_bits() == b.bed_elevation_m.to_bits()
                        && a.physical_connection_assertion == b.physical_connection_assertion
                })
        })
        && a.operations.len() == b.operations.len()
        && a.operations.iter().zip(&b.operations).all(|(a, b)| {
            a.structure_id == b.structure_id
                && match (a.operating_point, b.operating_point) {
                    (StructureOperatingPoint::Passive, StructureOperatingPoint::Passive)
                    | (
                        StructureOperatingPoint::PumpStoppedIsolated,
                        StructureOperatingPoint::PumpStoppedIsolated,
                    ) => true,
                    (
                        StructureOperatingPoint::PumpRunning { speed_ratio: a },
                        StructureOperatingPoint::PumpRunning { speed_ratio: b },
                    ) => a.to_bits() == b.to_bits(),
                    (
                        StructureOperatingPoint::GateOpening { opening_m: a },
                        StructureOperatingPoint::GateOpening { opening_m: b },
                    ) => a.to_bits() == b.to_bits(),
                    _ => false,
                }
        })
}
pub(super) fn same_pools(a: &LevelPoolSnapshotData, b: &LevelPoolSnapshotData) -> bool {
    a.schema == b.schema
        && a.contract == b.contract
        && a.source_policies == b.source_policies
        && a.config.same_bits(&b.config)
        && a.time.to_bits() == b.time.to_bits()
        && a.accepted_steps == b.accepted_steps
        && a.elements.len() == b.elements.len()
        && a.elements
            .iter()
            .zip(&b.elements)
            .all(|(a, b)| a.same_bits(b))
        && a.states.len() == b.states.len()
        && a.states
            .iter()
            .zip(&b.states)
            .all(|(a, b)| a.volume_m3.to_bits() == b.volume_m3.to_bits())
        && a.element_ledgers.len() == b.element_ledgers.len()
        && a.element_ledgers
            .iter()
            .zip(&b.element_ledgers)
            .all(|(a, b)| a.same_bits(b))
        && match (&a.last_step, &b.last_step) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same_bits(b),
            _ => false,
        }
}
pub(super) fn same_step(
    a: &CoupledLevelPoolStepDiagnostics,
    b: &CoupledLevelPoolStepDiagnostics,
) -> bool {
    let sources = match (&a.forcing_interval, &b.forcing_interval) {
        (None, None) => true,
        (Some(a), Some(b)) => a.same_bits(b),
        _ => false,
    };
    sources
        && a.external_inflows.len() == b.external_inflows.len()
        && a.external_inflows
            .iter()
            .zip(&b.external_inflows)
            .all(|(a, b)| a.same_bits(b))
        && bits(
            &[
                a.start_s,
                a.end_s,
                a.requested_end_s,
                a.head_residual_m,
                a.transfer_residual_m3,
                a.mass_residual_m3,
                a.aggregation_roundoff_bound_m3,
            ],
            &[
                b.start_s,
                b.end_s,
                b.requested_end_s,
                b.head_residual_m,
                b.transfer_residual_m3,
                b.mass_residual_m3,
                b.aggregation_roundoff_bound_m3,
            ],
        )
        && a.iterations == b.iterations
        && a.retries == b.retries
        && bits(&a.iteration.end_stages_m, &b.iteration.end_stages_m)
        && bits(
            &a.iteration.signed_volumes_m3,
            &b.iteration.signed_volumes_m3,
        )
        && a.iteration.pool_wet == b.iteration.pool_wet
        && a.iteration.attachment_a_wet == b.iteration.attachment_a_wet
        && a.iteration.attachment_b_wet == b.iteration.attachment_b_wet
        && a.transfers.len() == b.transfers.len()
        && a.transfers.iter().zip(&b.transfers).all(|(a, b)| {
            a.structure_id == b.structure_id
                && bits(
                    &[
                        a.proposed_volume_m3,
                        a.signed_volume_m3,
                        a.donor_limit_factor,
                    ],
                    &[
                        b.proposed_volume_m3,
                        b.signed_volume_m3,
                        b.donor_limit_factor,
                    ],
                )
                && same_optional(a.midpoint_stage_a_m, b.midpoint_stage_a_m)
                && same_optional(a.midpoint_stage_b_m, b.midpoint_stage_b_m)
        })
}
fn same_optional(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
        _ => false,
    }
}

pub(super) fn same_external(
    a: &[super::CoupledPoolInflowLedger],
    b: &[super::CoupledPoolInflowLedger],
) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.same_bits(b))
}
