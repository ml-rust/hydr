use super::snapshot::{
    ChannelPoolCounterData, ChannelPoolIntervalAccountingData, ChannelPoolIntervalSnapshotData,
    ChannelPoolNativeIntervalData, ChannelPoolSnapshotData,
};
use super::{ChannelPoolForcingInterval, ChannelPoolInitialState, ChannelPoolSetup};
use crate::channel::{ChannelBoundary, ChannelSnapshotData};
use crate::storage::LevelPoolSnapshotData;
use crate::structures::identity::bits;

pub(super) fn same_setup(a: &ChannelPoolSetup, b: &ChannelPoolSetup) -> bool {
    a.vertical_datum == b.vertical_datum
        && a.nonoverlap_assertion == b.nonoverlap_assertion
        && a.reaches.len() == b.reaches.len()
        && a.reaches
            .iter()
            .zip(&b.reaches)
            .all(|(a, b)| a.id == b.id && a.definition.same_bits(&b.definition))
        && a.pools.len() == b.pools.len()
        && a.pools.iter().zip(&b.pools).all(|(a, b)| a.same_bits(b))
        && a.connections.len() == b.connections.len()
        && a.connections.iter().zip(&b.connections).all(|(a, b)| {
            a.id == b.id
                && a.reach_id == b.reach_id
                && a.endpoint == b.endpoint
                && a.pool_id == b.pool_id
                && a.physical_connection_assertion == b.physical_connection_assertion
                && a.quiescent_reservoir_assertion == b.quiescent_reservoir_assertion
                && a.entrance_loss_coefficient.to_bits() == b.entrance_loss_coefficient.to_bits()
        })
        && a.config.storage.same_bits(&b.config.storage)
        && a.config.iteration_limit == b.config.iteration_limit
        && a.config.retry_limit == b.config.retry_limit
        && bits(&config_values(a), &config_values(b))
}

fn config_values(setup: &ChannelPoolSetup) -> [f64; 8] {
    let c = &setup.config;
    [
        c.head_absolute_m,
        c.head_relative,
        c.transfer_absolute_m3,
        c.transfer_relative,
        c.mass_absolute_m3,
        c.mass_relative,
        c.minimum_step_s,
        c.maximum_step_s,
    ]
}

pub(super) fn same_initial(a: &ChannelPoolInitialState, b: &ChannelPoolInitialState) -> bool {
    a.time_s.to_bits() == b.time_s.to_bits()
        && a.reaches.len() == b.reaches.len()
        && a.reaches.iter().zip(&b.reaches).all(|(a, b)| {
            a.time_s.to_bits() == b.time_s.to_bits()
                && bits(&a.volume_m3, &b.volume_m3)
                && bits(
                    &a.longitudinal_momentum_m4_per_s,
                    &b.longitudinal_momentum_m4_per_s,
                )
        })
        && a.pools.len() == b.pools.len()
        && a.pools
            .iter()
            .zip(&b.pools)
            .all(|(a, b)| a.volume_m3.to_bits() == b.volume_m3.to_bits())
}

pub(super) fn same_data(a: &ChannelPoolSnapshotData, b: &ChannelPoolSnapshotData) -> bool {
    same_setup(&a.setup, &b.setup)
        && same_initial(&a.initial, &b.initial)
        && same_reaches(&a.reaches, &b.reaches)
        && same_pools(&a.pools, &b.pools)
        && same_counters(&a.counters, &b.counters)
        && match (&a.last_interval, &b.last_interval) {
            (None, None) => true,
            (Some(a), Some(b)) => same_interval(a, b),
            _ => false,
        }
}

pub(super) fn same_reaches(a: &[ChannelSnapshotData], b: &[ChannelSnapshotData]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.same_bits(b))
}

pub(super) fn same_pools(a: &LevelPoolSnapshotData, b: &LevelPoolSnapshotData) -> bool {
    a.source_policies.is_none()
        && b.source_policies.is_none()
        && a.config.same_bits(&b.config)
        && a.elements.len() == b.elements.len()
        && a.elements
            .iter()
            .zip(&b.elements)
            .all(|(a, b)| a.same_bits(b))
        && a.time.to_bits() == b.time.to_bits()
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
        && a.accepted_steps == b.accepted_steps
        && match (&a.last_step, &b.last_step) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same_bits(b),
            _ => false,
        }
}

fn same_interval(a: &ChannelPoolIntervalSnapshotData, b: &ChannelPoolIntervalSnapshotData) -> bool {
    same_reaches(&a.previous.reaches, &b.previous.reaches)
        && same_pools(&a.previous.pools, &b.previous.pools)
        && same_counters(&a.previous.counters, &b.previous.counters)
        && same_forcing(&a.forcing, &b.forcing)
        && a.requested_end_s.to_bits() == b.requested_end_s.to_bits()
        && a.iterations == b.iterations
        && a.retries == b.retries
        && same_native_interval(&a.interval, &b.interval)
        && same_accounting(&a.accounting, &b.accounting)
}

fn same_forcing(a: &ChannelPoolForcingInterval, b: &ChannelPoolForcingInterval) -> bool {
    bits(&[a.start_s, a.end_s], &[b.start_s, b.end_s])
        && a.reaches.len() == b.reaches.len()
        && a.reaches.iter().zip(&b.reaches).all(|(a, b)| {
            a.reach_id == b.reach_id
                && same_boundary(a.left.as_ref(), b.left.as_ref())
                && same_boundary(a.right.as_ref(), b.right.as_ref())
                && a.lateral_inputs.len() == b.lateral_inputs.len()
                && a.lateral_inputs
                    .iter()
                    .zip(&b.lateral_inputs)
                    .all(|(a, b)| a.same_bits(b))
        })
}

fn same_boundary(a: Option<&ChannelBoundary>, b: Option<&ChannelBoundary>) -> bool {
    match (a, b) {
        (None, None) | (Some(ChannelBoundary::Wall), Some(ChannelBoundary::Wall)) => true,
        (
            Some(ChannelBoundary::Stage { elevation_m: a }),
            Some(ChannelBoundary::Stage { elevation_m: b }),
        ) => a.to_bits() == b.to_bits(),
        (
            Some(ChannelBoundary::Discharge {
                discharge_m3_per_s: a,
            }),
            Some(ChannelBoundary::Discharge {
                discharge_m3_per_s: b,
            }),
        ) => a.to_bits() == b.to_bits(),
        (
            Some(ChannelBoundary::Exterior {
                depth_m: ah,
                velocity_m_per_s: au,
            }),
            Some(ChannelBoundary::Exterior {
                depth_m: bh,
                velocity_m_per_s: bu,
            }),
        ) => bits(&[*ah, *au], &[*bh, *bu]),
        (
            Some(ChannelBoundary::QuiescentReservoir {
                elevation_m: ah,
                entrance_loss_coefficient: ak,
            }),
            Some(ChannelBoundary::QuiescentReservoir {
                elevation_m: bh,
                entrance_loss_coefficient: bk,
            }),
        ) => bits(&[*ah, *ak], &[*bh, *bk]),
        _ => false,
    }
}

pub(super) fn same_counters(a: &ChannelPoolCounterData, b: &ChannelPoolCounterData) -> bool {
    bits(&a.connection_in_m3, &b.connection_in_m3)
        && bits(&a.connection_out_m3, &b.connection_out_m3)
        && bits(&a.connection_in_bound_m3, &b.connection_in_bound_m3)
        && bits(&a.connection_out_bound_m3, &b.connection_out_bound_m3)
        && bits(&a.external_in_m3, &b.external_in_m3)
        && bits(&a.external_out_m3, &b.external_out_m3)
        && bits(&a.external_in_bound_m3, &b.external_in_bound_m3)
        && bits(&a.external_out_bound_m3, &b.external_out_bound_m3)
        && bits(
            &a.pool_incoming_assembly_bound_m3,
            &b.pool_incoming_assembly_bound_m3,
        )
        && bits(
            &a.pool_outgoing_assembly_bound_m3,
            &b.pool_outgoing_assembly_bound_m3,
        )
        && bits(
            &a.historical_physical_charge_m3,
            &b.historical_physical_charge_m3,
        )
}

pub(super) fn same_native_interval(
    a: &ChannelPoolNativeIntervalData,
    b: &ChannelPoolNativeIntervalData,
) -> bool {
    bits(&a.input_packet, &b.input_packet)
        && bits(&a.output_packet, &b.output_packet)
        && bits(&a.midpoint_pool_stages_m, &b.midpoint_pool_stages_m)
        && bits(&a.connection_volume_m3, &b.connection_volume_m3)
        && bits(
            &a.connection_momentum_m4_per_s,
            &b.connection_momentum_m4_per_s,
        )
        && bits(&a.external_volume_m3, &b.external_volume_m3)
        && bits(&a.cut_available_lower_m3, &b.cut_available_lower_m3)
        && bits(&a.cut_outgoing_upper_m3, &b.cut_outgoing_upper_m3)
        && bits(&a.pool_incoming_bound_m3, &b.pool_incoming_bound_m3)
        && bits(&a.pool_outgoing_bound_m3, &b.pool_outgoing_bound_m3)
        && bits(&a.head_residual_m, &b.head_residual_m)
        && bits(&a.transfer_residual_m3, &b.transfer_residual_m3)
}

pub(super) fn same_accounting(
    a: &ChannelPoolIntervalAccountingData,
    b: &ChannelPoolIntervalAccountingData,
) -> bool {
    bits(&a.storage_before_m3, &b.storage_before_m3)
        && bits(&a.storage_after_m3, &b.storage_after_m3)
        && bits(&a.external_in_m3, &b.external_in_m3)
        && bits(&a.external_out_m3, &b.external_out_m3)
        && bits(&a.lateral_in_m3, &b.lateral_in_m3)
        && bits(&a.lateral_out_m3, &b.lateral_out_m3)
        && bits(&a.residual_m3, &b.residual_m3)
        && bits(&a.roundoff_bound_m3, &b.roundoff_bound_m3)
        && bits(&a.physical_charge_m3, &b.physical_charge_m3)
}
