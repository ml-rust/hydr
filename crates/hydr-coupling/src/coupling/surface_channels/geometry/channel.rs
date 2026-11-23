//! Channel cross-section epochs in the surface/channel coordinator.
//!
//! Epoch times join the coordinator event set so no interval straddles a section change. At an
//! epoch, cell wet area, discharge, volume, and momentum stay bit-identical while stage recomputes
//! from the new section's certified area-to-stage recovery. Wet area above the new section's bank
//! limit rejects without retry.

use super::SurfaceChannelGeometryReceipt;
use crate::channel::{
    ChannelGeometry, ChannelReach, ChannelSectionSchedule, ChannelSnapshot, ChannelSnapshotData,
    active_channel_section_epochs, active_geometry_at,
    check_channel_section_schedules as check_reach_schedules, geometry_digest_bits,
    next_channel_section_event_after,
};
use crate::coupling::surface_channels::{
    SurfaceChannelError, SurfaceChannelNode, SurfaceChannelSetup,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedMappedBisectionAlgorithms;
use solvr::optimize::roots::{
    BatchedBisectionAlgorithms, BatchedNewtonAlgorithms, CertifiedBisectionAlgorithms,
    CertifiedUpperBracketAlgorithms,
};
use std::collections::BTreeSet;

/// Minimum bed elevation of `geometry` at cell `index`.
fn cell_bed(geometry: &ChannelGeometry, index: usize) -> Option<f64> {
    match geometry {
        ChannelGeometry::Rectangular { cells } => cells.get(index).map(|c| c.bed_elevation_m),
        ChannelGeometry::PrismaticTriangular {
            bed_elevation_m,
            cells,
            ..
        } => (index < cells.len()).then_some(*bed_elevation_m),
        ChannelGeometry::PrismaticSurveyed { profile, cells, .. } => {
            if index >= cells.len() {
                return None;
            }
            profile
                .points
                .iter()
                .map(|p| p.elevation_m)
                .reduce(f64::min)
                .or(Some(profile.maximum_stage_m))
        }
    }
}

/// Checks `setup.channel_section_schedules` against reaches, bank links, structure attachments,
/// ports, and pool ends.
pub(in super::super) fn check_channel_section_schedules(
    setup: &SurfaceChannelSetup,
) -> Result<(), SurfaceChannelError> {
    if setup.channel_section_schedules.is_empty() {
        return Ok(());
    }
    let reaches: Vec<(&str, &crate::channel::ChannelReachDefinition)> = setup
        .reaches
        .iter()
        .map(|r| (r.id.as_str(), &r.definition))
        .collect();
    check_reach_schedules(&reaches, &setup.channel_section_schedules).map_err(|error| {
        SurfaceChannelError::input(format!("channel_section_schedules: {error}"))
    })?;
    let fixed_reaches: BTreeSet<&str> = setup
        .ports
        .iter()
        .map(|p| p.reach_id.as_str())
        .chain(setup.pool_ends.iter().map(|e| e.reach_id.as_str()))
        .collect();
    for schedule in &setup.channel_section_schedules {
        if fixed_reaches.contains(schedule.reach_id.as_str()) {
            return Err(SurfaceChannelError::input(format!(
                "channel_section_schedules {:?}: reach={:?} owns a port or pool end; section epochs on ported or pool-ended reaches are not supported",
                schedule.schedule_id, schedule.reach_id
            )));
        }
        for (epoch_index, epoch) in schedule.epochs.iter().enumerate() {
            for link in &setup.links {
                if link.reach_id == schedule.reach_id
                    && let Some(bed) = cell_bed(&epoch.geometry, link.channel_cell)
                    && link.crest_elevation_m < bed
                {
                    return Err(SurfaceChannelError::input(format!(
                        "unobstructed-connection: channel_section_schedules {:?} epoch {epoch_index} at time_s={}: reach={:?} cell={} new bed={bed} exceeds bank link {:?} crest={}",
                        schedule.schedule_id,
                        epoch.time_s,
                        schedule.reach_id,
                        link.channel_cell,
                        link.id,
                        link.crest_elevation_m
                    )));
                }
            }
            for structure in &setup.structures {
                for attachment in [&structure.endpoint_a, &structure.endpoint_b] {
                    if let SurfaceChannelNode::ChannelCell { reach_id, cell } = &attachment.node
                        && reach_id == &schedule.reach_id
                        && let Some(bed) = cell_bed(&epoch.geometry, *cell)
                        && attachment.bed_elevation_m < bed
                    {
                        return Err(SurfaceChannelError::input(format!(
                            "unobstructed-pool-connection: channel_section_schedules {:?} epoch {epoch_index} at time_s={}: reach={:?} cell={cell} new bed={bed} exceeds structure {:?} attachment bed={}",
                            schedule.schedule_id,
                            epoch.time_s,
                            schedule.reach_id,
                            structure.element.id,
                            attachment.bed_elevation_m
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Earliest channel section epoch time strictly after `start_s`, or `None`.
pub(in super::super) fn next_channel_section_event(
    schedules: &[ChannelSectionSchedule],
    start_s: f64,
) -> Option<f64> {
    next_channel_section_event_after(schedules, start_s)
}

/// Active channel section epoch per schedule at `time_s`.
pub(in super::super) fn channel_section_epochs_at(
    schedules: &[ChannelSectionSchedule],
    time_s: f64,
) -> Vec<usize> {
    active_channel_section_epochs(schedules, time_s)
}

/// Updated reaches and receipts produced by applying active channel section epochs.
pub(in super::super) type AppliedSections<R> = (
    Option<Vec<ChannelReach<R>>>,
    Vec<SurfaceChannelGeometryReceipt>,
);

/// Applies any channel section epochs active at `start_s` to `reaches`, keeping cell volumes,
/// momenta, and water storage bit-identical, and returns receipts for every schedule active at
/// `start_s`.
pub(in super::super) fn apply_channel_sections<R: Runtime<DType = DType>>(
    setup: &SurfaceChannelSetup,
    reaches: &[ChannelReach<R>],
    start_s: f64,
) -> Result<AppliedSections<R>, SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    let schedules = &setup.channel_section_schedules;
    if schedules.is_empty() {
        return Ok((None, Vec::new()));
    }
    let epochs = active_channel_section_epochs(schedules, start_s);
    let mut updated: Option<Vec<ChannelReach<R>>> = None;
    let mut receipts = Vec::new();
    for (index, (decl, reach)) in setup.reaches.iter().zip(reaches).enumerate() {
        let scheduled = schedules
            .iter()
            .zip(&epochs)
            .find(|(s, _)| s.reach_id == decl.id);
        let Some((schedule, &epoch)) = scheduled else {
            continue;
        };
        let active = active_geometry_at(&decl.definition.geometry, schedule, start_s);
        let old_bits = geometry_digest_bits(&reach.definition().geometry);
        let new_bits = geometry_digest_bits(active);
        let at_epoch = schedule
            .epochs
            .iter()
            .any(|e| e.time_s.to_bits() == start_s.to_bits());
        if old_bits != new_bits {
            let next = reach.coupling_with_geometry(active.clone()).map_err(|error| {
                SurfaceChannelError::input(format!(
                    "channel-section-bank-limit: channel_section_schedules {:?} at time_s={start_s} reach={:?}: {error}",
                    schedule.schedule_id, schedule.reach_id
                ))
            })?;
            let list = updated.get_or_insert_with(|| reaches.to_vec());
            list[index] = next;
        }
        if at_epoch || old_bits != new_bits {
            receipts.push(SurfaceChannelGeometryReceipt {
                schedule_id: schedule.schedule_id.clone(),
                element_id: schedule.reach_id.clone(),
                time_s: start_s,
                epoch,
                old_value_bits: old_bits,
                new_value_bits: new_bits,
                storage_before_m3: 0.0,
                storage_after_m3: 0.0,
            });
        }
    }
    Ok((updated, receipts))
}

/// Restores a trusted reach snapshot, allowing section geometry epochs when `has_section`.
pub(in super::super) fn restore_reach<R: Runtime<DType = DType>>(
    reach: &ChannelReach<R>,
    trusted: &ChannelSnapshot<R>,
    has_section: bool,
    reach_id: &str,
) -> Result<ChannelReach<R>, SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let res = if has_section {
        reach.coupling_restore_with_geometry(trusted)
    } else {
        reach.restore(trusted)
    };
    res.map_err(|source| SurfaceChannelError::Channel {
        reach_id: reach_id.to_string(),
        source,
    })
}

/// Imports a serialized reach snapshot, allowing section geometry epochs when `has_section`.
pub(in super::super) fn import_reach<R: Runtime<DType = DType>>(
    reach: &ChannelReach<R>,
    data: &ChannelSnapshotData,
    has_section: bool,
    reach_id: &str,
) -> Result<ChannelReach<R>, SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let res = if has_section {
        reach.coupling_import_snapshot_with_geometry(data)
    } else {
        reach.coupling_import_snapshot(data)
    };
    res.map_err(|source| SurfaceChannelError::Channel {
        reach_id: reach_id.to_string(),
        source,
    })
}
