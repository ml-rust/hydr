//! Surface bed elevation epochs in the surface/channel coordinator.
//!
//! Epoch times join the coordinator event set so no interval straddles a bed step. At an epoch,
//! depth `h`, momenta `qx`/`qy`, and total water storage stay bit-identical while surface stage
//! shifts by the bed change.

use super::SurfaceChannelGeometryReceipt;
use crate::coupling::surface_channels::{
    SurfaceChannelError, SurfaceChannelNode, SurfaceChannelSetup,
};
use crate::solver::{
    Solver, SurfaceBedSchedule, active_bed_at, active_surface_bed_epochs, bed_digest_bits,
    check_surface_bed_schedules as check_solver_schedules, next_surface_bed_event_after,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, ReduceOps, ScalarOps, ShapeOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

/// Checks `setup.surface_bed_schedules` against the surface domain, bank links, structure
/// attachments, and ports.
pub(in super::super) fn check_surface_bed_schedules(
    setup: &SurfaceChannelSetup,
) -> Result<(), SurfaceChannelError> {
    if setup.surface_bed_schedules.is_empty() {
        return Ok(());
    }
    check_solver_schedules(&setup.surface.domain, &setup.surface_bed_schedules)
        .map_err(|error| SurfaceChannelError::input(format!("surface_bed_schedules: {error}")))?;
    let (nx, ny) = setup.surface.domain.grid().dimensions();
    let mut port_cells = std::collections::BTreeSet::new();
    for port in &setup.ports {
        for face in &port.faces {
            match face.axis {
                crate::solver::Axis::X => {
                    if face.column > 0 {
                        port_cells.insert(face.row * nx + (face.column - 1));
                    }
                    if face.column < nx {
                        port_cells.insert(face.row * nx + face.column);
                    }
                }
                crate::solver::Axis::Y => {
                    if face.row > 0 {
                        port_cells.insert((face.row - 1) * nx + face.column);
                    }
                    if face.row < ny {
                        port_cells.insert(face.row * nx + face.column);
                    }
                }
            }
        }
    }
    for schedule in &setup.surface_bed_schedules {
        for (epoch_index, epoch) in schedule.epochs.iter().enumerate() {
            for update in &epoch.updates {
                if port_cells.contains(&update.cell) {
                    return Err(SurfaceChannelError::input(format!(
                        "surface_bed_schedules {:?} epoch {epoch_index} at time_s={}: cell={} belongs to a reach-end port face; bed epochs on ported cells are not supported",
                        schedule.schedule_id, epoch.time_s, update.cell
                    )));
                }
                for link in &setup.links {
                    if link.surface_cell == update.cell
                        && link.crest_elevation_m < update.bed_elevation_m
                    {
                        return Err(SurfaceChannelError::input(format!(
                            "unobstructed-connection: surface_bed_schedules {:?} epoch {epoch_index} at time_s={}: cell={} new bed={} exceeds bank link {:?} crest={}",
                            schedule.schedule_id,
                            epoch.time_s,
                            update.cell,
                            update.bed_elevation_m,
                            link.id,
                            link.crest_elevation_m
                        )));
                    }
                }
                for structure in &setup.structures {
                    for attachment in [&structure.endpoint_a, &structure.endpoint_b] {
                        if attachment.node == SurfaceChannelNode::SurfaceCell(update.cell)
                            && attachment.bed_elevation_m < update.bed_elevation_m
                        {
                            return Err(SurfaceChannelError::input(format!(
                                "unobstructed-pool-connection: surface_bed_schedules {:?} epoch {epoch_index} at time_s={}: cell={} new bed={} exceeds structure {:?} attachment bed={}",
                                schedule.schedule_id,
                                epoch.time_s,
                                update.cell,
                                update.bed_elevation_m,
                                structure.element.id,
                                attachment.bed_elevation_m
                            )));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Earliest surface bed epoch time strictly after `start_s`, or `None`.
pub(in super::super) fn next_surface_bed_event(
    schedules: &[SurfaceBedSchedule],
    start_s: f64,
) -> Option<f64> {
    next_surface_bed_event_after(schedules, start_s)
}

/// Restores `surface` from `data` using the bed-epoch path when `allow_bed_epochs` is true.
pub(in super::super) fn restore_surface_data<R: Runtime<DType = DType>>(
    surface: &mut Solver<R>,
    data: &crate::solver::SnapshotData,
    ports: Option<&hydr_solver::internal::SurfacePorts<R>>,
    allow_bed_epochs: bool,
) -> Result<(), SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>
        + HypotOps<R>
        + CumulativeOps<R>
        + ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    if allow_bed_epochs {
        surface.coupling_restore_data_with_bed(data, ports)?;
    } else {
        surface.coupling_restore_data(data, ports)?;
    }
    Ok(())
}

/// Active surface bed epoch per schedule at `time_s`.
pub(in super::super) fn surface_bed_epochs_at(
    schedules: &[SurfaceBedSchedule],
    time_s: f64,
) -> Vec<usize> {
    active_surface_bed_epochs(schedules, time_s)
}

/// Applies any surface bed epochs active at `start_s` to `surface`, keeping depth, momenta, and
/// water storage bit-identical, and returns receipts for every schedule active at `start_s`.
pub(in super::super) fn apply_surface_bed<R: Runtime<DType = DType>>(
    declared_bed: &[f64],
    schedules: &[SurfaceBedSchedule],
    surface: &Solver<R>,
    start_s: f64,
) -> Result<(Solver<R>, Vec<SurfaceChannelGeometryReceipt>), SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>
        + HypotOps<R>
        + CumulativeOps<R>
        + ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    let old_bed = surface.domain().bed();
    let target_bed = active_bed_at(declared_bed, schedules, start_s);
    let changed = old_bed.len() != target_bed.len()
        || old_bed
            .iter()
            .zip(&target_bed)
            .any(|(a, b)| a.to_bits() != b.to_bits());
    let epochs = active_surface_bed_epochs(schedules, start_s);
    let old_bits = bed_digest_bits(old_bed);
    let new_bits = bed_digest_bits(&target_bed);
    let mut receipts = Vec::new();
    for (schedule, epoch) in schedules.iter().zip(epochs) {
        let at_epoch = schedule
            .epochs
            .iter()
            .any(|e| e.time_s.to_bits() == start_s.to_bits());
        if at_epoch || changed {
            receipts.push(SurfaceChannelGeometryReceipt {
                schedule_id: schedule.schedule_id.clone(),
                element_id: "surface".into(),
                time_s: start_s,
                epoch,
                old_value_bits: old_bits,
                new_value_bits: new_bits,
                storage_before_m3: 0.0,
                storage_after_m3: 0.0,
            });
        }
    }
    if !changed {
        return Ok((surface.coupling_clone(), receipts));
    }
    let next = surface.coupling_with_bed(target_bed)?;
    Ok((next, receipts))
}
