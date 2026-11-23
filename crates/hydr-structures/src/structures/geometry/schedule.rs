//! Checked structure geometry schedules and admission against declared elements.

use super::evaluation::apply_at;
use super::knot::{
    DiscreteGeometryKnot, ScalarGeometryParameter, StructureGeometryKnot, StructureGeometryTrack,
    TrackKey,
};
use crate::structures::blockage::effective;
use crate::structures::definition::text;
use crate::structures::{StructureElement, StructureError, StructureLaw};
use std::collections::BTreeSet;

/// Deterministic geometry schedule for one parameter of one structure, with provenance evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureGeometrySchedule {
    /// Unique schedule identity.
    pub schedule_id: String,
    /// Target structure identity.
    pub structure_id: String,
    /// Parameter track and its knots.
    pub track: StructureGeometryTrack,
    /// Provenance evidence for the scheduled geometry change.
    pub evidence: String,
}

impl StructureGeometrySchedule {
    /// Checks non-empty text, strictly increasing finite knot times, and track-local value domains.
    pub fn check(&self) -> Result<(), StructureError> {
        text(&self.schedule_id, "geometry_schedule.schedule_id")?;
        text(&self.structure_id, "geometry_schedule.structure_id")?;
        text(&self.evidence, "geometry_schedule.evidence")?;
        let ctx = format!(
            "schedule={:?},structure={:?},track={}",
            self.schedule_id,
            self.structure_id,
            self.track.name()
        );
        match &self.track {
            StructureGeometryTrack::Scalar { parameter, knots } => {
                check_scalar_knots(&ctx, *parameter, knots)
            }
            StructureGeometryTrack::CulvertSediment {
                bed_manning_n,
                knots,
            } => {
                if !(bed_manning_n.is_finite() && *bed_manning_n > 0.0) {
                    return Err(StructureError::definition(format!(
                        "{ctx},bed_manning_n={bed_manning_n:?}: supply a finite positive bed Manning n in s/m^(1/3)"
                    )));
                }
                check_times(&ctx, knots.iter().map(|k| k.time_s))?;
                for (index, knot) in knots.iter().enumerate() {
                    for (side, value) in [("left", knot.left), ("right", knot.right)] {
                        if !(value.is_finite() && value >= 0.0) {
                            return Err(StructureError::definition(format!(
                                "{ctx},knot={index},{side}={value:?}: supply a finite nonnegative sediment depth in metres"
                            )));
                        }
                    }
                }
                Ok(())
            }
            StructureGeometryTrack::CulvertClosedBarrels { knots } => {
                check_discrete_knots(&ctx, knots)
            }
            StructureGeometryTrack::Closed { knots } => check_discrete_knots(&ctx, knots),
        }
    }
}

fn check_times(ctx: &str, times: impl Iterator<Item = f64>) -> Result<(), StructureError> {
    let mut prev: Option<f64> = None;
    let mut count = 0usize;
    for (index, time) in times.enumerate() {
        if !time.is_finite() {
            return Err(StructureError::definition(format!(
                "{ctx},knot={index},time_s={time:?}: supply a finite knot time in seconds"
            )));
        }
        if let Some(before) = prev
            && time <= before
        {
            return Err(StructureError::definition(format!(
                "{ctx},knot={index},time_s={time:?},previous={before:?}: knot times must strictly increase"
            )));
        }
        prev = Some(time);
        count += 1;
    }
    if count == 0 {
        return Err(StructureError::definition(format!(
            "{ctx}: supply at least one geometry knot"
        )));
    }
    Ok(())
}

fn check_scalar_knots(
    ctx: &str,
    parameter: ScalarGeometryParameter,
    knots: &[StructureGeometryKnot],
) -> Result<(), StructureError> {
    check_times(ctx, knots.iter().map(|k| k.time_s))?;
    let is_fraction = matches!(
        parameter,
        ScalarGeometryParameter::WeirBlockage
            | ScalarGeometryParameter::OrificeBlockage
            | ScalarGeometryParameter::GateWidthBlockage
            | ScalarGeometryParameter::GateOpeningRangeBlockage
            | ScalarGeometryParameter::CulvertFaceABlockage
            | ScalarGeometryParameter::CulvertFaceBBlockage
    );
    let is_positive = matches!(
        parameter,
        ScalarGeometryParameter::WeirCrestLength
            | ScalarGeometryParameter::OrificeArea
            | ScalarGeometryParameter::GateWidth
            | ScalarGeometryParameter::GateMaximumOpening
    );
    for (index, knot) in knots.iter().enumerate() {
        for (side, value) in [("left", knot.left), ("right", knot.right)] {
            if !value.is_finite() {
                return Err(StructureError::definition(format!(
                    "{ctx},knot={index},{side}={value:?}: supply a finite parameter value"
                )));
            }
            if is_fraction && !(0.0..1.0).contains(&value) {
                return Err(StructureError::definition(format!(
                    "{ctx},knot={index},{side}={value:?}: blocked share must lie in [0, 1). Use a Closed step schedule for full closure"
                )));
            }
            if is_positive && value <= 0.0 {
                return Err(StructureError::definition(format!(
                    "{ctx},knot={index},{side}={value:?}: supply a finite positive dimension"
                )));
            }
        }
    }
    Ok(())
}

fn check_discrete_knots<T: Copy>(
    ctx: &str,
    knots: &[DiscreteGeometryKnot<T>],
) -> Result<(), StructureError> {
    check_times(ctx, knots.iter().map(|k| k.time_s))
}

/// Checks `schedules` against `elements`: unique schedule ids, known structures, law-compatible
/// tracks, no duplicate writer per `(structure_id, track)`, and valid resulting elements at every
/// knot's left and right side.
pub fn check_schedules(
    elements: &[StructureElement],
    schedules: &[StructureGeometrySchedule],
) -> Result<(), StructureError> {
    let mut schedule_ids = BTreeSet::new();
    let mut writers: BTreeSet<(&str, TrackKey)> = BTreeSet::new();
    for schedule in schedules {
        schedule.check()?;
        if !schedule_ids.insert(schedule.schedule_id.as_str()) {
            return Err(StructureError::definition(format!(
                "schedule={:?}: duplicate geometry schedule identity. Supply unique schedule identities",
                schedule.schedule_id
            )));
        }
        let Some(element) = elements.iter().find(|e| e.id == schedule.structure_id) else {
            return Err(StructureError::definition(format!(
                "schedule={:?},structure={:?}: target structure does not exist",
                schedule.schedule_id, schedule.structure_id
            )));
        };
        check_law_match(schedule, &element.law)?;
        if !writers.insert((schedule.structure_id.as_str(), schedule.track.key())) {
            return Err(StructureError::definition(format!(
                "schedule={:?},structure={:?},track={}: duplicate geometry writer for the same parameter. Keep one schedule per parameter",
                schedule.schedule_id,
                schedule.structure_id,
                schedule.track.name()
            )));
        }
    }
    // Check every element at every distinct knot time (both left and right limits).
    let mut times: Vec<f64> = schedules
        .iter()
        .flat_map(|s| s.track.knot_times())
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| a.to_bits() == b.to_bits());
    for element in elements {
        let mine: Vec<&StructureGeometrySchedule> = schedules
            .iter()
            .filter(|s| s.structure_id == element.id)
            .collect();
        if mine.is_empty() {
            continue;
        }
        for &t in &times {
            for after in [false, true] {
                let evaluated = apply_at(element, &mine, t, after)?;
                effective::check(&evaluated)?;
            }
        }
    }
    Ok(())
}

fn check_law_match(
    schedule: &StructureGeometrySchedule,
    law: &StructureLaw,
) -> Result<(), StructureError> {
    let ok = match (&schedule.track, law) {
        (StructureGeometryTrack::Closed { .. }, StructureLaw::Pump(_)) => false,
        (StructureGeometryTrack::Closed { .. }, _) => true,
        (
            StructureGeometryTrack::Scalar {
                parameter:
                    ScalarGeometryParameter::WeirCrestElevation
                    | ScalarGeometryParameter::WeirCrestLength
                    | ScalarGeometryParameter::WeirBlockage,
                ..
            },
            StructureLaw::SuppressedRectangularWeir(_),
        )
        | (
            StructureGeometryTrack::Scalar {
                parameter:
                    ScalarGeometryParameter::OrificeArea
                    | ScalarGeometryParameter::OrificeSillElevation
                    | ScalarGeometryParameter::OrificeCrownElevation
                    | ScalarGeometryParameter::OrificeBlockage,
                ..
            },
            StructureLaw::SubmergedOrifice(_),
        )
        | (
            StructureGeometryTrack::Scalar {
                parameter:
                    ScalarGeometryParameter::GateWidth
                    | ScalarGeometryParameter::GateSillElevation
                    | ScalarGeometryParameter::GateMaximumOpening
                    | ScalarGeometryParameter::GateWidthBlockage
                    | ScalarGeometryParameter::GateOpeningRangeBlockage,
                ..
            },
            StructureLaw::HydrostaticRectangularGate(_),
        )
        | (
            StructureGeometryTrack::Scalar {
                parameter:
                    ScalarGeometryParameter::CulvertFaceABlockage
                    | ScalarGeometryParameter::CulvertFaceBBlockage,
                ..
            }
            | StructureGeometryTrack::CulvertSediment { .. }
            | StructureGeometryTrack::CulvertClosedBarrels { .. },
            StructureLaw::Culvert(_),
        ) => true,
        _ => false,
    };
    if !ok {
        return Err(StructureError::definition(format!(
            "schedule={:?},structure={:?},track={}: track does not apply to this structure law. Supply a track matching the law",
            schedule.schedule_id,
            schedule.structure_id,
            schedule.track.name()
        )));
    }
    Ok(())
}
