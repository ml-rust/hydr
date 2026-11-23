//! Exclusive parameter authority table over structure operations, static conditions, and
//! prescribed geometry schedules, plus runtime gate-range re-checks against actuated gates.
//!
//! Built once at definition construction from declared owners and schedules. Any parameter with
//! two writers rejects with `duplicate-parameter-authority`. Native SWMM device authority is not
//! attached here; it joins this table in the drainage adapter unit.

use super::{SurfaceChannelError, SurfaceChannelOperationOwner, SurfaceChannelSetup};
use crate::structures::{
    BlockageGeometry, ControlledGateBatch, ControlledGatePreparedInterval, ScalarGeometryParameter,
    StructureCondition, StructureElement, StructureGeometrySchedule, StructureGeometryTrack,
    StructureLaw, StructureOperatingPoint, evaluate_element_interval,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Element owned by an entry in the authority table.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum AuthorityTarget {
    Structure(String),
    Pool(String),
    SurfaceCell(usize),
    ChannelReach(String),
}

impl fmt::Display for AuthorityTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structure(id) => write!(f, "structure={id:?}"),
            Self::Pool(id) => write!(f, "pool={id:?}"),
            Self::SurfaceCell(cell) => write!(f, "surface_cell={cell}"),
            Self::ChannelReach(id) => write!(f, "reach={id:?}"),
        }
    }
}

/// Written parameter of a target in the authority table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum AuthorityParameter {
    GateOpening,
    PumpOperation,
    PassiveOperation,
    WeirCrestElevation,
    WeirCrestLength,
    WeirBlockage,
    OrificeArea,
    OrificeSillElevation,
    OrificeCrownElevation,
    OrificeBlockage,
    GateWidth,
    GateSillElevation,
    GateMaximumOpening,
    GateWidthBlockage,
    GateOpeningRangeBlockage,
    CulvertFaceABlockage,
    CulvertFaceBBlockage,
    CulvertSediment,
    CulvertClosedBarrels,
    StructureClosed,
    PoolCurve,
    SurfaceBed,
    ChannelSection,
}

impl AuthorityParameter {
    fn of_track(track: &StructureGeometryTrack) -> Result<Self, SurfaceChannelError> {
        let param = match track {
            StructureGeometryTrack::Scalar { parameter, .. } => match parameter {
                ScalarGeometryParameter::WeirCrestElevation => Self::WeirCrestElevation,
                ScalarGeometryParameter::WeirCrestLength => Self::WeirCrestLength,
                ScalarGeometryParameter::WeirBlockage => Self::WeirBlockage,
                ScalarGeometryParameter::OrificeArea => Self::OrificeArea,
                ScalarGeometryParameter::OrificeSillElevation => Self::OrificeSillElevation,
                ScalarGeometryParameter::OrificeCrownElevation => Self::OrificeCrownElevation,
                ScalarGeometryParameter::OrificeBlockage => Self::OrificeBlockage,
                ScalarGeometryParameter::GateWidth => Self::GateWidth,
                ScalarGeometryParameter::GateSillElevation => Self::GateSillElevation,
                ScalarGeometryParameter::GateMaximumOpening => Self::GateMaximumOpening,
                ScalarGeometryParameter::GateWidthBlockage => Self::GateWidthBlockage,
                ScalarGeometryParameter::GateOpeningRangeBlockage => Self::GateOpeningRangeBlockage,
                ScalarGeometryParameter::CulvertFaceABlockage => Self::CulvertFaceABlockage,
                ScalarGeometryParameter::CulvertFaceBBlockage => Self::CulvertFaceBBlockage,
                other => {
                    return Err(SurfaceChannelError::input(format!(
                        "unsupported scalar geometry parameter {other:?}"
                    )));
                }
            },
            StructureGeometryTrack::CulvertSediment { .. } => Self::CulvertSediment,
            StructureGeometryTrack::CulvertClosedBarrels { .. } => Self::CulvertClosedBarrels,
            StructureGeometryTrack::Closed { .. } => Self::StructureClosed,
            other => {
                return Err(SurfaceChannelError::input(format!(
                    "unsupported geometry track {other:?}"
                )));
            }
        };
        Ok(param)
    }

    fn name(self) -> &'static str {
        match self {
            Self::GateOpening => "gate_opening",
            Self::PumpOperation => "pump_operation",
            Self::PassiveOperation => "passive_operation",
            Self::WeirCrestElevation => "weir_crest_elevation",
            Self::WeirCrestLength => "weir_crest_length",
            Self::WeirBlockage => "weir_blockage",
            Self::OrificeArea => "orifice_area",
            Self::OrificeSillElevation => "orifice_sill_elevation",
            Self::OrificeCrownElevation => "orifice_crown_elevation",
            Self::OrificeBlockage => "orifice_blockage",
            Self::GateWidth => "gate_width",
            Self::GateSillElevation => "gate_sill_elevation",
            Self::GateMaximumOpening => "gate_maximum_opening",
            Self::GateWidthBlockage => "gate_width_blockage",
            Self::GateOpeningRangeBlockage => "gate_opening_range_blockage",
            Self::CulvertFaceABlockage => "culvert_face_a_blockage",
            Self::CulvertFaceBBlockage => "culvert_face_b_blockage",
            Self::CulvertSediment => "culvert_sediment",
            Self::CulvertClosedBarrels => "culvert_closed_barrels",
            Self::StructureClosed => "closed",
            Self::PoolCurve => "pool_curve",
            Self::SurfaceBed => "surface_bed",
            Self::ChannelSection => "channel_section",
        }
    }
}

/// Exclusive writer recorded in the authority table.
#[derive(Clone, Debug, PartialEq, Eq)]
enum AuthorityWriter {
    ManualOperation,
    PrescribedTrajectory,
    ControlledGateActuator,
    ControlledPump,
    StaticCondition(&'static str),
    GeometrySchedule(String),
    PoolCurveSchedule(String),
    SurfaceBedSchedule(String),
    ChannelSectionSchedule(String),
}

impl fmt::Display for AuthorityWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ManualOperation => write!(f, "manual operation"),
            Self::PrescribedTrajectory => write!(f, "prescribed trajectory"),
            Self::ControlledGateActuator => write!(f, "controlled gate actuator"),
            Self::ControlledPump => write!(f, "controlled pump"),
            Self::StaticCondition(kind) => write!(f, "static element condition ({kind})"),
            Self::GeometrySchedule(id) => write!(f, "geometry_schedule {id:?}"),
            Self::PoolCurveSchedule(id) => write!(f, "pool_curve_schedule {id:?}"),
            Self::SurfaceBedSchedule(id) => write!(f, "surface_bed_schedule {id:?}"),
            Self::ChannelSectionSchedule(id) => write!(f, "channel_section_schedule {id:?}"),
        }
    }
}

/// Checked exclusive-writer table across structure operations, static blockage/closure conditions,
/// and all four geometry schedule families.
#[derive(Clone, Debug)]
pub(super) struct AuthorityTable {
    entries: BTreeMap<(AuthorityTarget, AuthorityParameter), AuthorityWriter>,
}

impl AuthorityTable {
    /// Builds the authority table for `setup`, rejecting any parameter with two writers and any
    /// manual gate opening that exceeds a scheduled gate opening range.
    pub(super) fn build(setup: &SurfaceChannelSetup) -> Result<Self, SurfaceChannelError> {
        let mut table = Self {
            entries: BTreeMap::new(),
        };
        for structure in &setup.structures {
            let target = AuthorityTarget::Structure(structure.element.id.clone());
            let (param, writer) = match (&structure.element.law, &structure.operation) {
                (
                    StructureLaw::HydrostaticRectangularGate(_),
                    SurfaceChannelOperationOwner::Manual(_),
                ) => (
                    AuthorityParameter::GateOpening,
                    AuthorityWriter::ManualOperation,
                ),
                (
                    StructureLaw::HydrostaticRectangularGate(_),
                    SurfaceChannelOperationOwner::Prescribed,
                ) => (
                    AuthorityParameter::GateOpening,
                    AuthorityWriter::PrescribedTrajectory,
                ),
                (
                    StructureLaw::HydrostaticRectangularGate(_),
                    SurfaceChannelOperationOwner::Actuated,
                ) => (
                    AuthorityParameter::GateOpening,
                    AuthorityWriter::ControlledGateActuator,
                ),
                (StructureLaw::Pump(_), SurfaceChannelOperationOwner::Manual(_)) => (
                    AuthorityParameter::PumpOperation,
                    AuthorityWriter::ManualOperation,
                ),
                (StructureLaw::Pump(_), SurfaceChannelOperationOwner::ControlledPump) => (
                    AuthorityParameter::PumpOperation,
                    AuthorityWriter::ControlledPump,
                ),
                (_, SurfaceChannelOperationOwner::Manual(_)) => (
                    AuthorityParameter::PassiveOperation,
                    AuthorityWriter::ManualOperation,
                ),
                // Mismatched law/owner pairs are already rejected by check_structures.
                _ => continue,
            };
            table.register(target.clone(), param, writer)?;
            table.register_static_condition(&target, &structure.element.condition)?;
        }
        for schedule in &setup.geometry_schedules {
            let target = AuthorityTarget::Structure(schedule.structure_id.clone());
            let param = AuthorityParameter::of_track(&schedule.track)?;
            table.register(
                target,
                param,
                AuthorityWriter::GeometrySchedule(schedule.schedule_id.clone()),
            )?;
        }
        for schedule in &setup.pool_curve_schedules {
            table.register(
                AuthorityTarget::Pool(schedule.pool_id.clone()),
                AuthorityParameter::PoolCurve,
                AuthorityWriter::PoolCurveSchedule(schedule.schedule_id.clone()),
            )?;
        }
        for schedule in &setup.surface_bed_schedules {
            let cells: BTreeSet<usize> = schedule
                .epochs
                .iter()
                .flat_map(|e| e.updates.iter().map(|u| u.cell))
                .collect();
            for cell in cells {
                table.register(
                    AuthorityTarget::SurfaceCell(cell),
                    AuthorityParameter::SurfaceBed,
                    AuthorityWriter::SurfaceBedSchedule(schedule.schedule_id.clone()),
                )?;
            }
        }
        for schedule in &setup.channel_section_schedules {
            table.register(
                AuthorityTarget::ChannelReach(schedule.reach_id.clone()),
                AuthorityParameter::ChannelSection,
                AuthorityWriter::ChannelSectionSchedule(schedule.schedule_id.clone()),
            )?;
        }
        check_manual_gate_ranges(setup)?;
        Ok(table)
    }

    fn register(
        &mut self,
        target: AuthorityTarget,
        parameter: AuthorityParameter,
        writer: AuthorityWriter,
    ) -> Result<(), SurfaceChannelError> {
        if let Some(existing) = self.entries.get(&(target.clone(), parameter)) {
            return Err(SurfaceChannelError::input(format!(
                "duplicate-parameter-authority: {target} parameter={} already owned by {existing}; cannot also be written by {writer}",
                parameter.name()
            )));
        }
        self.entries.insert((target, parameter), writer);
        Ok(())
    }

    fn register_static_condition(
        &mut self,
        target: &AuthorityTarget,
        condition: &StructureCondition,
    ) -> Result<(), SurfaceChannelError> {
        match condition {
            StructureCondition::AsBuilt => Ok(()),
            StructureCondition::Closed { .. } => self.register(
                target.clone(),
                AuthorityParameter::StructureClosed,
                AuthorityWriter::StaticCondition("closed"),
            ),
            StructureCondition::Blocked(blockage) => match blockage.geometry() {
                BlockageGeometry::Weir { crest_length } => {
                    if crest_length.blocked_share() > 0.0 {
                        self.register(
                            target.clone(),
                            AuthorityParameter::WeirBlockage,
                            AuthorityWriter::StaticCondition("blocked"),
                        )?;
                    }
                    Ok(())
                }
                BlockageGeometry::Orifice { area } => {
                    if area.blocked_share() > 0.0 {
                        self.register(
                            target.clone(),
                            AuthorityParameter::OrificeBlockage,
                            AuthorityWriter::StaticCondition("blocked"),
                        )?;
                    }
                    Ok(())
                }
                BlockageGeometry::Gate {
                    width,
                    opening_range,
                } => {
                    if width.blocked_share() > 0.0 {
                        self.register(
                            target.clone(),
                            AuthorityParameter::GateWidthBlockage,
                            AuthorityWriter::StaticCondition("blocked"),
                        )?;
                    }
                    if opening_range.blocked_share() > 0.0 {
                        self.register(
                            target.clone(),
                            AuthorityParameter::GateOpeningRangeBlockage,
                            AuthorityWriter::StaticCondition("blocked"),
                        )?;
                    }
                    Ok(())
                }
                BlockageGeometry::Culvert(cb) => {
                    if cb.face_a.blocked_share() > 0.0 {
                        self.register(
                            target.clone(),
                            AuthorityParameter::CulvertFaceABlockage,
                            AuthorityWriter::StaticCondition("blocked"),
                        )?;
                    }
                    if cb.face_b.blocked_share() > 0.0 {
                        self.register(
                            target.clone(),
                            AuthorityParameter::CulvertFaceBBlockage,
                            AuthorityWriter::StaticCondition("blocked"),
                        )?;
                    }
                    if cb.sediment.is_some() {
                        self.register(
                            target.clone(),
                            AuthorityParameter::CulvertSediment,
                            AuthorityWriter::StaticCondition("blocked"),
                        )?;
                    }
                    if cb.closed_barrels > 0 {
                        self.register(
                            target.clone(),
                            AuthorityParameter::CulvertClosedBarrels,
                            AuthorityWriter::StaticCondition("blocked"),
                        )?;
                    }
                    Ok(())
                }
                _ => Ok(()),
            },
        }
    }
}

/// Effective maximum gate opening of `element` (closed gates keep their base opening range).
pub(super) fn effective_gate_max_opening(element: &StructureElement) -> Option<f64> {
    let StructureLaw::HydrostaticRectangularGate(gate) = &element.law else {
        return None;
    };
    let base = gate.maximum_opening_m();
    match &element.condition {
        StructureCondition::AsBuilt | StructureCondition::Closed { .. } => Some(base),
        StructureCondition::Blocked(blockage) => match blockage.geometry() {
            BlockageGeometry::Gate { opening_range, .. } => Some(base * opening_range.open_share()),
            _ => Some(base),
        },
    }
}

/// Rejects any manual gate opening that exceeds a scheduled gate opening range at any knot side.
fn check_manual_gate_ranges(setup: &SurfaceChannelSetup) -> Result<(), SurfaceChannelError> {
    if setup.geometry_schedules.is_empty() {
        return Ok(());
    }
    for structure in &setup.structures {
        let SurfaceChannelOperationOwner::Manual(StructureOperatingPoint::GateOpening {
            opening_m,
        }) = structure.operation
        else {
            continue;
        };
        let mine: Vec<&StructureGeometrySchedule> = setup
            .geometry_schedules
            .iter()
            .filter(|s| s.structure_id == structure.element.id)
            .collect();
        if mine.is_empty() {
            continue;
        }
        let mut times: Vec<f64> = mine.iter().flat_map(|s| s.track.knot_times()).collect();
        times.sort_by(f64::total_cmp);
        times.dedup_by(|a, b| a.to_bits() == b.to_bits());
        for t in times {
            for dt in [-1e-9, 1e-9] {
                let (start, end) = if dt < 0.0 { (t + dt, t) } else { (t, t + dt) };
                let evaluated = evaluate_element_interval(&structure.element, &mine, start, end)
                    .map_err(|error| {
                        SurfaceChannelError::input(format!("geometry_schedules: {error}"))
                    })?;
                if let Some(max_open) = effective_gate_max_opening(&evaluated)
                    && opening_m > max_open
                {
                    return Err(SurfaceChannelError::input(format!(
                        "gate-range-conflict: structure={:?} manual opening_m={opening_m} exceeds scheduled effective maximum opening={max_open} at time_s={t}: reduce the manual opening or widen the scheduled range",
                        structure.element.id
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Re-checks every actuated gate's position and commanded target against the scheduled effective
/// maximum opening in `elements`. Rejects without retry on conflict.
pub(super) fn check_actuated_gate_ranges<R: Runtime<DType = DType>>(
    gate: &ControlledGateBatch<R>,
    prepared: Option<&ControlledGatePreparedInterval<R>>,
    elements: &[StructureElement],
    time_label: &str,
) -> Result<(), SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let defs = gate.actuators().definitions();
    let start_outputs = gate
        .actuators()
        .outputs()
        .map_err(|e| SurfaceChannelError::input(format!("operating child=controlled-gate: {e}")))?;
    let end_outputs = match prepared {
        Some(p) => Some(p.actuators().left_outputs().map_err(|e| {
            SurfaceChannelError::input(format!("operating child=controlled-gate: {e}"))
        })?),
        None => None,
    };
    for (index, (def, start_out)) in defs.iter().zip(&start_outputs).enumerate() {
        let Some(element) = elements.iter().find(|e| e.id == def.structure_id) else {
            continue;
        };
        let Some(max_open) = effective_gate_max_opening(element) else {
            continue;
        };
        let check_val = |kind: &str, val: f64| -> Result<(), SurfaceChannelError> {
            if val > max_open {
                return Err(SurfaceChannelError::input(format!(
                    "gate-range-conflict: structure={:?} at {time_label}: actuator {kind}={val} exceeds scheduled effective maximum opening={max_open}: keep actuator position and target within the scheduled gate range",
                    def.structure_id
                )));
            }
            Ok(())
        };
        check_val("position", start_out.position.opening_m)?;
        check_val("target", start_out.target_opening_m)?;
        if let Some(end_outs) = &end_outputs
            && let Some(end_out) = end_outs.get(index)
        {
            check_val("end_position", end_out.position.opening_m)?;
            check_val("end_target", end_out.target_opening_m)?;
        }
    }
    Ok(())
}
