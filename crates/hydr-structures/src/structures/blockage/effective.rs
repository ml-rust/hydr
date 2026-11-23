//! Condition admission and the effective geometry of each law.

use super::{BlockageFraction, BlockageGeometry, CulvertBlockage, StructureCondition};
use crate::structures::definition::text;
use crate::structures::{
    CulvertParameters, CulvertSide, DirectionPolicy, GateParameters, StructureElement,
    StructureError, StructureLaw, SubmergedOrificeParameters, SuppressedRectangularWeirParameters,
};

/// Checks the condition of `element` against its law.
///
/// A pump rejects `Blocked` and `Closed`. A blockage must match the law, and every effective
/// value must be a positive normal F64.
pub(crate) fn check(element: &StructureElement) -> Result<(), StructureError> {
    let id = &element.id;
    let blockage = match (&element.condition, &element.law) {
        (StructureCondition::AsBuilt, _) => return Ok(()),
        (condition, StructureLaw::Pump(_)) => {
            return Err(StructureError::definition(format!(
                "structure={id:?},condition={:?}: a pump curve has no geometry to block. Declare StructureCondition::AsBuilt and stop the pump with StructureOperatingPoint::PumpStoppedIsolated",
                condition.kind()
            )));
        }
        (StructureCondition::Closed { evidence }, _) => {
            return text(evidence, "condition.evidence");
        }
        (StructureCondition::Blocked(blockage), _) => blockage,
    };
    let geometry = blockage.geometry();
    match (geometry, &element.law) {
        (BlockageGeometry::Weir { crest_length }, StructureLaw::SuppressedRectangularWeir(p)) => {
            effective_value(id, "crest_length_m", p.crest_length_m(), *crest_length)
        }
        (BlockageGeometry::Orifice { area }, StructureLaw::SubmergedOrifice(p)) => {
            effective_value(id, "area_m2", p.area_m2(), *area)
        }
        (
            BlockageGeometry::Gate {
                width,
                opening_range,
            },
            StructureLaw::HydrostaticRectangularGate(p),
        ) => {
            effective_value(id, "width_m", p.width_m(), *width)?;
            effective_value(
                id,
                "maximum_opening_m",
                p.maximum_opening_m(),
                *opening_range,
            )
        }
        (BlockageGeometry::Culvert(culvert), StructureLaw::Culvert(p)) => {
            check_culvert(id, element.direction, p, culvert)
        }
        (geometry, law) => {
            let (name, action) = law_action(law);
            Err(StructureError::definition(format!(
                "structure={id:?}: a {} blockage cannot apply to a {name} law. {action}",
                geometry.name()
            )))
        }
    }
}

fn check_culvert(
    id: &str,
    direction: DirectionPolicy,
    parameters: &CulvertParameters,
    blockage: &CulvertBlockage,
) -> Result<(), StructureError> {
    if blockage.closed_barrels >= parameters.barrel_count() {
        return Err(StructureError::definition(format!(
            "structure={id:?},closed_barrels={},barrel_count={}: all barrels are closed. Declare StructureCondition::Closed",
            blockage.closed_barrels,
            parameters.barrel_count()
        )));
    }
    if direction != DirectionPolicy::BtoAOnly && blockage.face_b != BlockageFraction::NONE {
        return Err(StructureError::definition(format!(
            "structure={id:?},direction={direction:?},face_b={}: exit-face blockage is not modelled on an end that flow exits. Set face_b to BlockageFraction::NONE or declare DirectionPolicy::BtoAOnly",
            blockage.face_b.blocked_share()
        )));
    }
    if direction != DirectionPolicy::AtoBOnly && blockage.face_a != BlockageFraction::NONE {
        return Err(StructureError::definition(format!(
            "structure={id:?},direction={direction:?},face_a={}: exit-face blockage is not modelled on an end that flow exits. Set face_a to BlockageFraction::NONE or declare DirectionPolicy::AtoBOnly",
            blockage.face_a.blocked_share()
        )));
    }
    parameters.effective_barrel(Some(blockage))?;
    parameters.bed_elevation(CulvertSide::A, Some(blockage))?;
    parameters.bed_elevation(CulvertSide::B, Some(blockage))?;
    for side in [CulvertSide::A, CulvertSide::B] {
        parameters.effective_rows(side, Some(blockage))?;
    }
    Ok(())
}

/// Law variant name and the corrective action for a mismatched blockage.
fn law_action(law: &StructureLaw) -> (&'static str, &'static str) {
    match law {
        StructureLaw::HydrostaticRectangularGate(_) => (
            "HydrostaticRectangularGate",
            "Supply StructureBlockage::gate",
        ),
        StructureLaw::SubmergedOrifice(_) => {
            ("SubmergedOrifice", "Supply StructureBlockage::orifice")
        }
        StructureLaw::SuppressedRectangularWeir(_) => (
            "SuppressedRectangularWeir",
            "Supply StructureBlockage::weir",
        ),
        StructureLaw::Culvert(_) => ("Culvert", "Supply StructureBlockage::culvert"),
        StructureLaw::Pump(_) => ("Pump", "Declare StructureCondition::AsBuilt"),
    }
}

fn effective_value(
    id: &str,
    field: &str,
    base: f64,
    fraction: BlockageFraction,
) -> Result<(), StructureError> {
    let effective = fraction.apply(base);
    if !effective.is_normal() || effective <= 0.0 {
        return Err(StructureError::definition(format!(
            "structure={id:?},{field}={base},blocked_share={}: the effective value {effective} is not a positive normal F64. Supply a smaller blocked share or a larger {field}",
            fraction.blocked_share()
        )));
    }
    Ok(())
}

fn geometry(condition: &StructureCondition) -> Option<&BlockageGeometry> {
    match condition {
        StructureCondition::Blocked(blockage) => Some(blockage.geometry()),
        StructureCondition::AsBuilt | StructureCondition::Closed { .. } => None,
    }
}

/// Effective crest length `L_e` of an admitted weir.
pub(crate) fn weir_crest_length(
    condition: &StructureCondition,
    parameters: &SuppressedRectangularWeirParameters,
) -> f64 {
    match geometry(condition) {
        Some(BlockageGeometry::Weir { crest_length }) => {
            crest_length.apply(parameters.crest_length_m())
        }
        _ => parameters.crest_length_m(),
    }
}

/// Effective opening area `A_e` of an admitted orifice.
pub(crate) fn orifice_area(
    condition: &StructureCondition,
    parameters: &SubmergedOrificeParameters,
) -> f64 {
    match geometry(condition) {
        Some(BlockageGeometry::Orifice { area }) => area.apply(parameters.area_m2()),
        _ => parameters.area_m2(),
    }
}

/// Effective aperture width `B_e` of an admitted gate.
pub(crate) fn gate_width(condition: &StructureCondition, parameters: &GateParameters) -> f64 {
    match geometry(condition) {
        Some(BlockageGeometry::Gate { width, .. }) => width.apply(parameters.width_m()),
        _ => parameters.width_m(),
    }
}

/// Largest admitted opening `m_e` of an admitted gate.
///
/// A closed gate keeps its base range, so its operations stay admitted.
pub(crate) fn gate_maximum_opening(
    condition: &StructureCondition,
    parameters: &GateParameters,
) -> f64 {
    match geometry(condition) {
        Some(BlockageGeometry::Gate { opening_range, .. }) => {
            opening_range.apply(parameters.maximum_opening_m())
        }
        _ => parameters.maximum_opening_m(),
    }
}

/// Culvert blockage carried by `condition`, or `None` for `AsBuilt`, `Closed`, or other laws.
pub(crate) fn culvert_blockage(condition: &StructureCondition) -> Option<&CulvertBlockage> {
    match geometry(condition) {
        Some(BlockageGeometry::Culvert(culvert)) => Some(culvert),
        _ => None,
    }
}

/// Largest admitted opening of `element`, or `None` when it holds no gate.
pub(crate) fn gate_opening_limit(element: &StructureElement) -> Option<f64> {
    match &element.law {
        StructureLaw::HydrostaticRectangularGate(parameters) => {
            Some(gate_maximum_opening(&element.condition, parameters))
        }
        _ => None,
    }
}
