//! Admission of each structure law: gravity, control elevations, direction, and operation.
//!
//! Each rejection names its capability entry first.

use super::{SurfaceChannelOperationOwner, SurfaceChannelStructure};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::structures::{DirectionPolicy, StructureLaw, StructureOperatingPoint};

/// Rejects with `entry` for structure `id`.
pub(super) fn rejected(entry: &str, id: &str, detail: String) -> SurfaceChannelError {
    SurfaceChannelError::input(format!("{entry}: structure={id:?} {detail}"))
}

/// Checks the law's gravity, its control elevation against both attachment beds, and its
/// direction policy.
///
/// A culvert's invert at each end sits at or above that end's attachment bed. A dry end then
/// never stands above its invert, so it is a free outfall. A two-way culvert declares both ends.
pub(in super::super) fn check_law(
    structure: &SurfaceChannelStructure,
    beds: [f64; 2],
    gravity: f64,
) -> Result<(), SurfaceChannelError> {
    let element = &structure.element;
    let id = element.id.as_str();
    let (control, symmetric, law_gravity) = match &element.law {
        StructureLaw::Culvert(p) => {
            let alignment = p.alignment();
            for (name, invert, bed) in [
                ("invert_a_m", alignment.invert_a_m, beds[0]),
                ("invert_b_m", alignment.invert_b_m, beds[1]),
            ] {
                if invert < bed {
                    return Err(rejected(
                        "unobstructed-pool-connection",
                        id,
                        format!(
                            "{name}={invert} lies below its attachment bed={bed}: a dry end then stands above the invert; raise the invert to at least its bed"
                        ),
                    ));
                }
            }
            let ends = p.ends();
            (
                None,
                ends.a.is_some() && ends.b.is_some(),
                Some(p.gravity_m_per_s2()),
            )
        }
        StructureLaw::SuppressedRectangularWeir(p) => (
            Some(("crest", p.crest_m())),
            p.applicability().symmetric_reversal,
            None,
        ),
        StructureLaw::HydrostaticRectangularGate(p) => (
            Some(("sill", p.sill_m())),
            p.applicability().symmetric_reversal,
            Some(p.gravity_m_per_s2()),
        ),
        StructureLaw::SubmergedOrifice(p) => (
            Some(("sill", p.sill_m())),
            p.applicability().symmetric_reversal,
            Some(p.gravity_m_per_s2()),
        ),
        StructureLaw::Pump(_) => (None, false, None),
    };
    if let Some(law_gravity) = law_gravity
        && law_gravity.to_bits() != gravity.to_bits()
    {
        return Err(rejected(
            "datum-or-gravity-mismatch",
            id,
            format!(
                "gravity={law_gravity}, surface gravity={gravity}: supply bit-identical gravity"
            ),
        ));
    }
    if let Some((name, elevation)) = control
        && (elevation < beds[0] || elevation < beds[1])
    {
        return Err(rejected(
            "unobstructed-pool-connection",
            id,
            format!(
                "{name}={elevation} lies below attachment bed A={} or B={}: an unobstructed connection is unsupported; raise the {name} to at least both beds",
                beds[0], beds[1]
            ),
        ));
    }
    if element.direction == DirectionPolicy::Bidirectional && !symmetric {
        return Err(rejected(
            "non-symmetric-structure",
            id,
            match &element.law {
                StructureLaw::Pump(_) => {
                    "is a bidirectional pump: a pump moves water one way; set AtoBOnly or BtoAOnly"
                        .into()
                }
                StructureLaw::Culvert(_) => {
                    "is a bidirectional culvert without both ends: declare the B-end inlet for reversed flow or set a one-way direction policy".into()
                }
                _ => "is bidirectional without symmetric reversal evidence: supply a coefficient that holds under reversed flow or set a one-way direction policy".into(),
            },
        ));
    }
    Ok(())
}

/// The operation of `structure`, checked against its law kind.
///
/// A prescribed or actuated owner drives a gate, and a controlled-pump owner drives a pump. A
/// dynamic owner returns the closed state its operating child replaces on every attempt.
pub(super) fn operation(
    structure: &SurfaceChannelStructure,
) -> Result<StructureOperatingPoint, SurfaceChannelError> {
    let law = &structure.element.law;
    let point = match (&structure.operation, law) {
        (SurfaceChannelOperationOwner::Manual(point), _) => point,
        (
            SurfaceChannelOperationOwner::Prescribed | SurfaceChannelOperationOwner::Actuated,
            StructureLaw::HydrostaticRectangularGate(_),
        ) => return Ok(StructureOperatingPoint::GateOpening { opening_m: 0.0 }),
        (SurfaceChannelOperationOwner::ControlledPump, StructureLaw::Pump(_)) => {
            return Ok(StructureOperatingPoint::PumpStoppedIsolated);
        }
        (owner, _) => {
            return Err(rejected(
                "operation-ownership",
                &structure.element.id,
                format!(
                    "owner={owner:?} does not drive its law: give a prescribed or actuated owner to a gate and a controlled-pump owner to a pump"
                ),
            ));
        }
    };
    let admitted = match law {
        StructureLaw::Culvert(_)
        | StructureLaw::SuppressedRectangularWeir(_)
        | StructureLaw::SubmergedOrifice(_) => {
            matches!(point, StructureOperatingPoint::Passive)
        }
        StructureLaw::HydrostaticRectangularGate(_) => {
            matches!(point, StructureOperatingPoint::GateOpening { .. })
        }
        StructureLaw::Pump(_) => matches!(
            point,
            StructureOperatingPoint::PumpRunning { .. }
                | StructureOperatingPoint::PumpStoppedIsolated
        ),
    };
    if !admitted {
        return Err(rejected(
            "operation-ownership",
            &structure.element.id,
            format!(
                "operation={point:?} does not drive its law: give a weir, orifice, or culvert Passive, a gate GateOpening, and a pump PumpRunning or PumpStoppedIsolated"
            ),
        ));
    }
    Ok(*point)
}
