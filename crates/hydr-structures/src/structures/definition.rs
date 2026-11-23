//! Concrete law definitions, exact endpoint identities, and declared ideal flaps.

use super::{
    StructureError, StructureInputErrorKind, SubmergedOrificeParameters,
    SuppressedRectangularWeirParameters,
};

/// Explicitly admitted physical closures with separate applicability declarations.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureLaw {
    /// HDS-5 culvert: discharge inverts the larger of the inlet and outlet control headwaters.
    ///
    /// Build a batch that holds it with [`StructureBatch::with_culverts`](super::StructureBatch::with_culverts).
    Culvert(super::CulvertParameters),
    /// Thin hydrostatic rectangular aperture with an explicit instantaneous opening.
    HydrostaticRectangularGate(super::GateParameters),
    /// Explicit oriented reference pump curve with declared speed/applicability policy.
    Pump(super::PumpParameters),
    /// Both supported wet stages must lie strictly above the declared opening crown.
    SubmergedOrifice(SubmergedOrificeParameters),
    /// Suppressed sharp-crested rectangular free/submerged overflow with adopted Villemonte correction.
    SuppressedRectangularWeir(SuppressedRectangularWeirParameters),
}
/// Explicit ideal direction policy. Blockage lives in the element's condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectionPolicy {
    /// Reversed flow requires explicit symmetric coefficient applicability.
    Bidirectional,
    /// Ideal flap allows positive A-to-B flow only.
    AtoBOnly,
    /// Ideal flap allows negative B-to-A flow only.
    BtoAOnly,
}
/// Nominal proposed physical direction before ideal flap blocking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowDirection {
    /// Positive signed discharge.
    AtoB,
    /// Negative signed discharge.
    BtoA,
}
/// Immutable physical destination and coefficient identity.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureElement {
    /// Unique nonempty structure identity.
    pub id: String,
    /// Ordered positive-direction source endpoint identity.
    pub endpoint_a: String,
    /// Ordered positive-direction destination endpoint identity, distinct from A.
    pub endpoint_b: String,
    /// Exact common vertical reference. No transformation occurs.
    pub datum: String,
    /// Explicit ideal flap or symmetric direction policy.
    pub direction: DirectionPolicy,
    /// Complete concrete physical law with its unblocked geometry.
    pub law: StructureLaw,
    /// Physical condition of the declared geometry: as built, blocked, or closed.
    pub condition: super::StructureCondition,
}
pub(crate) fn text(value: &str, field: &str) -> Result<(), StructureError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            format!(
                "{field}={value:?}: supply nonempty explicit identity/evidence without control characters"
            ),
        ));
    }
    Ok(())
}
