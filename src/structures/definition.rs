//! Concrete law definitions, exact endpoint identities, and declared ideal flaps.

use super::{
    StructureError, StructureInputErrorKind, SubmergedOrificeParameters,
    SuppressedRectangularWeirParameters,
};

/// Only the two explicitly admitted physical closures.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureLaw {
    /// Both supported wet stages must lie strictly above the declared opening crown.
    SubmergedOrifice(SubmergedOrificeParameters),
    /// Suppressed sharp-crested rectangular free/submerged overflow with adopted Villemonte correction.
    SuppressedRectangularWeir(SuppressedRectangularWeirParameters),
}
/// Explicit ideal direction policy, without inferred blockage or controls.
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
    /// Complete concrete physical law.
    pub law: StructureLaw,
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
