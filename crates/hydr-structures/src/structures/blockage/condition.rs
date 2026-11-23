//! Physical condition of a structure and its typed blockage.

use super::BlockageFraction;
use crate::structures::StructureError;
use crate::structures::definition::text;

/// Physical condition of a structure. The geometry in its law is the unblocked geometry.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureCondition {
    /// The law's geometry as declared.
    AsBuilt,
    /// A typed blockage of the declared geometry.
    Blocked(StructureBlockage),
    /// Fully blocked. Exact zero discharge.
    Closed {
        /// Evidence for the closure, such as a survey or an inspection record.
        evidence: String,
    },
}

/// Kind of a [`StructureCondition`], without its data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureConditionKind {
    /// [`StructureCondition::AsBuilt`].
    AsBuilt,
    /// [`StructureCondition::Blocked`].
    Blocked,
    /// [`StructureCondition::Closed`].
    Closed,
}

impl StructureCondition {
    /// Kind of this condition.
    pub fn kind(&self) -> StructureConditionKind {
        match self {
            Self::AsBuilt => StructureConditionKind::AsBuilt,
            Self::Blocked(_) => StructureConditionKind::Blocked,
            Self::Closed { .. } => StructureConditionKind::Closed,
        }
    }
}

/// Blocked shares of the geometry of one law.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockageGeometry {
    /// Blocked share of a weir crest length.
    Weir {
        /// Blocked share of the crest length.
        crest_length: BlockageFraction,
    },
    /// Blocked share of an orifice opening area.
    Orifice {
        /// Blocked share of the opening area.
        area: BlockageFraction,
    },
    /// Blocked shares of a gate aperture width and opening range.
    Gate {
        /// Blocked share of the aperture width.
        width: BlockageFraction,
        /// Blocked share of the maximum opening. Every opening above the blocked limit rejects.
        opening_range: BlockageFraction,
    },
}

impl BlockageGeometry {
    /// Variant name, for messages.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Weir { .. } => "Weir",
            Self::Orifice { .. } => "Orifice",
            Self::Gate { .. } => "Gate",
        }
    }

    /// Every blocked share, in field order.
    pub(crate) fn fractions(&self) -> Vec<BlockageFraction> {
        match *self {
            Self::Weir { crest_length } => vec![crest_length],
            Self::Orifice { area } => vec![area],
            Self::Gate {
                width,
                opening_range,
            } => vec![width, opening_range],
        }
    }
}

/// A typed blockage of one law's declared geometry, with its applicability evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureBlockage {
    geometry: BlockageGeometry,
    applicability_evidence: String,
}

impl StructureBlockage {
    /// Blocked share of a weir crest length.
    pub fn weir(
        crest_length: BlockageFraction,
        applicability_evidence: String,
    ) -> Result<Self, StructureError> {
        Self::new(
            BlockageGeometry::Weir { crest_length },
            applicability_evidence,
        )
    }

    /// Blocked share of an orifice opening area.
    pub fn orifice(
        area: BlockageFraction,
        applicability_evidence: String,
    ) -> Result<Self, StructureError> {
        Self::new(BlockageGeometry::Orifice { area }, applicability_evidence)
    }

    /// Blocked shares of a gate aperture width and of its maximum opening.
    pub fn gate(
        width: BlockageFraction,
        opening_range: BlockageFraction,
        applicability_evidence: String,
    ) -> Result<Self, StructureError> {
        Self::new(
            BlockageGeometry::Gate {
                width,
                opening_range,
            },
            applicability_evidence,
        )
    }

    fn new(
        geometry: BlockageGeometry,
        applicability_evidence: String,
    ) -> Result<Self, StructureError> {
        text(&applicability_evidence, "blockage.applicability_evidence")?;
        Ok(Self {
            geometry,
            applicability_evidence,
        })
    }

    /// Blocked shares of the law geometry.
    pub fn geometry(&self) -> &BlockageGeometry {
        &self.geometry
    }

    /// Evidence that the law applies to the blocked geometry.
    pub fn applicability_evidence(&self) -> &str {
        &self.applicability_evidence
    }
}
