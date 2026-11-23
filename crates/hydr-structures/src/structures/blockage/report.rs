//! Base and effective geometry of each structure, derived from its definition.

use super::{StructureCondition, StructureConditionKind, effective};
use crate::structures::{StructureElement, StructureLaw};

/// Declared and effective geometry of one structure element.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureGeometryReport {
    /// Ordered element index.
    pub element: usize,
    /// Kind of the element's condition.
    pub condition: StructureConditionKind,
    /// Geometry as declared in the law.
    pub base: StructureGeometry,
    /// Geometry the law evaluates. `None` when the element is closed.
    pub effective: Option<StructureGeometry>,
}

/// Geometry of one law that a blockage can change.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum StructureGeometry {
    /// Suppressed rectangular weir.
    Weir {
        /// Crest length in metres.
        crest_length_m: f64,
    },
    /// Submerged orifice.
    Orifice {
        /// Opening area in m².
        area_m2: f64,
    },
    /// Hydrostatic rectangular gate.
    Gate {
        /// Aperture width in metres.
        width_m: f64,
        /// Largest admitted opening in metres.
        maximum_opening_m: f64,
    },
    /// Culvert.
    #[non_exhaustive]
    Culvert {
        /// Barrel rise in metres.
        rise_m: f64,
        /// Bed elevation at end A on the common datum, in metres.
        bed_a_m: f64,
        /// Bed elevation at end B on the common datum, in metres.
        bed_b_m: f64,
        /// Number of open identical barrels.
        barrel_count: u32,
        /// Open share of the face at ends A and B.
        face_open_share: [f64; 2],
    },
    /// Pump. A pump curve has no geometry.
    Pump,
}

/// Geometry report of every element, in element order.
pub(crate) fn reports(elements: &[StructureElement]) -> Vec<StructureGeometryReport> {
    elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            let condition = element.condition.kind();
            let effective = match condition {
                StructureConditionKind::Closed => None,
                StructureConditionKind::AsBuilt | StructureConditionKind::Blocked => {
                    Some(geometry(element, &element.condition))
                }
            };
            StructureGeometryReport {
                element: index,
                condition,
                base: geometry(element, &StructureCondition::AsBuilt),
                effective,
            }
        })
        .collect()
}

fn geometry(element: &StructureElement, condition: &StructureCondition) -> StructureGeometry {
    match &element.law {
        StructureLaw::SuppressedRectangularWeir(parameters) => StructureGeometry::Weir {
            crest_length_m: effective::weir_crest_length(condition, parameters),
        },
        StructureLaw::SubmergedOrifice(parameters) => StructureGeometry::Orifice {
            area_m2: effective::orifice_area(condition, parameters),
        },
        StructureLaw::HydrostaticRectangularGate(parameters) => StructureGeometry::Gate {
            width_m: effective::gate_width(condition, parameters),
            maximum_opening_m: effective::gate_maximum_opening(condition, parameters),
        },
        StructureLaw::Culvert(parameters) => {
            let alignment = parameters.alignment();
            StructureGeometry::Culvert {
                rise_m: parameters.barrel().rise_m(),
                bed_a_m: alignment.invert_a_m,
                bed_b_m: alignment.invert_b_m,
                barrel_count: parameters.barrel_count(),
                face_open_share: [1.0, 1.0],
            }
        }
        StructureLaw::Pump(_) => StructureGeometry::Pump,
    }
}
