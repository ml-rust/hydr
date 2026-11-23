//! Base and effective geometry of each structure, derived from its definition.

use super::{StructureCondition, StructureConditionKind, effective};
use crate::structures::{CulvertBarrelShape, CulvertSide, StructureElement, StructureLaw};

/// Precomputed full-area and floor-width enclosures of one culvert element.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CulvertGeometrySummary {
    /// Element index in the batch.
    pub(crate) element: usize,
    /// Unblocked barrel full area enclosure, in m².
    pub(crate) base_full_area_m2: (f64, f64),
    /// Open-section full area enclosure, in m².
    pub(crate) effective_full_area_m2: (f64, f64),
    /// Unblocked floor width enclosure, in metres.
    pub(crate) base_floor_width_m: (f64, f64),
    /// Effective bed/floor width enclosure, in metres.
    pub(crate) effective_floor_width_m: (f64, f64),
}

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
        /// Open barrel rise in metres.
        rise_m: f64,
        /// Enclosure of the full open-section area in m².
        full_area_m2: (f64, f64),
        /// Bed elevation at end A on the common datum, in metres.
        bed_a_m: f64,
        /// Bed elevation at end B on the common datum, in metres.
        bed_b_m: f64,
        /// Number of open identical barrels.
        barrel_count: u32,
        /// Open share of the face at ends A and B.
        face_open_share: [f64; 2],
        /// Enclosure of the wetted floor or sediment bed width in metres.
        floor_width_m: (f64, f64),
        /// Bed-to-wall Manning roughness ratio `n_bed / n_wall`, if a floor or bed is declared.
        roughness_ratio: Option<f64>,
    },
    /// Pump. A pump curve has no geometry.
    Pump,
}

/// Geometry report of every element, in element order.
pub(crate) fn reports(
    elements: &[StructureElement],
    culverts: &[CulvertGeometrySummary],
) -> Vec<StructureGeometryReport> {
    elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            let summary = culverts.iter().find(|entry| entry.element == index);
            let condition = element.condition.kind();
            let effective = match condition {
                StructureConditionKind::Closed => None,
                StructureConditionKind::AsBuilt | StructureConditionKind::Blocked => {
                    Some(geometry(element, &element.condition, summary, false))
                }
            };
            StructureGeometryReport {
                element: index,
                condition,
                base: geometry(element, &StructureCondition::AsBuilt, summary, true),
                effective,
            }
        })
        .collect()
}

fn geometry(
    element: &StructureElement,
    condition: &StructureCondition,
    summary: Option<&CulvertGeometrySummary>,
    base: bool,
) -> StructureGeometry {
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
            let blockage = effective::culvert_blockage(condition);
            let barrel = parameters
                .effective_barrel(blockage)
                .unwrap_or_else(|_| parameters.barrel());
            let bed_a_m = parameters
                .bed_elevation(CulvertSide::A, blockage)
                .unwrap_or_else(|_| parameters.invert(CulvertSide::A));
            let bed_b_m = parameters
                .bed_elevation(CulvertSide::B, blockage)
                .unwrap_or_else(|_| parameters.invert(CulvertSide::B));
            let closed_barrels = blockage.map_or(0, |b| b.closed_barrels);
            let face_open_share = blockage.map_or([1.0, 1.0], |b| {
                [b.face_a.open_share(), b.face_b.open_share()]
            });
            let default_floor = if matches!(
                parameters.barrel().shape(),
                CulvertBarrelShape::Box | CulvertBarrelShape::Arch
            ) {
                let span = parameters.barrel().span_m();
                (span, span)
            } else {
                (0.0, 0.0)
            };
            let (full_area_m2, floor_width_m) = match (summary, base) {
                (Some(s), true) => (s.base_full_area_m2, s.base_floor_width_m),
                (Some(s), false) => (s.effective_full_area_m2, s.effective_floor_width_m),
                (None, _) => ((0.0, 0.0), default_floor),
            };
            StructureGeometry::Culvert {
                rise_m: barrel.open_rise_m(),
                full_area_m2,
                bed_a_m,
                bed_b_m,
                barrel_count: parameters.barrel_count().saturating_sub(closed_barrels),
                face_open_share,
                floor_width_m,
                roughness_ratio: barrel.floor().map(|f| f.roughness_ratio),
            }
        }
        StructureLaw::Pump(_) => StructureGeometry::Pump,
    }
}
