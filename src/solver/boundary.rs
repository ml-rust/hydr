use super::EngineError;
use crate::model::Domain;
use std::collections::BTreeSet;

/// Direction of a Cartesian face's positive coordinate normal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Axis {
    /// Positive x direction.
    X,
    /// Positive y direction.
    Y,
}

/// Face coordinates in the Cartesian grid, using zero-based columns and rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BoundaryFace {
    /// Face normal direction.
    pub axis: Axis,
    /// X faces admit columns 0..=nx. Y faces admit columns 0..nx.
    pub column: usize,
    /// X faces admit rows 0..ny. Y faces admit rows 0..=ny.
    pub row: usize,
}

/// Declared compatible external full flux, constant over its forcing interval.
///
/// The provider attests source completeness and conservative wave bounds.
/// This contract implements no stage/discharge characteristic closure.
#[derive(Clone, Debug, PartialEq)]
pub struct FluxPort {
    /// Exterior face receiving this flux.
    pub face: BoundaryFace,
    /// Signed outward water rate in m³/s. Negative enters the domain.
    pub water_rate: f64,
    /// Outward x/y momentum flux integrated over face width, in m⁴/s².
    ///
    /// Include normal pressure and all bed/interface corrections exactly once.
    pub momentum_rate: [f64; 2],
    /// Conservative absolute normal signal-speed bound in m/s.
    pub signal_speed: f64,
    /// Provider identity, compatible state/regime, and basis for the speed bound.
    pub provenance: String,
}

/// Immutable assignment of genuine walls and source-complete exterior ports.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryConditions {
    ports: Vec<BoundaryFace>,
}

impl BoundaryConditions {
    /// Declares every exterior face and inactive-cell interface a genuine impermeable wall.
    ///
    /// Never use this constructor to replace an unknown outlet or missing terrain.
    pub fn impermeable_walls() -> Self {
        Self { ports: Vec::new() }
    }

    /// Declares named exterior ports. All remaining boundaries are genuine impermeable walls.
    ///
    /// # Errors
    /// Rejects duplicate faces, interior faces, and faces adjoining inactive cells.
    pub fn with_ports(domain: &Domain, ports: Vec<BoundaryFace>) -> Result<Self, EngineError> {
        let mut seen = BTreeSet::new();
        for face in &ports {
            exterior_cell(domain, *face)?;
            if !seen.insert(*face) {
                return Err(EngineError::rejected(
                    "boundary",
                    format!("duplicate face {face:?}: supply each exterior face once"),
                ));
            }
        }
        Ok(Self { ports })
    }

    /// Returns the required port faces. Every forcing interval must supply each face once.
    pub fn ports(&self) -> &[BoundaryFace] {
        &self.ports
    }
}

pub(super) fn exterior_cell(domain: &Domain, face: BoundaryFace) -> Result<usize, EngineError> {
    let (nx, ny) = domain.grid().dimensions();
    let cell = match face.axis {
        Axis::X if face.row < ny && (face.column == 0 || face.column == nx) => {
            Some(face.row * nx + if face.column == 0 { 0 } else { nx - 1 })
        }
        Axis::Y if face.column < nx && (face.row == 0 || face.row == ny) => {
            Some(if face.row == 0 {
                face.column
            } else {
                (ny - 1) * nx + face.column
            })
        }
        _ => None,
    };
    match cell {
        Some(cell) if domain.active()[cell] => Ok(cell),
        _ => Err(EngineError::rejected(
            "boundary",
            format!("face {face:?} is not an active exterior face: correct its support"),
        )),
    }
}
