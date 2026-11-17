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

/// Prescribed boundary elevation with a compatible wet-subcritical trace.
#[derive(Clone, Debug, PartialEq)]
pub struct StageBoundary {
    /// Exterior face receiving this condition.
    pub face: BoundaryFace,
    /// Water elevation in metres, using the domain vertical datum.
    pub elevation: f64,
    /// Incoming velocity along the positive other coordinate, in m/s.
    ///
    /// X faces use positive y. Y faces use positive x. Reversed inflow requires this value.
    pub inflow_tangential_velocity: Option<f64>,
}

/// Prescribed outward numerical water rate with a compatible wet-subcritical trace.
#[derive(Clone, Debug, PartialEq)]
pub struct DischargeBoundary {
    /// Exterior face receiving this condition.
    pub face: BoundaryFace,
    /// Signed outward water rate in m³/s. Negative enters the domain.
    pub outward_rate: f64,
    /// Incoming velocity along the positive other coordinate, in m/s.
    pub inflow_tangential_velocity: Option<f64>,
}

/// Complete externally maintained reservoir state on a boundary's adjacent bed.
#[derive(Clone, Debug, PartialEq)]
pub enum ExteriorState {
    /// Exactly dry exterior, without a fabricated elevation or velocity.
    Dry,
    /// Wet exterior with complete normal and tangential velocity data.
    Wet {
        /// Water elevation in metres on the domain vertical datum.
        elevation: f64,
        /// Outward-normal velocity in m/s. Negative points toward the domain.
        outward_velocity: f64,
        /// Velocity along the positive other global coordinate in m/s.
        tangential_velocity: f64,
    },
}

/// Complete Riemann data supplied by an externally maintained reservoir.
///
/// This represents no finite exterior donor volume and prescribes no achieved face discharge.
#[derive(Clone, Debug, PartialEq)]
pub struct ExteriorBoundary {
    /// Active exterior face receiving this reservoir state.
    pub face: BoundaryFace,
    /// Complete wet or explicitly dry exterior state.
    pub state: ExteriorState,
}

impl ExteriorBoundary {
    #[doc(hidden)]
    pub fn conserved(&self, domain: &Domain) -> Result<[f64; 3], EngineError> {
        let cell = exterior_cell(domain, self.face)?;
        let ExteriorState::Wet {
            elevation,
            outward_velocity,
            tangential_velocity,
        } = self.state
        else {
            return Ok([0.0; 3]);
        };
        let h = elevation - domain.bed()[cell];
        let normal = h * outward_velocity;
        let tangent = h * tangential_velocity;
        if ![
            elevation,
            outward_velocity,
            tangential_velocity,
            h,
            normal,
            tangent,
        ]
        .iter()
        .all(|value| value.is_finite())
            || h <= 0.0
            || (outward_velocity != 0.0 && normal == 0.0)
            || (tangential_velocity != 0.0 && tangent == 0.0)
        {
            return Err(EngineError::rejected(
                "exterior boundary",
                format!(
                    "state={self:?}, bed={}: require finite positive wet depth and representable velocity products",
                    domain.bed()[cell]
                ),
            ));
        }
        Ok([h, normal, tangent])
    }
}

/// Physical data on one declared open face during a forcing interval.
#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryForcing {
    /// Source-complete full flux with provider-attested signal bounds.
    Flux(FluxPort),
    /// Elevation determines a compatible wet-subcritical boundary trace.
    Stage(StageBoundary),
    /// Water rate determines a compatible wet-subcritical boundary trace.
    Discharge(DischargeBoundary),
    /// Complete exterior state enters the same numerical Riemann solver as interior faces.
    Exterior(ExteriorBoundary),
    /// Copies a dry or strictly supercritical-outgoing interior state outside.
    Transmissive(BoundaryFace),
}

impl BoundaryForcing {
    /// Returns the exterior face receiving this condition.
    pub fn face(&self) -> BoundaryFace {
        match self {
            Self::Flux(value) => value.face,
            Self::Stage(value) => value.face,
            Self::Discharge(value) => value.face,
            Self::Exterior(value) => value.face,
            Self::Transmissive(face) => *face,
        }
    }
}

impl From<FluxPort> for BoundaryForcing {
    fn from(value: FluxPort) -> Self {
        Self::Flux(value)
    }
}

/// Immutable assignment of genuine walls and declared exterior open faces.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryConditions {
    open_faces: Vec<BoundaryFace>,
}

impl BoundaryConditions {
    /// Declares every exterior face and inactive-cell interface a genuine impermeable wall.
    ///
    /// Never use this constructor to replace an unknown outlet or missing terrain.
    pub fn impermeable_walls() -> Self {
        Self {
            open_faces: Vec::new(),
        }
    }

    /// Declares named exterior open faces. All remaining boundaries are genuine impermeable walls.
    ///
    /// # Errors
    /// Rejects duplicate faces, interior faces, and faces adjoining inactive cells.
    pub fn with_open_faces(
        domain: &Domain,
        open_faces: Vec<BoundaryFace>,
    ) -> Result<Self, EngineError> {
        let mut seen = BTreeSet::new();
        for face in &open_faces {
            exterior_cell(domain, *face)?;
            if !seen.insert(*face) {
                return Err(EngineError::rejected(
                    "boundary",
                    format!("duplicate face {face:?}: supply each exterior face once"),
                ));
            }
        }
        Ok(Self { open_faces })
    }

    /// Returns the required open faces. Every forcing interval must supply each face once.
    pub fn open_faces(&self) -> &[BoundaryFace] {
        &self.open_faces
    }
}

#[inline]
pub fn exterior_cell(domain: &Domain, face: BoundaryFace) -> Result<usize, EngineError> {
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
