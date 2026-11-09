use super::BoundaryFace;

/// Cell-supported physical output at the solver's accepted time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellResult {
    /// Outside the grid or inside an excluded impermeable cell. No dry inference applies.
    Unsupported,
    /// Exactly zero represented depth. Elevation and velocity are undefined.
    Dry,
    /// Positive represented water, including positive films.
    Wet {
        /// Depth above the represented bed in metres.
        depth: f64,
        /// Water elevation in metres on the domain's vertical datum.
        elevation: f64,
        /// Depth-averaged x/y velocity in m/s.
        velocity: [f64; 2],
        /// X/y discharge per width in m²/s.
        discharge_per_width: [f64; 2],
    },
}

/// Shared numerical face flux over one accepted interval, without interpolation.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceTransfer {
    /// Grid support of this face. Interior faces use the same coordinate convention.
    pub face: BoundaryFace,
    /// Active cell on the negative-coordinate side, if present.
    pub left: Option<usize>,
    /// Active cell on the positive-coordinate side, if present.
    pub right: Option<usize>,
    /// Positive-axis mass, x momentum, and y momentum flux per width.
    ///
    /// Units are m²/s, m³/s², and m³/s². Interior bed corrections are reported separately.
    pub flux: [f64; 3],
    /// Side-specific normal pressure corrections in m³/s².
    pub pressure_corrections: [f64; 2],
    /// Final conservative absolute normal signal speed in m/s. Zero admits an exactly dry face.
    pub signal_speed: f64,
    /// Signed positive-axis water volume crossing this face in m³.
    pub volume: f64,
}
