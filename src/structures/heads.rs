//! Explicit wet, known-dry, and unknown endpoint support.

/// Original endpoint support without a fabricated physical dry water elevation.
#[derive(Clone, Debug, PartialEq)]
pub enum HeadSupport {
    /// Finite quiescent water-surface stage on the common datum.
    Wet {
        /// Water-surface elevation in metres.
        stage_m: f64,
    },
    /// Known dry bed, admitted only for eligible weir connections.
    Dry {
        /// Bed elevation in metres, at or below a weir crest.
        bed_elevation_m: f64,
    },
    /// Unknown influential support always rejects.
    Unknown,
}
/// Explicit head convention. Energy-head packets remain unsupported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadConvention {
    /// Quiescent water-surface stage, following the selected mechanism scope.
    QuiescentStage,
    /// Explicitly unsupported energy head, without silent convention conversion.
    EnergyHead,
}
/// One exact endpoint identity and its original physical support.
#[derive(Clone, Debug, PartialEq)]
pub struct EndpointHead {
    /// Exact declared endpoint identity.
    pub id: String,
    /// Original wet, dry, or unknown support.
    pub support: HeadSupport,
}
/// Complete ordered original input packet for one structure.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureHeads {
    /// Exact declared structure identity.
    pub structure_id: String,
    /// Original endpoint A identity and support.
    pub endpoint_a: EndpointHead,
    /// Original endpoint B identity and support.
    pub endpoint_b: EndpointHead,
    /// Exact declared common vertical reference.
    pub datum: String,
    /// Explicit input head convention.
    pub convention: HeadConvention,
}
