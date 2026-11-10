/// Ordered geometric properties at one explicitly requested stage.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSectionProperties {
    /// Stable source section identity.
    pub section_id: String,
    /// Original query operand, not a reported dry water level, in metres.
    pub query_stage_m: f64,
    /// True when the requested stage lies at or below the minimum bed.
    pub dry: bool,
    /// Total wetted cross-sectional area, in square metres.
    pub area_m2: f64,
    /// Wetted horizontal width in metres, absent for dry geometry.
    pub top_width_m: Option<f64>,
    /// Bed and bank contact length, in metres.
    pub wetted_perimeter_m: f64,
    /// Area divided by wetted perimeter, in metres, absent for dry geometry.
    pub hydraulic_radius_m: Option<f64>,
    /// Area divided by top width, in metres, absent for dry geometry.
    pub hydraulic_depth_m: Option<f64>,
    /// Hydrostatic force divided by water density and gravity, in cubic metres.
    pub hydrostatic_integral_m3: f64,
}
