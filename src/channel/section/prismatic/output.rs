/// Host geometry bounds for one original depth operand.
#[derive(Clone, Debug, PartialEq)]
pub struct PrismaticSectionProperties {
    /// Stable identity of the queried source section.
    pub section_id: String,
    /// Original requested depth above the bed, in metres.
    pub query_depth_m: f64,
    /// True when the original requested depth is zero.
    pub dry: bool,
    /// Geometric depth bounds `[lower, upper]`, in metres.
    pub depth_m: [f64; 2],
    /// Wetted area bounds `[lower, upper]`, in square metres.
    pub area_m2: [f64; 2],
    /// Wet horizontal width bounds `[lower, upper]`, in metres. Absent when dry.
    pub top_width_m: Option<[f64; 2]>,
    /// Hydrostatic integral bounds `[lower, upper]`, in cubic metres.
    pub hydrostatic_integral_m3: [f64; 2],
    /// Area divided by width bounds `[lower, upper]`, in metres. Absent when dry.
    pub hydraulic_depth_m: Option<[f64; 2]>,
}

/// Host geometry bounds conditioned on one original conserved-area operand.
#[derive(Clone, Debug, PartialEq)]
pub struct PrismaticRecoveredProperties {
    /// Stable identity of the queried source section.
    pub section_id: String,
    /// Original requested conserved area, in square metres.
    pub requested_area_m2: f64,
    /// True when the original requested area is zero.
    pub dry: bool,
    /// Recovered geometric depth bounds `[lower, upper]`, in metres.
    pub depth_m: [f64; 2],
    /// Forward area bounds across recovered depths, in square metres.
    ///
    /// The array contains `[lower, upper]`. The original requested area remains authoritative.
    pub forward_area_m2: [f64; 2],
    /// Wet horizontal width bounds `[lower, upper]`, in metres. Absent when dry.
    pub top_width_m: Option<[f64; 2]>,
    /// Hydrostatic integral bounds `[lower, upper]`, in cubic metres.
    pub hydrostatic_integral_m3: [f64; 2],
    /// Requested area divided by width bounds `[lower, upper]`, in metres. Absent when dry.
    pub hydraulic_depth_m: Option<[f64; 2]>,
}
