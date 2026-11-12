pub type Bounds = [u64; 2];
#[derive(Clone, Copy)]
pub struct Profile {
    pub id: &'static str,
    pub coordinates: &'static [[u64; 2]],
    pub cap_stage: u64,
}
#[derive(Clone, Copy)]
pub struct Case {
    pub id: &'static str,
    pub owner: usize,
    pub input_high: u64,
    pub input_low: u64,
    pub resolved: bool,
    pub dry: bool,
    pub geometry: [Option<Bounds>; 5],
    pub true_band_indices: &'static [usize],
    pub perimeter: Bounds,
    pub radius: Option<Bounds>,
    pub perimeter_allowance: u64,
    pub radius_allowance: u64,
    pub conditioned_radius: bool,
    pub radius_positive_lower: bool,
}
pub const GEOMETRY_ORDER: [&str; 5] = [
    "depth",
    "forward_area",
    "width",
    "integral",
    "hydraulic_depth",
];
#[path = "asymmetric_pointed_same_area_cases.rs"]
mod asymmetric_pointed_same_area_cases;
#[path = "cap_benches_excluded_cases.rs"]
mod cap_benches_excluded_cases;
#[path = "case_index.rs"]
mod case_index;
#[path = "decimal_elevation_knot_cases.rs"]
mod decimal_elevation_knot_cases;
#[path = "decimal_original_coordinates_cases.rs"]
mod decimal_original_coordinates_cases;
#[path = "multiple_bands_cases.rs"]
mod multiple_bands_cases;
#[path = "profile_data.rs"]
mod profile_data;
#[path = "radius_decreases_across_widening_band_cases.rs"]
mod radius_decreases_across_widening_band_cases;
#[path = "rectangle_cases.rs"]
mod rectangle_cases;
#[path = "symmetric_pointed_cases.rs"]
mod symmetric_pointed_cases;
#[path = "three_band_cases.rs"]
mod three_band_cases;
#[path = "translated_large_datum_cases.rs"]
mod translated_large_datum_cases;
#[path = "trapezoid_cases.rs"]
mod trapezoid_cases;
#[path = "triangle_cases.rs"]
mod triangle_cases;
#[path = "vertical_public_range_cases.rs"]
mod vertical_public_range_cases;
pub use case_index::CASES;
pub use profile_data::PROFILES;
