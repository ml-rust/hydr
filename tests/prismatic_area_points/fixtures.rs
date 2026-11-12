pub type Bounds = [u64; 2];
#[derive(Clone, Copy)]
pub struct Profile {
    pub id: &'static str,
    pub coordinates: &'static [[u64; 2]],
    pub cap_stage: u64,
}
#[derive(Clone, Copy)]
pub struct BandCase {
    pub band_index: usize,
    pub bounds: [Bounds; 4],
    pub allowances: [u64; 4],
    pub exact_zero: [bool; 4],
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
    pub bands: &'static [BandCase],
}
#[derive(Clone, Copy)]
pub struct FormulaCase {
    pub id: &'static str,
    pub slope: u64,
    pub area: u64,
    pub depth: Bounds,
}
pub const GEOMETRY_ORDER: [&str; 5] = [
    "depth",
    "forward_area",
    "width",
    "integral",
    "hydraulic_depth",
];
pub const LOCAL_ORDER: [&str; 4] = ["lowerarea", "upperarea", "lowerdepth", "upperdepth"];
#[path = "case_index.rs"]
mod case_index;
#[path = "formula_data.rs"]
mod formula_data;
#[path = "profile_data.rs"]
mod profile_data;
#[path = "rectangle_cases.rs"]
mod rectangle_cases;
#[path = "three_band_cases.rs"]
mod three_band_cases;
#[path = "trapezoid_cases.rs"]
mod trapezoid_cases;
#[path = "triangle_cases.rs"]
mod triangle_cases;
pub use case_index::CASES;
pub use formula_data::FORMULAS;
pub use profile_data::PROFILES;
