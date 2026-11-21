//! Barrel batches, depth tensors and containment checks shared by the culvert barrel tests.

use super::reference_rows::State;
use super::runtime::{Runtime, client};
use hydr::structures::{
    BarrelState, CulvertBarrel, CulvertBarrelBatch, CulvertBarrelGeometryRow, StructureError,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

pub type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Quantity names in fixture order.
pub const QUANTITIES: [&str; 5] = ["area", "top_width", "perimeter", "radius", "moment"];

pub use super::lanes::barrel;

pub fn batch(barrels: Vec<CulvertBarrel>) -> Result<CulvertBarrelBatch<Runtime>, StructureError> {
    CulvertBarrelBatch::new(client(), barrels)
}

pub fn tensor(values: &[f64]) -> Result<Tensor<Runtime>, numr::error::Error> {
    Tensor::from_slice(values, &[values.len()], client().device())
}

pub fn geometry(
    batch: &CulvertBarrelBatch<Runtime>,
    depths: &[f64],
) -> Result<Vec<CulvertBarrelGeometryRow>, Box<dyn std::error::Error>> {
    Ok(batch.geometry(&tensor(depths)?)?.project()?)
}

/// `(lower, upper)` of each quantity in fixture order.
pub fn quantities(row: &CulvertBarrelGeometryRow) -> [(f64, f64); 5] {
    [
        row.area_m2,
        row.top_width_m,
        row.wetted_perimeter_m,
        row.hydraulic_radius_m,
        row.hydrostatic_moment_m3,
    ]
}

pub fn state(value: State) -> BarrelState {
    match value {
        State::Dry => BarrelState::Dry,
        State::Free => BarrelState::Free,
        State::Full => BarrelState::Full,
    }
}

/// True when the exact value with round-down and round-up bits `reference` lies in `bounds`.
pub fn contains(bounds: (f64, f64), reference: [u64; 2]) -> bool {
    bounds.0 <= f64::from_bits(reference[0]) && f64::from_bits(reference[1]) <= bounds.1
}
