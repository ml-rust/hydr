//! Inlet batches and enclosure checks shared by the culvert inlet tests.

use super::inlet_rows::Regime;
use super::lanes::Lane;
use super::runtime::{Runtime, client};
use hydr::structures::{CulvertInletBatch, CulvertInletHeadwaterRow, CulvertInletRow, InletRegime};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

pub type TestResult = Result<(), Box<dyn std::error::Error>>;

pub fn tensor(values: &[f64]) -> Result<Tensor<Runtime>, numr::error::Error> {
    Tensor::from_slice(values, &[values.len()], client().device())
}

pub fn batch(
    rows: Vec<CulvertInletRow>,
) -> Result<CulvertInletBatch<Runtime>, hydr::structures::StructureError> {
    CulvertInletBatch::new(client(), rows)
}

/// Headwater rows of one batch over `lanes`.
pub fn headwater(
    lanes: &[Lane],
) -> Result<Vec<CulvertInletHeadwaterRow>, Box<dyn std::error::Error>> {
    let batch = batch(lanes.iter().map(|lane| lane.row).collect())?;
    let flows: Vec<f64> = lanes.iter().map(|lane| lane.discharge).collect();
    Ok(batch.headwater(&tensor(&flows)?)?.project()?)
}

/// Requires `bounds` to hold the exact value with round-down and round-up bits `reference`, and
/// every point of `bounds` to lie within `relative` of it.
pub fn holds(bounds: (f64, f64), reference: [u64; 2], relative: f64, what: &str) -> TestResult {
    let (low, high) = (f64::from_bits(reference[0]), f64::from_bits(reference[1]));
    if !(bounds.0 <= low && high <= bounds.1) {
        return Err(format!(
            "{what}: [{:e}, {:e}] misses the reference [{low:e}, {high:e}]",
            bounds.0, bounds.1
        )
        .into());
    }
    let reach = (high - bounds.0).max(bounds.1 - low);
    let allowed = relative * low.abs().max(high.abs());
    if reach > allowed {
        return Err(format!(
            "{what}: enclosure reaches {reach:e} from the reference, above {allowed:e}"
        )
        .into());
    }
    Ok(())
}

/// Whether the reported regime is the reference regime or a limit next to it.
pub fn regime_agrees(found: InletRegime, expected: Regime) -> bool {
    use InletRegime as I;
    matches!(
        (found, expected),
        (I::Unsubmerged | I::UnsubmergedLimit, Regime::Unsubmerged)
            | (
                I::UnsubmergedLimit | I::Transition | I::SubmergedLimit,
                Regime::Transition
            )
            | (I::SubmergedLimit | I::Submerged, Regime::Submerged)
    )
}
