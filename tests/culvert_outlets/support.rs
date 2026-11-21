//! Outlet batches and enclosure checks shared by the culvert outlet tests.

use super::lanes::{Lane, query};
use super::runtime::{Runtime, client};
use hydr::structures::{CulvertOutletHeadwaterRow, CulvertOutletJumpRow};

pub type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Headwater and jump rows of one query.
pub type Evaluated = (Vec<CulvertOutletHeadwaterRow>, Vec<CulvertOutletJumpRow>);

/// Headwater and jump rows of one batch over `lanes`.
pub fn evaluate(lanes: &[Lane]) -> Result<Evaluated, Box<dyn std::error::Error>> {
    let (batch, discharge, tailwater) = query::<Runtime>(&client(), lanes)?;
    let found = batch.jump(&discharge, &tailwater)?;
    Ok((found.headwater().project()?, found.project()?))
}

/// Headwater rows of one batch over `lanes`.
pub fn headwater(
    lanes: &[Lane],
) -> Result<Vec<CulvertOutletHeadwaterRow>, Box<dyn std::error::Error>> {
    let (batch, discharge, tailwater) = query::<Runtime>(&client(), lanes)?;
    Ok(batch.headwater(&discharge, &tailwater)?.project()?)
}

/// Requires `bounds` to hold the exact value with round-down and round-up bits `reference`,
/// and every point of `bounds` to lie within `allowed` of it.
pub fn holds_within(
    bounds: (f64, f64),
    reference: [u64; 2],
    allowed: f64,
    what: &str,
) -> TestResult {
    let (low, high) = (f64::from_bits(reference[0]), f64::from_bits(reference[1]));
    if !(bounds.0 <= low && high <= bounds.1) {
        return Err(format!(
            "{what}: [{:e}, {:e}] misses the reference [{low:e}, {high:e}]",
            bounds.0, bounds.1
        )
        .into());
    }
    let reach = (high - bounds.0).max(bounds.1 - low);
    if reach > allowed {
        return Err(format!(
            "{what}: enclosure reaches {reach:e} from the reference, above {allowed:e}"
        )
        .into());
    }
    Ok(())
}

/// [`holds_within`] with a tolerance relative to the reference.
pub fn holds(bounds: (f64, f64), reference: [u64; 2], relative: f64, what: &str) -> TestResult {
    let scale = f64::from_bits(reference[0])
        .abs()
        .max(f64::from_bits(reference[1]).abs());
    holds_within(bounds, reference, relative * scale, what)
}
