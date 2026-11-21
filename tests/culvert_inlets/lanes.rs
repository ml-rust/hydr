//! Inlet-control lanes from the frozen inlet rows, on any runtime.
//!
//! Each lane is one row and one discharge. `Debug` prints each F64 in round-trip form, so equal
//! text means equal bits, signed zero included.

use super::inlet_rows::{BARRELS, GRAVITY, InletRow, ROWS, SLOPES};
use hydr::structures::{CulvertBarrel, CulvertInlet, CulvertInletBatch, CulvertInletRow};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::CertifiedBisectionAlgorithms;

pub type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// Batch widths around one and two vector widths.
pub const WIDTHS: [usize; 6] = [1, 31, 32, 33, 64, 65];

/// Most lanes per layout check. Width 65 at its starting offset still wraps once.
const MOST_LANES: usize = 66;

/// One batch row and the discharge it is queried at.
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    pub row: CulvertInletRow,
    pub discharge: f64,
}

/// Inlet row of the frozen barrel, code and slope of `row`, at the frozen gravity.
pub fn inlet_row(row: &InletRow) -> Outcome<CulvertInletRow> {
    let (circular, rise, span) = *BARRELS
        .get(row.barrel)
        .ok_or_else(|| format!("{}: no barrel {}", row.label, row.barrel))?;
    let rise = f64::from_bits(rise);
    let barrel = if circular {
        CulvertBarrel::circular(rise)?
    } else {
        CulvertBarrel::rectangular_box(f64::from_bits(span), rise)?
    };
    let slope = SLOPES
        .get(row.slope)
        .ok_or_else(|| format!("{}: no slope {}", row.label, row.slope))?;
    Ok(CulvertInletRow {
        barrel,
        inlet: CulvertInlet::hds5(row.code)?,
        slope: f64::from_bits(*slope),
        gravity_m_per_s2: f64::from_bits(GRAVITY),
    })
}

/// One lane per frozen inlet row, in fixture order.
pub fn reference_lanes() -> Outcome<Vec<Lane>> {
    ROWS.iter()
        .map(|row| {
            Ok(Lane {
                row: inlet_row(row)?,
                discharge: f64::from_bits(row.discharge),
            })
        })
        .collect()
}

/// At most `MOST_LANES` evenly spaced reference lanes.
pub fn lanes() -> Outcome<Vec<Lane>> {
    let all = reference_lanes()?;
    let count = all.len();
    if count <= MOST_LANES {
        return Ok(all);
    }
    Ok((0..MOST_LANES)
        .filter_map(|i| all.get(i * count / MOST_LANES).copied())
        .collect())
}

/// Lane indices of a batch of `width` rows, starting at offset `width` and wrapping.
pub fn layout(width: usize, count: usize) -> Vec<usize> {
    (0..width).map(|j| (width + j) % count).collect()
}

/// Runs the headwater query over `lanes` in one batch and returns one `Debug` text per row.
pub fn outputs<R>(client: &R::Client, lanes: &[Lane]) -> Outcome<Vec<String>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + CorrectlyRoundedOps<R>
        + RuntimeClient<R>
        + CertifiedBisectionAlgorithms<R>,
{
    let batch =
        CulvertInletBatch::<R>::new(client.clone(), lanes.iter().map(|lane| lane.row).collect())?;
    let flows: Vec<f64> = lanes.iter().map(|lane| lane.discharge).collect();
    let discharge = Tensor::<R>::from_slice(&flows, &[flows.len()], client.device())?;
    let rows = batch.headwater(&discharge)?.project()?;
    Ok(rows.iter().map(|row| format!("{row:?}")).collect())
}
