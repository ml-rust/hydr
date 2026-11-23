//! Outlet-control lanes from the frozen outlet rows, on any runtime.
//!
//! Each lane is one row, one discharge and one tailwater. `Debug` prints each F64 in round-trip
//! form, so equal text means equal bits, signed zero and NaN payloads aside.

use super::outlet_rows::{FULL_ROWS, GRAVITY, Inputs, OutletRow, PROFILE_ROWS};
use hydr::structures::{CulvertBarrel, CulvertOutletBatch, CulvertOutletRow};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedBisectionAlgorithms;

pub type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// Batch widths: one row, and one past each of 32 and 64 lanes.
pub const WIDTHS: [usize; 3] = [1, 33, 65];

/// One batch row with the discharge and tailwater it is queried at.
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    pub row: CulvertOutletRow,
    pub discharge: f64,
    pub tailwater: f64,
}

/// Outlet row of the frozen inputs at the frozen gravity.
pub fn outlet_row(inputs: &Inputs) -> Outcome<CulvertOutletRow> {
    let rise = f64::from_bits(inputs.rise);
    let barrel = if inputs.circular {
        CulvertBarrel::circular(rise)?
    } else {
        CulvertBarrel::rectangular_box(f64::from_bits(inputs.span), rise)?
    };
    Ok(CulvertOutletRow {
        barrel,
        length_m: f64::from_bits(inputs.length),
        manning_n: f64::from_bits(inputs.manning),
        entrance_loss: f64::from_bits(inputs.ke),
        slope: f64::from_bits(inputs.slope),
        gravity_m_per_s2: f64::from_bits(GRAVITY),
        face_blockage: hydr::structures::BlockageFraction::NONE,
    })
}

/// The lane of one frozen row.
pub fn lane(row: &OutletRow) -> Outcome<Lane> {
    Ok(Lane {
        row: outlet_row(&row.inputs)?,
        discharge: f64::from_bits(row.inputs.discharge),
        tailwater: f64::from_bits(row.inputs.tailwater),
    })
}

/// One lane per frozen row: the full-barrel rows, then the profile rows.
pub fn reference_lanes() -> Outcome<Vec<Lane>> {
    FULL_ROWS.iter().chain(PROFILE_ROWS).map(lane).collect()
}

/// Profile rows of the layout checks: every regime and jump status on both shapes.
const LAYOUT_ROWS: [(&str, bool); 9] = [
    ("full_mild", true),
    ("m1", false),
    ("m2_critical_outlet", true),
    ("rising_to_crown", false),
    ("adverse", true),
    ("full_outlet_free_inlet", false),
    ("steep_supercritical", true),
    ("steep_jump", false),
    ("steep_drowned", true),
];

/// The first full-barrel row and the layout profile rows.
pub fn layout_lanes() -> Outcome<Vec<Lane>> {
    let mut lanes = vec![lane(FULL_ROWS.first().ok_or("no full-barrel row")?)?];
    for (name, circular) in LAYOUT_ROWS {
        let row = PROFILE_ROWS
            .iter()
            .find(|row| row.name == name && row.inputs.circular == circular)
            .ok_or_else(|| format!("no profile row {name}"))?;
        lanes.push(lane(row)?);
    }
    Ok(lanes)
}

/// Lane indices of a batch of `width` rows, starting at offset `width` and wrapping.
pub fn layout(width: usize, count: usize) -> Vec<usize> {
    (0..width).map(|j| (width + j) % count).collect()
}

/// Batch and query tensors of `lanes` on `client`.
pub fn query<R>(
    client: &R::Client,
    lanes: &[Lane],
) -> Outcome<(CulvertOutletBatch<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + CorrectlyRoundedOps<R>
        + RuntimeClient<R>,
{
    let batch =
        CulvertOutletBatch::<R>::new(client.clone(), lanes.iter().map(|lane| lane.row).collect())?;
    let flows: Vec<f64> = lanes.iter().map(|lane| lane.discharge).collect();
    let tails: Vec<f64> = lanes.iter().map(|lane| lane.tailwater).collect();
    let discharge = Tensor::<R>::from_slice(&flows, &[flows.len()], client.device())?;
    let tailwater = Tensor::<R>::from_slice(&tails, &[tails.len()], client.device())?;
    Ok((batch, discharge, tailwater))
}

/// Runs the headwater and jump queries over `lanes` in one batch, one `Debug` text per row.
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
        + CertifiedBisectionAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let (batch, discharge, tailwater) = query::<R>(client, lanes)?;
    let found = batch.jump(&discharge, &tailwater)?;
    let headwater = found.headwater().project()?;
    let jump = found.project()?;
    Ok(headwater
        .iter()
        .zip(&jump)
        .map(|(row, jump)| format!("{row:?} {jump:?}"))
        .collect())
}
