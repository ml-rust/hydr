//! Batch layouts of the arc lanes and their query runners, on any runtime.
//!
//! Each runner returns one `Debug` text per row. `Debug` prints each F64 in round-trip form, so
//! equal text means equal bits, signed zero included.

use super::depth_rows;
use super::lanes::{BarrelLane, InletLane, Outcome, OutletLane, Query};
use hydr::structures::{CulvertBarrelBatch, CulvertClient, CulvertInletBatch, CulvertOutletBatch};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::fmt::Debug;

/// Lane indices of a batch of `width` rows, starting at offset `width` and wrapping.
///
/// Each width starts elsewhere, so one lane input sits at a different position per width.
pub fn layout(width: usize, count: usize) -> Vec<usize> {
    (0..width).map(|j| (width + j) % count).collect()
}

/// The lanes at `layout(width, lanes.len())`.
pub fn picked<T: Copy>(width: usize, lanes: &[T]) -> Vec<T> {
    layout(width, lanes.len())
        .into_iter()
        .filter_map(|i| lanes.get(i).copied())
        .collect()
}

fn texts<T: Debug>(rows: Vec<T>) -> Vec<String> {
    rows.iter().map(|row| format!("{row:?}")).collect()
}

fn column<R: Runtime<DType = DType>>(
    client: &R::Client,
    values: Vec<f64>,
) -> Result<Tensor<R>, numr::error::Error> {
    Tensor::<R>::from_slice(&values, &[values.len()], client.device())
}

/// Runs `query` over `lanes` in one batch and returns one `Debug` text per row.
pub fn barrel_outputs<R>(
    client: &R::Client,
    query: Query,
    lanes: &[BarrelLane],
) -> Outcome<Vec<String>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let batch = CulvertBarrelBatch::<R>::new(
        client.clone(),
        lanes.iter().map(|lane| lane.barrel).collect(),
    )?;
    let value = |k: usize| column::<R>(client, lanes.iter().map(|lane| lane.values[k]).collect());
    Ok(match query {
        Query::Geometry => texts(batch.geometry(&value(0)?)?.project()?),
        Query::Range => texts(batch.geometry_over(&value(0)?, &value(1)?)?.project()?),
        Query::Full => texts(batch.full_geometry()?.project()?),
        Query::Critical => texts(
            batch
                .critical_depth(&value(0)?, f64::from_bits(depth_rows::GRAVITY))?
                .project()?,
        ),
        Query::Normal => texts(
            batch
                .normal_depth(&value(0)?, &value(1)?, &value(2)?)?
                .project()?,
        ),
    })
}

/// Runs the headwater query over `lanes` in one batch and returns one `Debug` text per row.
pub fn inlet_outputs<R>(client: &R::Client, lanes: &[InletLane]) -> Outcome<Vec<String>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let batch =
        CulvertInletBatch::<R>::new(client.clone(), lanes.iter().map(|lane| lane.row).collect())?;
    let discharge = column::<R>(client, lanes.iter().map(|lane| lane.discharge).collect())?;
    Ok(texts(batch.headwater(&discharge)?.project()?))
}

/// Runs the headwater and jump queries over `lanes` in one batch, one `Debug` text per row.
pub fn outlet_outputs<R>(client: &R::Client, lanes: &[OutletLane]) -> Outcome<Vec<String>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let batch =
        CulvertOutletBatch::<R>::new(client.clone(), lanes.iter().map(|lane| lane.row).collect())?;
    let discharge = column::<R>(client, lanes.iter().map(|lane| lane.discharge).collect())?;
    let tailwater = column::<R>(client, lanes.iter().map(|lane| lane.tailwater).collect())?;
    let found = batch.jump(&discharge, &tailwater)?;
    let headwater = found.headwater().project()?;
    let jump = found.project()?;
    Ok(headwater
        .iter()
        .zip(&jump)
        .map(|(row, jump)| format!("{row:?} {jump:?}"))
        .collect())
}
