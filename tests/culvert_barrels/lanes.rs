//! Mixed circle and box lanes for the four barrel queries, on any runtime.
//!
//! Each query returns one `Debug` text per row. `Debug` prints each F64 in round-trip form,
//! so equal text means equal bits, signed zero included.

use super::critical_rows::{CRITICAL, GRAVITY};
use super::dense_rows::DENSE;
use super::normal_rows::NORMAL;
use super::reference_rows::{BARRELS, Barrel, ROWS};
use hydr::structures::{CulvertBarrel, CulvertBarrelBatch, StructureError};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::CertifiedBisectionAlgorithms;
use std::fmt::Debug;

pub type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// Batch widths around one and two vector widths.
pub const WIDTHS: [usize; 6] = [1, 31, 32, 33, 64, 65];

/// Most lanes per query. Width 65 at its starting offset still wraps once.
const MOST_LANES: usize = 66;

/// One public barrel query.
#[derive(Clone, Copy, Debug)]
pub enum Query {
    /// `geometry` at `values[0]`.
    Geometry,
    /// `geometry_over` from `values[0]` to `values[1]`.
    Range,
    /// `critical_depth` for discharge `values[0]` at the frozen gravity.
    Critical,
    /// `normal_depth` for discharge, Manning n and slope in `values`.
    Normal,
}

pub const QUERIES: [Query; 4] = [
    Query::Geometry,
    Query::Range,
    Query::Critical,
    Query::Normal,
];

/// One row of a batch: its barrel and the query inputs it reads.
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    pub barrel: CulvertBarrel,
    pub values: [f64; 3],
}

pub fn barrel(entry: &Barrel) -> Result<CulvertBarrel, StructureError> {
    let rise = f64::from_bits(entry.rise);
    if entry.circular {
        CulvertBarrel::circular(rise)
    } else {
        CulvertBarrel::rectangular_box(f64::from_bits(entry.span), rise)
    }
}

/// Takes one row of each barrel in turn, so circle and box rows alternate in groups of three.
fn interleaved<T>(rows: &[T], barrel_of: fn(&T) -> usize) -> Vec<&T> {
    let mut groups: Vec<Vec<&T>> = (0..BARRELS.len()).map(|_| Vec::new()).collect();
    for row in rows {
        groups[barrel_of(row)].push(row);
    }
    let longest = groups.iter().map(Vec::len).max().unwrap_or(0);
    (0..longest)
        .flat_map(|k| groups.iter().filter_map(move |group| group.get(k).copied()))
        .collect()
}

/// Keeps at most `MOST_LANES` evenly spaced rows.
fn spaced<T: Copy>(rows: Vec<T>) -> Vec<T> {
    let count = rows.len();
    if count <= MOST_LANES {
        return rows;
    }
    (0..MOST_LANES)
        .map(|i| rows[i * count / MOST_LANES])
        .collect()
}

fn lane(barrel_index: usize, values: [f64; 3]) -> Result<Lane, StructureError> {
    Ok(Lane {
        barrel: barrel(&BARRELS[barrel_index])?,
        values,
    })
}

/// Evenly spaced lanes of `query` for the layout widths, circles and boxes interleaved.
pub fn lanes(query: Query) -> Result<Vec<Lane>, StructureError> {
    Ok(spaced(reference_lanes(query)?))
}

/// One lane per dense-sweep depth of each shape, for point geometry.
pub fn dense_lanes() -> Result<Vec<Lane>, StructureError> {
    let mut made = Vec::new();
    for (entry, rows) in DENSE {
        let culvert = barrel(entry)?;
        made.extend(rows.iter().map(|row| Lane {
            barrel: culvert,
            values: [f64::from_bits(row.depth), 0.0, 0.0],
        }));
    }
    Ok(made)
}

/// Lanes of `query` over every frozen reference row, circles and boxes interleaved.
pub fn reference_lanes(query: Query) -> Result<Vec<Lane>, StructureError> {
    Ok(match query {
        Query::Geometry => interleaved(ROWS, |row| row.barrel)
            .into_iter()
            .map(|row| lane(row.barrel, [f64::from_bits(row.depth), 0.0, 0.0]))
            .collect::<Result<_, _>>()?,
        Query::Range => interleaved(ROWS, |row| row.barrel)
            .into_iter()
            .filter(|row| f64::from_bits(row.depth) < f64::from_bits(BARRELS[row.barrel].rise))
            .map(|row| {
                let depth = f64::from_bits(row.depth);
                lane(row.barrel, [0.5 * depth, depth, 0.0])
            })
            .collect::<Result<_, _>>()?,
        Query::Critical => interleaved(CRITICAL, |row| row.barrel)
            .into_iter()
            .map(|row| lane(row.barrel, [f64::from_bits(row.discharge), 0.0, 0.0]))
            .collect::<Result<_, _>>()?,
        Query::Normal => interleaved(NORMAL, |row| row.barrel)
            .into_iter()
            .map(|row| {
                let values = [row.discharge, row.manning_n, row.slope].map(f64::from_bits);
                lane(row.barrel, values)
            })
            .collect::<Result<_, _>>()?,
    })
}

/// Lane indices of a batch of `width` rows, starting at offset `width` and wrapping.
///
/// Each width starts elsewhere, so one lane input sits at a different position per width.
pub fn layout(width: usize, count: usize) -> Vec<usize> {
    (0..width).map(|j| (width + j) % count).collect()
}

fn texts<T: Debug>(rows: Vec<T>) -> Vec<String> {
    rows.iter().map(|row| format!("{row:?}")).collect()
}

/// Runs `query` over `lanes` in one batch and returns one `Debug` text per row.
pub fn outputs<R>(client: &R::Client, query: Query, lanes: &[Lane]) -> Outcome<Vec<String>>
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
    let batch = CulvertBarrelBatch::<R>::new(
        client.clone(),
        lanes.iter().map(|lane| lane.barrel).collect(),
    )?;
    let column = |k: usize| {
        let values: Vec<f64> = lanes.iter().map(|lane| lane.values[k]).collect();
        Tensor::<R>::from_slice(&values, &[values.len()], client.device())
    };
    Ok(match query {
        Query::Geometry => texts(batch.geometry(&column(0)?)?.project()?),
        Query::Range => texts(batch.geometry_over(&column(0)?, &column(1)?)?.project()?),
        Query::Critical => texts(
            batch
                .critical_depth(&column(0)?, f64::from_bits(GRAVITY))?
                .project()?,
        ),
        Query::Normal => texts(
            batch
                .normal_depth(&column(0)?, &column(1)?, &column(2)?)?
                .project()?,
        ),
    })
}

/// Full-state geometry of every lane barrel, one `Debug` text per row.
pub fn full_outputs<R>(client: &R::Client, lanes: &[Lane]) -> Outcome<Vec<String>>
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
    let batch = CulvertBarrelBatch::<R>::new(
        client.clone(),
        lanes.iter().map(|lane| lane.barrel).collect(),
    )?;
    Ok(texts(batch.full_geometry()?.project()?))
}
