//! Batch layouts: arc, circle and box lanes return the bits of their single-row queries.
//!
//! `Debug` prints each F64 in round-trip form, so equal text means equal bits.

use super::geometry::{Outcome, tensor};
use super::reference_rows::{BARRELS, ROWS};
use super::runtime::{Runtime, client};
use super::{TestResult, barrel};
use hydr::structures::{CulvertBarrel, CulvertBarrelBatch, CulvertBarrelShape};

/// Batch widths around one and two vector widths.
const WIDTHS: [usize; 3] = [1, 33, 65];

/// Circle and box lanes mixed into the arc lanes, as depth fractions of their rise.
const CLOSED_FRACTIONS: [f64; 6] = [0.0, 1e-3, 0.05, 0.5, 0.999, 1.25];

/// One row: its barrel and the depth range it reads. A point query reads the upper end.
#[derive(Clone, Copy, Debug)]
struct Lane {
    barrel: CulvertBarrel,
    lower: f64,
    upper: f64,
}

/// Every reference row of every arc barrel, with a circle and a box row after every third.
fn lanes() -> Outcome<Vec<Lane>> {
    let closed = [
        CulvertBarrel::circular(1.3)?,
        CulvertBarrel::rectangular_box(2.0, 1.1)?,
    ];
    let mut made = Vec::new();
    for (k, row) in ROWS.iter().enumerate() {
        let entry = BARRELS.get(row.barrel).ok_or("row names no barrel")?;
        let depth = f64::from_bits(row.depth);
        made.push(Lane {
            barrel: barrel(entry)?,
            lower: 0.5 * depth,
            upper: depth,
        });
        if k % 3 == 2 {
            let index = (k / 3) % (closed.len() * CLOSED_FRACTIONS.len());
            let shape = closed[index % closed.len()];
            let fraction = CLOSED_FRACTIONS[index / closed.len()];
            let depth = fraction * shape.rise_m();
            made.push(Lane {
                barrel: shape,
                lower: 0.5 * depth,
                upper: depth,
            });
        }
    }
    Ok(made)
}

/// Point and range geometry of `lanes` as one text per row. Range rows reaching the crown
/// take the point query alone.
fn texts(lanes: &[Lane]) -> Outcome<Vec<String>> {
    let barrels: Vec<CulvertBarrel> = lanes.iter().map(|lane| lane.barrel).collect();
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    let upper: Vec<f64> = lanes.iter().map(|lane| lane.upper).collect();
    let points = batch.geometry(&tensor(&upper)?)?.project()?;
    let free: Vec<bool> = lanes
        .iter()
        .map(|lane| lane.upper < lane.barrel.rise_m())
        .collect();
    let lower: Vec<f64> = lanes
        .iter()
        .zip(&free)
        .map(|(lane, &free)| if free { lane.lower } else { 0.0 })
        .collect();
    let top: Vec<f64> = lanes
        .iter()
        .zip(&free)
        .map(|(lane, &free)| if free { lane.upper } else { 0.0 })
        .collect();
    let spans = batch
        .geometry_over(&tensor(&lower)?, &tensor(&top)?)?
        .project()?;
    Ok(points
        .iter()
        .zip(&spans)
        .zip(&free)
        .map(|((point, span), &free)| {
            if free {
                format!("{point:?} {span:?}")
            } else {
                format!("{point:?}")
            }
        })
        .collect())
}

#[test]
fn arc_barrel_queries_return_single_row_bits_at_every_layout_width() -> TestResult {
    let lanes = lanes()?;
    let mut singles = Vec::with_capacity(lanes.len());
    for lane in &lanes {
        let [single] = <[String; 1]>::try_from(texts(std::slice::from_ref(lane))?)
            .map_err(|rows| format!("single query returned {} rows", rows.len()))?;
        singles.push(single);
    }
    for width in WIDTHS {
        // Each width starts elsewhere, so one lane sits at a different position per width.
        let picked: Vec<usize> = (0..width).map(|j| (width + j) % lanes.len()).collect();
        let batch: Vec<Lane> = picked
            .iter()
            .filter_map(|&i| lanes.get(i).copied())
            .collect();
        let rows = texts(&batch)?;
        assert_eq!(rows.len(), width);
        for (position, (row, &index)) in rows.iter().zip(&picked).enumerate() {
            let single = singles.get(index).ok_or("missing single row")?;
            assert_eq!(row, single, "width {width} lane {position}");
        }
    }
    Ok(())
}

#[test]
fn circle_and_box_rows_keep_their_bits_beside_arc_rows() -> TestResult {
    let lanes = lanes()?;
    let plain = |lane: &Lane| {
        matches!(
            lane.barrel.shape(),
            CulvertBarrelShape::Circular | CulvertBarrelShape::Box
        )
    };
    let closed: Vec<Lane> = lanes.iter().filter(|lane| plain(lane)).copied().collect();
    let mixed = texts(&lanes)?;
    let alone = texts(&closed)?;
    let beside: Vec<&String> = lanes
        .iter()
        .zip(&mixed)
        .filter(|(lane, _)| plain(lane))
        .map(|(_, text)| text)
        .collect();
    assert_eq!(beside.len(), alone.len());
    for (with_arcs, without) in beside.iter().zip(&alone) {
        assert_eq!(*with_arcs, without);
    }
    Ok(())
}
