//! Batch layouts: arc, circle and box lanes return the bits of their single-row queries, for
//! geometry, depths, inlet control and outlet control.
//!
//! `Debug` prints each F64 in round-trip form, so equal text means equal bits.

use super::TestResult;
use super::lane_outputs::{barrel_outputs, inlet_outputs, layout, outlet_outputs};
use super::lanes::{
    BarrelLane, InletLane, Outcome, OutletLane, QUERIES, Query, WIDTHS, barrel_lanes, inlet_lanes,
    outlet_lanes,
};
use super::runtime::{Runtime, client};
use hydr::structures::{CulvertBarrel, CulvertBarrelShape};

/// Requires every lane of every width to equal its single-row text.
///
/// `run` evaluates one batch of lanes. `what` names the query in errors.
fn same_at_every_width<T: Copy>(
    what: &str,
    lanes: &[T],
    run: impl Fn(&[T]) -> Outcome<Vec<String>>,
) -> TestResult {
    let mut singles = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let [single] = <[String; 1]>::try_from(run(std::slice::from_ref(lane))?)
            .map_err(|rows| format!("{what} single query returned {} rows", rows.len()))?;
        singles.push(single);
    }
    for width in WIDTHS {
        let picked = layout(width, lanes.len());
        let batch: Vec<T> = picked
            .iter()
            .filter_map(|&i| lanes.get(i).copied())
            .collect();
        let rows = run(&batch)?;
        if rows.len() != width {
            return Err(format!("{what} width {width} returned {} rows", rows.len()).into());
        }
        for (position, (row, &index)) in rows.iter().zip(&picked).enumerate() {
            let single = singles.get(index).ok_or("missing single row")?;
            if row != single {
                return Err(format!(
                    "{what} width {width} lane {position}: {row} differs from the single-row {single}"
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Layout check of one barrel query.
fn barrel_layouts(query: Query) -> TestResult {
    let client = client();
    same_at_every_width(&format!("{query:?}"), &barrel_lanes(query)?, |batch| {
        barrel_outputs::<Runtime>(&client, query, batch)
    })
}

#[test]
fn arc_point_geometry_returns_single_row_bits_at_every_layout_width() -> TestResult {
    barrel_layouts(Query::Geometry)
}

#[test]
fn arc_range_geometry_returns_single_row_bits_at_every_layout_width() -> TestResult {
    barrel_layouts(Query::Range)
}

#[test]
fn arc_full_geometry_returns_single_row_bits_at_every_layout_width() -> TestResult {
    barrel_layouts(Query::Full)
}

#[test]
fn arc_critical_depth_returns_single_row_bits_at_every_layout_width() -> TestResult {
    barrel_layouts(Query::Critical)
}

#[test]
fn arc_normal_depth_returns_single_row_bits_at_every_layout_width() -> TestResult {
    barrel_layouts(Query::Normal)
}

#[test]
fn arc_inlet_control_returns_single_row_bits_at_every_layout_width() -> TestResult {
    let client = client();
    same_at_every_width("inlet", &inlet_lanes()?, |batch| {
        inlet_outputs::<Runtime>(&client, batch)
    })
}

#[test]
fn arc_outlet_control_returns_single_row_bits_at_every_layout_width() -> TestResult {
    let client = client();
    same_at_every_width("outlet", &outlet_lanes()?, |batch| {
        outlet_outputs::<Runtime>(&client, batch)
    })
}

fn plain(barrel: &CulvertBarrel) -> bool {
    matches!(
        barrel.shape(),
        CulvertBarrelShape::Circular | CulvertBarrelShape::Box
    )
}

/// Requires the circle and box lanes to return the same text beside the arc lanes as alone.
fn closed_unchanged<T: Copy>(
    what: &str,
    lanes: &[T],
    closed: impl Fn(&T) -> bool,
    run: impl Fn(&[T]) -> Outcome<Vec<String>>,
) -> TestResult {
    let alone: Vec<T> = lanes.iter().filter(|lane| closed(lane)).copied().collect();
    let mixed = run(lanes)?;
    let beside: Vec<&String> = lanes
        .iter()
        .zip(&mixed)
        .filter(|(lane, _)| closed(lane))
        .map(|(_, text)| text)
        .collect();
    let alone = run(&alone)?;
    if beside.len() != alone.len() || beside.is_empty() {
        return Err(format!("{what}: {} closed lanes beside arcs", beside.len()).into());
    }
    for (with_arcs, without) in beside.iter().zip(&alone) {
        if *with_arcs != without {
            return Err(format!("{what}: {with_arcs} beside arcs, {without} alone").into());
        }
    }
    Ok(())
}

#[test]
fn circle_and_box_rows_keep_their_bits_beside_arc_rows() -> TestResult {
    let client = client();
    for query in QUERIES {
        closed_unchanged(
            &format!("{query:?}"),
            &barrel_lanes(query)?,
            |lane: &BarrelLane| plain(&lane.barrel),
            |batch| barrel_outputs::<Runtime>(&client, query, batch),
        )?;
    }
    closed_unchanged(
        "inlet",
        &inlet_lanes()?,
        |lane: &InletLane| plain(&lane.row.barrel),
        |batch| inlet_outputs::<Runtime>(&client, batch),
    )?;
    closed_unchanged(
        "outlet",
        &outlet_lanes()?,
        |lane: &OutletLane| plain(&lane.row.barrel),
        |batch| outlet_outputs::<Runtime>(&client, batch),
    )
}
