//! Critical and normal depth on floored rows beside rows without a floor: lane independence,
//! unchanged bits and monotone sweeps.

use super::runtime::{Runtime, client};
use super::section_rows::SECTIONS;
use super::sections::{Outcome, floored, tensor};
use hydr::structures::{
    BarrelDepthStatus, CulvertBarrel, CulvertBarrelBatch, CulvertBarrelDepthRow, CulvertFloor,
};

/// One lane: a barrel with its discharge, Manning `n` and slope.
type Lane = (CulvertBarrel, f64, f64, f64);

/// Status and endpoint bits of one projected depth row.
fn bits<S: BarrelDepthStatus>(row: &CulvertBarrelDepthRow<S>) -> (S, Option<(u64, u64)>) {
    (
        row.status,
        row.depth_m
            .map(|(lower, upper)| (lower.to_bits(), upper.to_bits())),
    )
}

/// Critical and normal depth bits of every lane, in one batch.
type DepthBits = (
    Vec<(hydr::structures::CriticalDepthStatus, Option<(u64, u64)>)>,
    Vec<(hydr::structures::NormalDepthStatus, Option<(u64, u64)>)>,
);

fn depths(lanes: &[Lane]) -> Outcome<DepthBits> {
    let barrels = lanes.iter().map(|lane| lane.0).collect();
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    let column = |f: fn(&Lane) -> f64| -> Vec<f64> { lanes.iter().map(f).collect() };
    let flows = tensor(&column(|lane| lane.1))?;
    let critical = batch.critical_depth(&flows, 9.81)?.project()?;
    let normal = batch
        .normal_depth(
            &flows,
            &tensor(&column(|lane| lane.2))?,
            &tensor(&column(|lane| lane.3))?,
        )?
        .project()?;
    Ok((
        critical.iter().map(bits).collect(),
        normal.iter().map(bits).collect(),
    ))
}

/// A discharge that puts the critical depth inside the open section, scaled by `factor`.
fn flow(barrel: &CulvertBarrel, factor: f64) -> f64 {
    factor * 9.81f64.sqrt() * barrel.span_m() * barrel.open_rise_m().powf(1.5)
}

/// Barrels without a floor, a unit rise among them.
fn unfloored() -> Outcome<Vec<CulvertBarrel>> {
    let mut barrels = vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::circular(0.3)?,
        CulvertBarrel::rectangular_box(2.0, 1.0)?,
        CulvertBarrel::rectangular_box(2.0, 1.5)?,
    ];
    for name in [
        "horizontal-ellipse-0.5",
        "vertical-ellipse-0.1",
        "pipe-arch-0.2",
        "arch-0.25",
    ] {
        let entry = SECTIONS
            .iter()
            .find(|s| s.name == name)
            .ok_or("no such section")?;
        barrels.push(floored(entry)?.base());
    }
    Ok(barrels)
}

/// Lanes of floored and unfloored barrels across dry, free, near-crown and adverse rows.
fn lanes() -> Outcome<Vec<Lane>> {
    let factors = [0.0, 0.05, 0.3, 0.7, 2.0];
    let slopes = [0.01, 0.001, 0.02, 0.0, 0.005];
    let mut lanes = Vec::new();
    let mut barrels = unfloored()?;
    for entry in SECTIONS {
        barrels.push(floored(entry)?);
    }
    // A floored circle far from unit scale runs in its frame.
    let tiny = 2f64.powi(-400);
    barrels.push(
        CulvertBarrel::circular(1.2 * tiny)?.with_floor(CulvertFloor {
            sediment_depth_m: 0.3 * tiny,
            roughness_ratio: 2.5,
        })?,
    );
    for (k, barrel) in barrels.into_iter().enumerate() {
        let factor = factors.get(k % factors.len()).copied().unwrap_or(0.3);
        let slope = slopes.get(k % slopes.len()).copied().unwrap_or(0.01);
        lanes.push((barrel, flow(&barrel, factor), 0.012, slope));
    }
    Ok(lanes)
}

#[test]
fn depth_layouts_of_width_1_33_and_65_return_single_row_bits() -> Outcome<()> {
    let lanes = lanes()?;
    let single = lanes
        .iter()
        .map(|lane| {
            let (critical, normal) = depths(std::slice::from_ref(lane))?;
            match (critical.first(), normal.first()) {
                (Some(c), Some(n)) => Ok((*c, *n)),
                _ => Err("one row expected".into()),
            }
        })
        .collect::<Outcome<Vec<_>>>()?;
    for width in [1usize, 33, 65] {
        let picks: Vec<usize> = (0..width).map(|k| (7 * k + width) % lanes.len()).collect();
        let picked: Vec<Lane> = picks
            .iter()
            .filter_map(|&k| lanes.get(k).copied())
            .collect();
        let (critical, normal) = depths(&picked)?;
        for (lane, (&k, row)) in picks.iter().zip(critical.iter().zip(&normal)).enumerate() {
            let expected = single.get(k).map(|(c, n)| (c, n));
            assert_eq!(Some((row.0, row.1)), expected, "width {width} lane {lane}");
        }
    }
    Ok(())
}

#[test]
fn rows_without_a_floor_keep_their_depth_bits_beside_floored_rows() -> Outcome<()> {
    let mut alone = Vec::new();
    for barrel in unfloored()? {
        for (factor, slope) in [
            (0.0, 0.01),
            (1e-6, 0.01),
            (0.1, 0.001),
            (0.5, 0.02),
            (3.0, 0.0),
        ] {
            alone.push((barrel, flow(&barrel, factor), 0.013, slope));
        }
    }
    let before = depths(&alone)?;
    let mut mixed = Vec::new();
    for (k, lane) in alone.iter().enumerate() {
        let entry = SECTIONS.get(k % SECTIONS.len()).ok_or("no section")?;
        let bed = floored(entry)?;
        mixed.extend([*lane, (bed, flow(&bed, 0.3), 0.012, 0.01)]);
    }
    let after = depths(&mixed)?;
    assert_eq!(every_other(&after.0), before.0);
    assert_eq!(every_other(&after.1), before.1);
    Ok(())
}

/// Rows `0, 2, 4, ...` of `rows`.
fn every_other<T: Copy>(rows: &[T]) -> Vec<T> {
    rows.iter().step_by(2).copied().collect()
}

/// `true` when no later enclosure lies wholly below an earlier one.
fn rising<S: BarrelDepthStatus>(rows: &[CulvertBarrelDepthRow<S>]) -> bool {
    let known: Vec<(f64, f64)> = rows.iter().filter_map(|row| row.depth_m).collect();
    known
        .iter()
        .enumerate()
        .all(|(i, (lower, _))| known.iter().skip(i + 1).all(|(_, upper)| lower <= upper))
}

#[test]
fn floored_depths_never_reverse_with_discharge_roughness_or_slope() -> Outcome<()> {
    let names = [
        "circle-0.2",
        "circle-0.95",
        "circle-0.2-smooth",
        "box-0.3",
        "vertical-ellipse-0.1",
        "pipe-arch-0.2",
        "arch-0",
    ];
    for name in names {
        let entry = SECTIONS
            .iter()
            .find(|s| s.name == name)
            .ok_or("no such section")?;
        let barrel = floored(entry)?;
        let steps: Vec<f64> = (0..33)
            .map(|k| 2f64.powf(f64::from(k) / 4.0 - 6.0))
            .collect();
        let base = flow(&barrel, 0.1);
        let flows: Vec<f64> = steps.iter().map(|f| f * base).collect();
        let batch = CulvertBarrelBatch::<Runtime>::new(client(), vec![barrel; steps.len()])?;
        let critical = batch.critical_depth(&tensor(&flows)?, 9.81)?.project()?;
        assert!(rising(&critical), "{name}: critical in Q");
        let fixed = |value: f64| vec![value; steps.len()];
        let normal = batch
            .normal_depth(
                &tensor(&flows)?,
                &tensor(&fixed(0.012))?,
                &tensor(&fixed(0.01))?,
            )?
            .project()?;
        assert!(rising(&normal), "{name}: normal in Q");
        let roughness: Vec<f64> = steps.iter().map(|f| 0.004 * f.sqrt()).collect();
        let flat = fixed(base);
        let normal = batch
            .normal_depth(
                &tensor(&flat)?,
                &tensor(&roughness)?,
                &tensor(&fixed(0.01))?,
            )?
            .project()?;
        assert!(rising(&normal), "{name}: normal in n");
        // Depth falls as the slope rises, so the reversed sweep rises.
        let slopes: Vec<f64> = steps.iter().rev().map(|f| 0.05 * f).collect();
        let normal = batch
            .normal_depth(&tensor(&flat)?, &tensor(&fixed(0.012))?, &tensor(&slopes)?)?
            .project()?;
        assert!(rising(&normal), "{name}: normal in S");
    }
    Ok(())
}
