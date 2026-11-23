//! Floored rows beside rows without a floor: lane independence, unchanged bits and ranges.

use super::runtime::{Runtime, client};
use super::section_rows::SECTIONS;
use super::sections::{Outcome, floored, quantities, tensor};
use hydr::structures::{
    BarrelState, CulvertBarrel, CulvertBarrelBatch, CulvertBarrelGeometryRow, CulvertFloor,
    StructureError,
};

/// Every bound of a projected row, as bits.
fn bits(row: &CulvertBarrelGeometryRow) -> (BarrelState, Vec<u64>) {
    let pairs = [
        row.area_m2,
        row.top_width_m,
        row.wetted_perimeter_m,
        row.wall_perimeter_m,
        row.floor_width_m,
        row.weighted_perimeter_m,
        row.hydraulic_radius_m,
        row.hydrostatic_moment_m3,
    ];
    let values = pairs
        .iter()
        .flat_map(|(lower, upper)| [lower.to_bits(), upper.to_bits()])
        .collect();
    (row.state, values)
}

fn geometry(barrels: Vec<CulvertBarrel>, depths: &[f64]) -> Outcome<Vec<CulvertBarrelGeometryRow>> {
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    Ok(batch.geometry(&tensor(depths)?)?.project()?)
}

/// Barrels without a floor: a circle, a box and the base of every arc section.
fn unfloored() -> Outcome<Vec<CulvertBarrel>> {
    let mut barrels = vec![
        CulvertBarrel::circular(1.2)?,
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

/// Lanes of floored and unfloored barrels with depths across dry, free and full.
fn lanes() -> Outcome<Vec<(CulvertBarrel, f64)>> {
    let mut lanes = Vec::new();
    for (k, barrel) in unfloored()?.into_iter().enumerate() {
        let fraction = [0.0, 0.3, 0.7, 1.0, 1.5][k % 5];
        lanes.push((barrel, fraction * barrel.rise_m()));
    }
    for (k, entry) in SECTIONS.iter().enumerate() {
        let barrel = floored(entry)?;
        let fraction = [0.0, 2f64.powi(-30), 0.4, 0.999, 1.0, 1.5][k % 6];
        lanes.push((barrel, fraction * barrel.open_rise_m()));
    }
    // A floored circle far from unit scale runs in its frame.
    let tiny = 2f64.powi(-400);
    let scaled = CulvertBarrel::circular(1.2 * tiny)?.with_floor(CulvertFloor {
        sediment_depth_m: 0.3 * tiny,
        roughness_ratio: 2.5,
    })?;
    lanes.push((scaled, 0.5 * scaled.open_rise_m()));
    Ok(lanes)
}

#[test]
fn layouts_of_width_1_33_and_65_return_single_row_bits() -> Outcome<()> {
    let lanes = lanes()?;
    let single = lanes
        .iter()
        .map(|&(barrel, depth)| {
            let rows = geometry(vec![barrel], &[depth])?;
            rows.first()
                .map(bits)
                .ok_or_else(|| "one row expected".into())
        })
        .collect::<Outcome<Vec<_>>>()?;
    for width in [1usize, 33, 65] {
        let picks: Vec<usize> = (0..width).map(|k| (7 * k + width) % lanes.len()).collect();
        let barrels = picks
            .iter()
            .filter_map(|&k| lanes.get(k).map(|l| l.0))
            .collect();
        let depths: Vec<f64> = picks
            .iter()
            .filter_map(|&k| lanes.get(k).map(|l| l.1))
            .collect();
        let rows = geometry(barrels, &depths)?;
        for (lane, (&k, row)) in picks.iter().zip(&rows).enumerate() {
            assert_eq!(Some(&bits(row)), single.get(k), "width {width} lane {lane}");
        }
    }
    Ok(())
}

#[test]
fn rows_without_a_floor_keep_their_bits_beside_floored_rows() -> Outcome<()> {
    let barrels = unfloored()?;
    let fractions = [0.0, 2f64.powi(-30), 1e-6, 0.2, 0.5, 0.8, 0.999, 1.0, 1.5];
    let mut alone = (Vec::new(), Vec::new());
    for barrel in &barrels {
        for fraction in fractions {
            alone.0.push(*barrel);
            alone.1.push(fraction * barrel.rise_m());
        }
    }
    let before = geometry(alone.0.clone(), &alone.1)?;
    // The same rows with a floored row after each.
    let mut mixed = (Vec::new(), Vec::new());
    for (k, (barrel, depth)) in alone.0.iter().zip(&alone.1).enumerate() {
        let entry = SECTIONS.get(k % SECTIONS.len()).ok_or("no section")?;
        let bed = floored(entry)?;
        mixed.0.extend([*barrel, bed]);
        mixed.1.extend([*depth, 0.5 * bed.open_rise_m()]);
    }
    let after = geometry(mixed.0, &mixed.1)?;
    for (k, row) in before.iter().enumerate() {
        let beside = after.get(2 * k).ok_or("row missing")?;
        assert_eq!(bits(row), bits(beside), "row {k}");
        assert_eq!(row.floor_width_m, (0.0, 0.0), "row {k}");
        assert_eq!(row.wall_perimeter_m, row.wetted_perimeter_m, "row {k}");
        assert_eq!(row.weighted_perimeter_m, row.wetted_perimeter_m, "row {k}");
    }
    Ok(())
}

#[test]
fn ranges_hold_every_point_inside_them() -> Outcome<()> {
    let barrels = SECTIONS.iter().map(floored).collect::<Outcome<Vec<_>>>()?;
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels.clone())?;
    for (from, to) in [(0.0, 0.5), (0.1, 0.6), (0.4, 0.999)] {
        let lower: Vec<f64> = barrels.iter().map(|b| from * b.open_rise_m()).collect();
        let upper: Vec<f64> = barrels.iter().map(|b| to * b.open_rise_m()).collect();
        let range = batch
            .geometry_over(&tensor(&lower)?, &tensor(&upper)?)?
            .project()?;
        for step in 0..=8 {
            let t = f64::from(step) / 8.0;
            let depths: Vec<f64> = lower
                .iter()
                .zip(&upper)
                .map(|(low, high)| low + t * (high - low))
                .collect();
            let points = batch.geometry(&tensor(&depths)?)?.project()?;
            for ((entry, held), point) in SECTIONS.iter().zip(&range).zip(&points) {
                let pairs = quantities(held).into_iter().zip(quantities(point));
                for (quantity, (outer, inner)) in pairs.enumerate() {
                    assert!(
                        outer.0 <= inner.0 && inner.1 <= outer.1,
                        "{} [{from}, {to}] at {t}: quantity {quantity} {inner:?} outside {outer:?}",
                        entry.name
                    );
                }
                assert_eq!(held.floor_width_m, point.floor_width_m, "{}", entry.name);
            }
        }
    }
    // A range that reaches the open rise rejects.
    let crown: Vec<f64> = barrels.iter().map(|b| b.open_rise_m()).collect();
    let zero = vec![0.0; barrels.len()];
    let result = batch.geometry_over(&tensor(&zero)?, &tensor(&crown)?);
    assert!(
        matches!(
            &result,
            Err(StructureError::InvalidInput { context, .. })
                if context.contains("open_rise_m=") && context.contains("the range reaches the crown")
        ),
        "{:?}",
        result.err()
    );
    Ok(())
}
