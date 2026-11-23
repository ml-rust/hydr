//! Open sections above a floor against the frozen references, with their identities.

use super::runtime::{Runtime, client};
use super::section_rows::{LAYOUT, ROWS, Row, SECTIONS, Section, Shape, State};
use hydr::structures::{
    BarrelState, CulvertBarrel, CulvertBarrelBatch, CulvertBarrelGeometryRow, CulvertFloor,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

pub type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// Quantity names in fixture order.
const QUANTITIES: [&str; 7] = [
    "area",
    "top_width",
    "wall_perimeter",
    "perimeter",
    "weighted_perimeter",
    "radius",
    "moment",
];

pub fn tensor(values: &[f64]) -> Result<Tensor<Runtime>, numr::error::Error> {
    Tensor::from_slice(values, &[values.len()], client().device())
}

/// The floored barrel of one evidence section.
pub fn floored(section: &Section) -> Outcome<CulvertBarrel> {
    let inputs: Vec<f64> = section.inputs.iter().map(|b| f64::from_bits(*b)).collect();
    let barrel = match (section.shape, inputs.as_slice()) {
        (Shape::Circular, &[d]) => CulvertBarrel::circular(d)?,
        (Shape::Box, &[b, d]) => CulvertBarrel::rectangular_box(b, d)?,
        (Shape::HorizontalEllipse, &[b, d, r]) => CulvertBarrel::horizontal_ellipse(b, d, r)?,
        (Shape::VerticalEllipse, &[b, d, r]) => CulvertBarrel::vertical_ellipse(b, d, r)?,
        (Shape::PipeArch, &[b, d, c, r]) => CulvertBarrel::pipe_arch(b, d, c, r)?,
        (Shape::Arch, &[b, d]) => CulvertBarrel::arch(b, d)?,
        _ => return Err(format!("section {}: wrong input count", section.name).into()),
    };
    Ok(barrel.with_floor(CulvertFloor {
        sediment_depth_m: f64::from_bits(section.sediment_depth),
        roughness_ratio: f64::from_bits(section.roughness_ratio),
    })?)
}

fn section(index: usize) -> Outcome<&'static Section> {
    Ok(SECTIONS.get(index).ok_or("no such section")?)
}

/// `(lower, upper)` of each quantity in fixture order.
pub fn quantities(row: &CulvertBarrelGeometryRow) -> [(f64, f64); 7] {
    [
        row.area_m2,
        row.top_width_m,
        row.wall_perimeter_m,
        row.wetted_perimeter_m,
        row.weighted_perimeter_m,
        row.hydraulic_radius_m,
        row.hydrostatic_moment_m3,
    ]
}

fn state(value: State) -> BarrelState {
    match value {
        State::Dry => BarrelState::Dry,
        State::Free => BarrelState::Free,
        State::Full => BarrelState::Full,
    }
}

/// Point geometry of every reference row in one batch, so all shapes interleave.
fn reference_geometry() -> Outcome<Vec<CulvertBarrelGeometryRow>> {
    let barrels = ROWS
        .iter()
        .map(|row| floored(section(row.section)?))
        .collect::<Outcome<Vec<_>>>()?;
    let depths: Vec<f64> = ROWS.iter().map(|row| f64::from_bits(row.depth)).collect();
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    Ok(batch.geometry(&tensor(&depths)?)?.project()?)
}

/// Every failure of `row` against its reference, as text.
fn failures(reference: &Row, row: &CulvertBarrelGeometryRow) -> Vec<String> {
    let name = SECTIONS.get(reference.section).map_or("?", |s| s.name);
    let fraction = reference.fraction;
    let mut found = Vec::new();
    if row.state != state(reference.state) {
        found.push(format!("{name} at {fraction}: state {:?}", row.state));
    }
    let pairs = quantities(row)
        .into_iter()
        .zip(reference.values.iter().zip(reference.relative_width));
    for ((bounds, (exact, allowance)), quantity) in pairs.zip(QUANTITIES) {
        let [down, up] = exact.map(f64::from_bits);
        if !(bounds.0 <= down && up <= bounds.1) {
            found.push(format!(
                "{name} at {fraction}: {quantity} {bounds:?} misses [{down:e}, {up:e}]"
            ));
        }
        let width = bounds.1 - bounds.0;
        let scale = down.abs().max(up.abs());
        if width > allowance * scale {
            found.push(format!(
                "{name} at {fraction}: {quantity} width {:e} exceeds {allowance:e}",
                width / scale
            ));
        }
    }
    found
}

#[test]
fn every_section_row_lies_inside_its_enclosure_within_the_frozen_width() -> Outcome<()> {
    let rows = reference_geometry()?;
    assert_eq!(rows.len(), ROWS.len());
    let found: Vec<String> = ROWS
        .iter()
        .zip(&rows)
        .flat_map(|(reference, row)| failures(reference, row))
        .collect();
    assert!(found.is_empty(), "{}", found.join("\n"));
    Ok(())
}

#[test]
fn admitted_floors_keep_the_frozen_bed_and_open_rise() -> Outcome<()> {
    for entry in SECTIONS {
        let barrel = floored(entry)?;
        let floor = barrel.floor().ok_or("the floor was lost")?;
        assert_eq!(
            floor.sediment_depth_m.to_bits(),
            entry.sediment_depth,
            "{}",
            entry.name
        );
        assert_eq!(
            barrel.open_rise_m().to_bits(),
            entry.reduced_rise,
            "{}",
            entry.name
        );
        assert_eq!(
            floor.roughness_ratio.to_bits(),
            entry.roughness_ratio,
            "{}",
            entry.name
        );
    }
    Ok(())
}

/// One frozen section value inside its enclosure within the layout allowance.
fn layout_failure(
    index: usize,
    name: &str,
    found: (f64, f64),
    reduced_rise: f64,
) -> Option<String> {
    let value = LAYOUT
        .iter()
        .find(|value| value.section == index && value.name == name)?;
    let [down, up] = value.bounds.map(f64::from_bits);
    let scale = down.abs().max(up.abs()).max(reduced_rise.powi(value.power));
    let allowed = value.relative_width * scale;
    let holds = found.0 <= down && up <= found.1 && found.1 - found.0 <= allowed;
    (!holds).then(|| {
        let section = SECTIONS.get(index).map_or("?", |s| s.name);
        format!("{section} {name}: {found:?} against [{down:e}, {up:e}], allowance {allowed:e}")
    })
}

#[test]
fn bed_and_full_state_hold_the_frozen_section_values() -> Outcome<()> {
    let barrels = SECTIONS.iter().map(floored).collect::<Outcome<Vec<_>>>()?;
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels.clone())?;
    let full = batch.full_geometry()?.project()?;
    let dry = batch
        .geometry(&tensor(&vec![0.0; SECTIONS.len()])?)?
        .project()?;
    let mut failed = Vec::new();
    for (index, ((barrel, crown), bed)) in barrels.iter().zip(&full).zip(&dry).enumerate() {
        let reduced = barrel.open_rise_m();
        let checks = [
            ("bed_width", bed.floor_width_m),
            ("weighted_bed_width", bed.weighted_perimeter_m),
            ("full_area", crown.area_m2),
            ("full_wall_perimeter", crown.wall_perimeter_m),
            ("full_perimeter", crown.wetted_perimeter_m),
            ("full_weighted_perimeter", crown.weighted_perimeter_m),
            ("full_radius", crown.hydraulic_radius_m),
            ("full_moment", crown.hydrostatic_moment_m3),
        ];
        failed.extend(
            checks
                .into_iter()
                .filter_map(|(name, found)| layout_failure(index, name, found, reduced)),
        );
        assert_eq!(crown.state, BarrelState::Full, "{}", barrel.open_rise_m());
        assert_eq!(crown.top_width_m, (0.0, 0.0));
        assert_eq!(crown.floor_width_m, bed.floor_width_m);
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    Ok(())
}

#[test]
fn bed_rows_are_exact_zeros_with_the_bed_wetted() -> Outcome<()> {
    let rows = reference_geometry()?;
    for (reference, row) in ROWS.iter().zip(&rows) {
        if reference.state != State::Dry {
            continue;
        }
        let name = SECTIONS.get(reference.section).map_or("?", |s| s.name);
        assert_eq!(row.state, BarrelState::Dry, "{name}");
        for value in [
            row.area_m2,
            row.wall_perimeter_m,
            row.hydraulic_radius_m,
            row.hydrostatic_moment_m3,
        ] {
            assert_eq!(value, (0.0, 0.0), "{name}");
        }
        assert_eq!(row.top_width_m, row.floor_width_m, "{name}");
        assert_eq!(row.wetted_perimeter_m, row.floor_width_m, "{name}");
        assert!(row.floor_width_m.0 > 0.0, "{name}");
    }
    Ok(())
}

/// `true` when `[a.0 + b.0, a.1 + b.1]`, widened one ulp each way, meets `c`.
fn sum_meets(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    (a.0 + b.0).next_down() <= c.1 && c.0 <= (a.1 + b.1).next_up()
}

#[test]
fn perimeter_and_weighted_perimeter_add_the_bed_to_the_walls() -> Outcome<()> {
    let rows = reference_geometry()?;
    for (reference, row) in ROWS.iter().zip(&rows) {
        let entry = section(reference.section)?;
        let ratio = f64::from_bits(entry.roughness_ratio);
        // rho^(3/2) between its two roundings.
        let weight = (
            (ratio * ratio.sqrt()).next_down(),
            (ratio * ratio.sqrt()).next_up(),
        );
        let bed = row.floor_width_m;
        let weighted_bed = ((weight.0 * bed.0).next_down(), (weight.1 * bed.1).next_up());
        let what = format!("{} at {}", entry.name, reference.fraction);
        assert!(
            sum_meets(row.wall_perimeter_m, bed, row.wetted_perimeter_m),
            "{what}: P"
        );
        assert!(
            sum_meets(row.wall_perimeter_m, weighted_bed, row.weighted_perimeter_m),
            "{what}: P_n"
        );
        let (area, perimeter) = (row.area_m2, row.wetted_perimeter_m);
        let lower = (area.0 / perimeter.1).next_down();
        let upper = (area.1 / perimeter.0).next_up();
        let radius = row.hydraulic_radius_m;
        assert!(lower <= radius.1 && radius.0 <= upper, "{what}: R");
    }
    Ok(())
}

#[test]
fn dense_sweeps_hold_no_certified_reversal() -> Outcome<()> {
    const STEPS: usize = 257;
    let mut barrels = Vec::new();
    let mut depths = Vec::new();
    for entry in SECTIONS {
        let barrel = floored(entry)?;
        let top = barrel.open_rise_m();
        for step in 0..STEPS {
            barrels.push(barrel);
            depths.push(top * 1.25 * step as f64 / (STEPS - 1) as f64);
        }
    }
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    let rows = batch.geometry(&tensor(&depths)?)?.project()?;
    for (index, entry) in SECTIONS.iter().enumerate() {
        let sweep = rows
            .get(index * STEPS..(index + 1) * STEPS)
            .ok_or("a sweep is missing")?;
        for pair in sweep.windows(2) {
            let [low, high] = pair else { continue };
            let rising = [
                (low.area_m2, high.area_m2),
                (low.wall_perimeter_m, high.wall_perimeter_m),
                (low.weighted_perimeter_m, high.weighted_perimeter_m),
                (low.hydrostatic_moment_m3, high.hydrostatic_moment_m3),
            ];
            for (quantity, (below, above)) in rising.into_iter().enumerate() {
                assert!(
                    above.1 >= below.0,
                    "{}: quantity {quantity} falls from {below:?} to {above:?}",
                    entry.name
                );
            }
        }
    }
    Ok(())
}
