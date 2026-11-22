//! Arc barrels scaled by exact powers of two answer as the unit-scale barrel does, scaled by
//! the same power of two.
//!
//! Lengths scale by `2^k`, areas by `2^2k`, moments by `2^3k`, critical-depth discharges by
//! `2^(5k/2)` and normal-depth discharges by `2^(8k/3)`.
//!
//! - Construction and geometry: bit for bit, each value outside the F64 range rounded outward.
//! - Depth, inlet and outlet queries: the scaled enclosure meets the unit enclosure scaled. Their
//!   targets take logs of metre values, whose rounding differs with the scale.
//! - At `2^-900` every positive discharge of the unit barrel falls below F64, so depth queries
//!   there check the typed outcomes of zero and of a represented discharge. At `2^300` the
//!   outlet squares a discharge above `2^1024`, so outlet rows run at `2^±60`.

use super::runtime::{Runtime, client};
use hydr::structures::{
    CriticalDepthStatus, CulvertBarrel, CulvertBarrelBatch, CulvertBarrelGeometryRow, CulvertInlet,
    CulvertInletBatch, CulvertInletRow, CulvertOutletBatch, CulvertOutletRow, NormalDepthStatus,
    StructureError, StructureInputErrorKind,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;
type TestResult = Outcome<()>;

/// Powers of two the barrels are scaled by.
const SCALES: [i32; 3] = [-900, 0, 300];

/// Depth fractions of the geometry rows.
const FRACTIONS: [f64; 9] = [
    0.0,
    9.5367431640625e-7,
    0.01,
    0.2,
    0.5,
    0.8,
    0.999,
    1.0,
    1.5,
];

/// One evidence barrel per arc shape at unit scale, as constructor inputs, with an inlet code.
const BARRELS: [(&str, &[f64], u8); 4] = [
    ("horizontal", &[1.524, 0.9652, 0.419_475_92], 30),
    ("vertical", &[0.9652, 1.524, 0.419_475_92], 34),
    ("pipe-arch", &[1.8542, 1.397, 0.4572, 1.938_528], 37),
    ("arch", &[3.048, 1.6002], 45),
];

fn build(name: &str, inputs: &[f64], power: i32) -> Outcome<CulvertBarrel> {
    let v: Vec<f64> = inputs.iter().map(|x| times(*x, power, 1)).collect();
    Ok(match (name, v.as_slice()) {
        ("horizontal", &[b, d, r]) => CulvertBarrel::horizontal_ellipse(b, d, r)?,
        ("vertical", &[b, d, r]) => CulvertBarrel::vertical_ellipse(b, d, r)?,
        ("pipe-arch", &[b, d, c, r]) => CulvertBarrel::pipe_arch(b, d, c, r)?,
        ("arch", &[b, d]) => CulvertBarrel::arch(b, d)?,
        _ => return Err(format!("{name}: wrong input count").into()),
    })
}

/// `value 2^(power steps)`, multiplied one step at a time, each step exact unless it underflows.
fn times(value: f64, power: i32, steps: i32) -> f64 {
    (0..steps).fold(value, |v, _| v * 2f64.powi(power))
}

/// The largest F64 at or below `value.0 2^(power steps)`, and the smallest at or above
/// `value.1 2^(power steps)`.
fn image(value: (f64, f64), power: i32, steps: i32) -> (f64, f64) {
    let back = |v: f64| times(v, -power, steps);
    let mut lower = times(value.0, power, steps);
    while back(lower) > value.0 {
        lower = lower.next_down();
    }
    let mut upper = times(value.1, power, steps);
    while back(upper) < value.1 {
        upper = upper.next_up();
    }
    (lower, upper)
}

fn tensor(values: &[f64]) -> Outcome<Tensor<Runtime>> {
    Ok(Tensor::<Runtime>::from_slice(
        values,
        &[values.len()],
        client().device(),
    )?)
}

fn batch(barrel: CulvertBarrel, rows: usize) -> Outcome<CulvertBarrelBatch<Runtime>> {
    Ok(CulvertBarrelBatch::<Runtime>::new(
        client(),
        vec![barrel; rows],
    )?)
}

fn geometry(barrel: CulvertBarrel, depths: &[f64]) -> Outcome<Vec<CulvertBarrelGeometryRow>> {
    Ok(batch(barrel, depths.len())?
        .geometry(&tensor(depths)?)?
        .project()?)
}

#[test]
fn scaled_arc_barrels_answer_geometry_as_the_unit_barrel_scaled() -> TestResult {
    for (name, inputs, _) in BARRELS {
        let unit = build(name, inputs, 0)?;
        let depths: Vec<f64> = FRACTIONS.iter().map(|f| f * unit.rise_m()).collect();
        let reference = geometry(unit, &depths)?;
        let full = batch(unit, 1)?.full_geometry()?.project()?;
        for power in SCALES {
            let barrel = build(name, inputs, power)?;
            let scaled: Vec<f64> = depths.iter().map(|y| times(*y, power, 1)).collect();
            let rows = geometry(barrel, &scaled)?;
            let crowns = batch(barrel, 1)?.full_geometry()?.project()?;
            let pairs = rows.iter().zip(&reference).chain(crowns.iter().zip(&full));
            for (index, (row, expected)) in pairs.enumerate() {
                let what = format!("{name} 2^{power} row {index}: {row:?} against {expected:?}");
                assert_eq!(row.state, expected.state, "{what}");
                let quantities = [
                    (row.area_m2, expected.area_m2, 2),
                    (row.top_width_m, expected.top_width_m, 1),
                    (row.wetted_perimeter_m, expected.wetted_perimeter_m, 1),
                    (row.hydraulic_radius_m, expected.hydraulic_radius_m, 1),
                    (row.hydrostatic_moment_m3, expected.hydrostatic_moment_m3, 3),
                ];
                for (found, unit_value, steps) in quantities {
                    assert_eq!(found, image(unit_value, power, steps), "{what}");
                }
            }
        }
    }
    Ok(())
}

/// Relative width a scaled enclosure may reach. The unit enclosures are far tighter. The
/// targets add logs of metre values near `2^750`, whose rounding widens them.
const SCALED_WIDTH: f64 = 1e-9;

/// Whether `found` meets the enclosure `expected` scaled by `2^power` and stays within
/// `SCALED_WIDTH` of it. Two absent enclosures agree.
fn meets(found: Option<(f64, f64)>, expected: Option<(f64, f64)>, power: i32) -> bool {
    match (found, expected) {
        (None, None) => true,
        (Some((lower, upper)), Some(value)) => {
            let (low, high) = image(value, power, 1);
            let reach = (high - lower).max(upper - low);
            lower <= high && low <= upper && reach <= SCALED_WIDTH * high.abs().max(low.abs())
        }
        _ => false,
    }
}

#[test]
fn scaled_arc_barrels_answer_depths_as_the_unit_barrel_scaled() -> TestResult {
    for (name, inputs, _) in BARRELS {
        let unit = build(name, inputs, 0)?;
        let flows = [0.0, 0.05, 0.6, 2.5];
        let unit_batch = batch(unit, flows.len())?;
        let critical = unit_batch
            .critical_depth(&tensor(&flows)?, 9.81)?
            .project()?;
        let n = tensor(&[0.024; 4])?;
        let slope = tensor(&[1e-3; 4])?;
        let normal = unit_batch
            .normal_depth(&tensor(&flows)?, &n, &slope)?
            .project()?;
        // 2^300: critical discharges scale by 2^750 and normal discharges by 2^800.
        let large = batch(build(name, inputs, 300)?, flows.len())?;
        let q: Vec<f64> = flows.iter().map(|q| times(*q, 750, 1)).collect();
        let found = large.critical_depth(&tensor(&q)?, 9.81)?.project()?;
        for (row, expected) in found.iter().zip(&critical) {
            assert_eq!(row.status, expected.status, "{name}: {row:?}");
            assert!(
                meets(row.depth_m, expected.depth_m, 300),
                "{name}: {row:?} {expected:?}"
            );
        }
        let q: Vec<f64> = flows.iter().map(|q| times(*q, 800, 1)).collect();
        let found = large.normal_depth(&tensor(&q)?, &n, &slope)?.project()?;
        for (row, expected) in found.iter().zip(&normal) {
            assert_eq!(row.status, expected.status, "{name}: {row:?}");
            assert!(
                meets(row.depth_m, expected.depth_m, 300),
                "{name}: {row:?} {expected:?}"
            );
        }
        // 2^-900: zero discharge is dry, and a represented discharge lies above the crown.
        let tiny = build(name, inputs, -900)?;
        let small = batch(tiny, 2)?;
        let found = small
            .critical_depth(&tensor(&[0.0, 1e-300])?, 9.81)?
            .project()?;
        assert_eq!(found[0].status, CriticalDepthStatus::Dry, "{name}");
        assert_eq!(found[1].status, CriticalDepthStatus::AboveCrown, "{name}");
        assert!(
            found[1]
                .depth_m
                .is_some_and(|(lower, upper)| 0.0 < lower && upper == tiny.rise_m()),
            "{name}: {found:?}"
        );
        let found = small
            .normal_depth(
                &tensor(&[0.0, 1e-300])?,
                &tensor(&[0.024; 2])?,
                &tensor(&[1e-3; 2])?,
            )?
            .project()?;
        assert_eq!(found[0].status, NormalDepthStatus::Dry, "{name}");
        assert_eq!(found[1].status, NormalDepthStatus::AboveCapacity, "{name}");
    }
    Ok(())
}

#[test]
fn scaled_arc_barrels_answer_inlet_control_as_the_unit_barrel_scaled() -> TestResult {
    // X in 0.5, 3.75 and 6 at unit scale: unsubmerged, transition and submerged.
    for (name, inputs, code) in BARRELS {
        let headwater = |power: i32, flows: &[f64]| -> Outcome<Vec<(f64, f64)>> {
            let row = CulvertInletRow {
                barrel: build(name, inputs, power)?,
                inlet: CulvertInlet::hds5(code)?,
                slope: 0.01,
                gravity_m_per_s2: 9.81,
            };
            let batch = CulvertInletBatch::<Runtime>::new(client(), vec![row; flows.len()])?;
            let rows = batch.headwater(&tensor(flows)?)?.project()?;
            Ok(rows.iter().map(|row| row.headwater_m).collect())
        };
        let flows = [0.3, 2.5, 4.0];
        let reference = headwater(0, &flows)?;
        let q: Vec<f64> = flows.iter().map(|q| times(*q, 750, 1)).collect();
        for (found, expected) in headwater(300, &q)?.iter().zip(&reference) {
            assert!(
                meets(Some(*found), Some(*expected), 300),
                "{name}: {found:?} {expected:?}"
            );
        }
        // At 2^-900 the discharge at X = 3.5 falls below F64, and admission says so.
        let row = CulvertInletRow {
            barrel: build(name, inputs, -900)?,
            inlet: CulvertInlet::hds5(code)?,
            slope: 0.01,
            gravity_m_per_s2: 9.81,
        };
        match CulvertInletBatch::<Runtime>::new(client(), vec![row]) {
            Err(StructureError::InvalidInput {
                kind: StructureInputErrorKind::Definition,
                context,
                element: Some(0),
            }) => assert!(
                context.contains("lies below the normal F64 range"),
                "{context}"
            ),
            other => return Err(format!("{name}: expected a rejection, found {other:?}").into()),
        }
    }
    Ok(())
}

#[test]
fn scaled_arc_barrels_answer_outlet_control_as_the_unit_barrel_scaled() -> TestResult {
    // Lengths scale by 2^60, discharges by 2^150 and Manning n by 2^10, so every friction slope
    // and Froude number stays the same.
    for (name, inputs, _) in BARRELS {
        let outlet = |power: i32, flows: &[f64], tails: &[f64]| -> Outcome<Vec<(f64, f64)>> {
            let row = CulvertOutletRow {
                barrel: build(name, inputs, power)?,
                length_m: times(40.0, power, 1),
                manning_n: times(0.013, power / 6, 1),
                entrance_loss: 0.5,
                slope: 0.002,
                gravity_m_per_s2: 9.81,
            };
            let batch = CulvertOutletBatch::<Runtime>::new(client(), vec![row; flows.len()])?;
            let rows = batch
                .headwater(&tensor(flows)?, &tensor(tails)?)?
                .project()?;
            Ok(rows.iter().map(|row| row.headwater_m).collect())
        };
        let unit = build(name, inputs, 0)?;
        let flows = [0.4, 1.5, 3.0];
        let tails = [0.0, 0.6 * unit.rise_m(), 1.3 * unit.rise_m()];
        let reference = outlet(0, &flows, &tails)?;
        for power in [-60, 60] {
            let q: Vec<f64> = flows.iter().map(|q| times(*q, power * 5 / 2, 1)).collect();
            let tw: Vec<f64> = tails.iter().map(|t| times(*t, power, 1)).collect();
            for (found, expected) in outlet(power, &q, &tw)?.iter().zip(&reference) {
                assert!(
                    meets(Some(*found), Some(*expected), power),
                    "{name} 2^{power}: {found:?} {expected:?}"
                );
            }
        }
        // At 2^-900 and 2^300 the march products leave F64 and the query reports it.
        for power in [-900, 300] {
            let flows = [1e-300, times(1.0, power * 5 / 2, 1).max(1e-300)];
            match outlet(power, &flows, &[0.0; 2]) {
                Err(error) => assert!(
                    matches!(
                        error.downcast_ref::<StructureError>(),
                        Some(StructureError::Numerical { .. })
                    ),
                    "{name} 2^{power}: {error}"
                ),
                Ok(rows) => return Err(format!("{name} 2^{power}: {rows:?}").into()),
            }
        }
    }
    Ok(())
}
