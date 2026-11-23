//! Dense discharge sweeps of every admitted code: rising headwater and continuity at both limits.

use super::coefficient_rows::{CODES, Shape};
use super::inlet_rows::{BARRELS, GRAVITY, RELATIVE_TOLERANCE, SLOPES};
use super::lanes::Lane;
use super::support::{TestResult, headwater};
use hydr::structures::InletRegime;
use hydr::structures::{CulvertBarrel, CulvertInlet, CulvertInletHeadwaterRow, CulvertInletRow};

/// `X` values of the rising sweep, each at least 0.05 from its neighbours.
const SWEEP: [f64; 24] = [
    0.02, 0.1, 0.25, 0.5, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.1, 3.3, 3.45, 3.55, 3.7, 3.85, 3.95,
    4.05, 4.3, 4.8, 5.5, 6.5, 8.0, 12.0,
];

/// F64 steps taken on each side of a limit discharge.
const STEPS: usize = 16;

/// Each admitted code on its barrel at each frozen slope, as `(row, Q at X = 1)`.
fn curves() -> Result<Vec<(CulvertInletRow, f64)>, Box<dyn std::error::Error>> {
    let mut made = Vec::new();
    for entry in CODES.iter().filter(|entry| entry.admitted) {
        let (circular, rise, span) = *BARRELS
            .get(usize::from(entry.shape == Shape::Box))
            .ok_or("no frozen barrel for the code shape")?;
        let rise = f64::from_bits(rise);
        let (barrel, area) = if circular {
            (
                CulvertBarrel::circular(rise)?,
                std::f64::consts::PI * rise * rise / 4.0,
            )
        } else {
            let span = f64::from_bits(span);
            (CulvertBarrel::rectangular_box(span, rise)?, span * rise)
        };
        // Q at X = 1 is A D^0.5 / Ku, with Ku^2 = 1 / 0.3048.
        let unit_discharge = area * (rise * 0.3048).sqrt();
        for slope in SLOPES {
            let row = CulvertInletRow {
                barrel,
                inlet: CulvertInlet::hds5(entry.code)?,
                slope: f64::from_bits(slope),
                gravity_m_per_s2: f64::from_bits(GRAVITY),
                face_blockage: hydr::structures::BlockageFraction::NONE,
            };
            made.push((row, unit_discharge));
        }
    }
    Ok(made)
}

/// One curve and its headwater rows, in sweep order.
type Curve = (CulvertInletRow, Vec<CulvertInletHeadwaterRow>);

/// Runs `discharges(unit)` for every curve in one batch and splits the results per curve.
fn sweep(discharges: impl Fn(f64) -> Vec<f64>) -> Result<Vec<Curve>, Box<dyn std::error::Error>> {
    let curves = curves()?;
    let mut lanes = Vec::new();
    for (row, unit) in &curves {
        lanes.extend(discharges(*unit).into_iter().map(|discharge| Lane {
            row: *row,
            discharge,
        }));
    }
    let found = headwater(&lanes)?;
    let per = found.len() / curves.len();
    Ok(curves
        .into_iter()
        .zip(found.chunks(per))
        .map(|((row, _), chunk)| (row, chunk.to_vec()))
        .collect())
}

fn name(row: &CulvertInletRow) -> String {
    format!("code {:?} slope {}", row.inlet.code(), row.slope)
}

#[test]
fn headwater_strictly_rises_with_discharge_across_every_band() -> TestResult {
    let results = sweep(|unit| SWEEP.iter().map(|x| x * unit).collect())?;
    for (row, curve) in &results {
        let regimes: Vec<InletRegime> = curve.iter().map(|r| r.regime).collect();
        for needed in [
            InletRegime::Unsubmerged,
            InletRegime::Transition,
            InletRegime::Submerged,
        ] {
            if !regimes.contains(&needed) {
                return Err(format!("{}: the sweep misses {needed:?}", name(row)).into());
            }
        }
        for (before, after) in curve.iter().zip(curve.iter().skip(1)) {
            if before.headwater_m.1 >= after.headwater_m.0 {
                return Err(format!(
                    "{}: {:?} then {:?} does not certifiably rise",
                    name(row),
                    before.headwater_m,
                    after.headwater_m
                )
                .into());
            }
        }
    }
    Ok(())
}

/// `STEPS` F64 steps on each side of `limit`, in increasing order.
fn around(limit: f64) -> Vec<f64> {
    let mut below = vec![limit];
    for _ in 0..STEPS {
        let next = below.last().map_or(limit, |q: &f64| q.next_down());
        below.push(next);
    }
    below.reverse();
    let mut above = below;
    for _ in 0..STEPS {
        let next = above.last().map_or(limit, |q: &f64| q.next_up());
        above.push(next);
    }
    above
}

#[test]
fn headwater_is_continuous_at_both_limits() -> TestResult {
    let unsubmerged = sweep(|unit| around(3.5 * unit))?;
    let submerged = sweep(|unit| around(4.0 * unit))?;
    let sides = [
        (
            3.5,
            &unsubmerged,
            [InletRegime::Unsubmerged, InletRegime::Transition],
        ),
        (
            4.0,
            &submerged,
            [InletRegime::Transition, InletRegime::Submerged],
        ),
    ];
    for (limit, results, [left, right]) in sides {
        for (row, curve) in results {
            let lowest = curve
                .iter()
                .map(|r| r.headwater_m.0)
                .fold(f64::INFINITY, f64::min);
            let highest = curve
                .iter()
                .map(|r| r.headwater_m.1)
                .fold(f64::NEG_INFINITY, f64::max);
            if highest - lowest > RELATIVE_TOLERANCE * highest.abs() {
                return Err(format!(
                    "{} near X = {limit}: headwater spans [{lowest:e}, {highest:e}]",
                    name(row)
                )
                .into());
            }
            let bands = curve.iter().map(|r| r.regime).collect::<Vec<_>>();
            if bands.first() != Some(&left) || bands.last() != Some(&right) {
                return Err(format!("{} near X = {limit}: bands {bands:?}", name(row)).into());
            }
            for (before, after) in curve.iter().zip(curve.iter().skip(1)) {
                if before.headwater_m.0 > after.headwater_m.1 {
                    return Err(
                        format!("{} near X = {limit}: certified reversal", name(row)).into(),
                    );
                }
            }
        }
    }
    Ok(())
}
