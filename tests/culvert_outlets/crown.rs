//! A steep box whose critical depth reaches the crown: discharge and tailwater sweeps through
//! its critical capacity, the outlet turning full, and the discharge where the barrel primes.

use super::lanes::Lane;
use super::support::{TestResult, headwater};
use super::sweep::{WIDTH, continuous, resolved, rising, spread, transition};
use hydr::structures::{CulvertBarrel, CulvertOutletHeadwaterRow, CulvertOutletRow, OutletRegime};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

const SPAN: f64 = 2.0;
const RISE: f64 = 1.5;
const LENGTH: f64 = 40.0;
const MANNING: f64 = 0.013;
const SLOPE: f64 = 0.02;

/// Ulps on each side of the priming discharge. Rows next to it are ties, whose headwater is
/// the hull of the crown-critical and the full-barrel value.
const NEIGHBOURS: u64 = 256;

/// Single-row bisection steps that close the priming bracket to adjacent discharges.
const STEPS: usize = 64;

/// A 2 m by 1.5 m box at slope 0.02. Its critical capacity is near 11.5 m^3/s.
fn steep_box(discharge: f64, tailwater: f64) -> Outcome<Lane> {
    Ok(Lane {
        row: CulvertOutletRow {
            barrel: CulvertBarrel::rectangular_box(SPAN, RISE)?,
            length_m: LENGTH,
            manning_n: MANNING,
            entrance_loss: 0.5,
            slope: SLOPE,
            gravity_m_per_s2: 9.81,
        },
        discharge,
        tailwater,
    })
}

fn one(lane: Lane) -> Outcome<CulvertOutletHeadwaterRow> {
    let rows = headwater(&[lane])?;
    let row = rows.first().copied().ok_or("no headwater row")?;
    resolved(
        &rows,
        &format!("Q={} TW={}", lane.discharge, lane.tailwater),
    )?;
    Ok(row)
}

#[test]
fn steep_box_resolves_and_never_falls_through_its_critical_capacity() -> TestResult {
    let mut lanes = Vec::new();
    for tailwater in [0.0, 1.125, 1.8] {
        for discharge in spread(10.0, 26.0) {
            lanes.push(steep_box(discharge, tailwater)?);
        }
    }
    for discharge in [12.0, 20.0] {
        for tailwater in spread(0.0, 2.5) {
            lanes.push(steep_box(discharge, tailwater)?);
        }
    }
    let found = headwater(&lanes)?;
    let points = spread(0.0, 1.0).len();
    for (index, sweep) in found.chunks(points).enumerate() {
        rising(sweep, |_, _| false, &format!("steep box sweep {index}"))?;
    }
    Ok(())
}

#[test]
fn steep_box_headwater_is_continuous_at_its_critical_capacity() -> TestResult {
    // Free and drowned outlets: critical depth reaches the crown as the discharge rises.
    for tailwater in [0.0, 1.8] {
        let base = steep_box(0.0, tailwater)?;
        let pair = transition(
            |discharge| Lane { discharge, ..base },
            11.0,
            12.0,
            |row| row.inlet_head_m.0 >= RISE,
            "critical depth reaching the crown",
        )?;
        continuous(pair, WIDTH * RISE, &format!("capacity at TW={tailwater}"))?;
    }
    Ok(())
}

/// `(Sf_full - S) L` of the box at `discharge`.
fn full_gain(discharge: f64) -> f64 {
    let area = SPAN * RISE;
    let radius = area / (2.0 * (SPAN + RISE));
    let velocity = discharge / area;
    (MANNING * MANNING * velocity * velocity / radius.powf(4.0 / 3.0) - SLOPE) * LENGTH
}

/// Bisects the input in `[low, high]` where `after` turns true down to adjacent F64 values,
/// and returns that input with the rows of `NEIGHBOURS` ulps on each side of it. Every row
/// must resolve and none may fall.
fn neighbourhood(
    make: impl Fn(f64) -> Lane,
    [mut low, mut high]: [f64; 2],
    after: impl Fn(&CulvertOutletHeadwaterRow) -> bool,
    what: &str,
) -> Outcome<(f64, Vec<CulvertOutletHeadwaterRow>)> {
    if after(&one(make(low))?) || !after(&one(make(high))?) {
        return Err(format!("{what}: transition lies outside [{low}, {high}]").into());
    }
    for _ in 0..STEPS {
        let middle = 0.5 * (low + high);
        if middle <= low || middle >= high {
            break;
        }
        if after(&one(make(middle))?) {
            high = middle;
        } else {
            low = middle;
        }
    }
    let bits = low.to_bits();
    let lanes: Vec<Lane> = (bits - NEIGHBOURS..=bits + NEIGHBOURS)
        .map(|bits| make(f64::from_bits(bits)))
        .collect();
    let found = headwater(&lanes)?;
    rising(&found, |_, _| false, what)?;
    let (Some(first), Some(last)) = (found.first(), found.last()) else {
        return Err(format!("{what}: no neighbourhood rows").into());
    };
    if first.regime != OutletRegime::InletCritical || !after(last) {
        return Err(format!("{what}: ends {:?} and {:?}", first.regime, last.regime).into());
    }
    Ok((low, found))
}

#[test]
fn steep_box_primes_with_a_step_and_no_unresolved_discharge() -> TestResult {
    // Free-surface friction at the crown reaches the slope and the barrel runs full.
    let base = steep_box(0.0, 0.0)?;
    let (discharge, found) = neighbourhood(
        |discharge| Lane { discharge, ..base },
        [20.0, 24.0],
        |row| row.regime == OutletRegime::PartlyFull,
        "priming discharge",
    )?;
    let (Some(first), Some(last)) = (found.first(), found.last()) else {
        return Err("no priming rows".into());
    };
    // The step is the full-barrel friction gain over the barrel.
    let step = last.headwater_m.0 - first.headwater_m.1;
    let gain = full_gain(discharge);
    if (step - gain).abs() > 1e-6 * gain {
        return Err(format!("priming step {step:e}, full-barrel gain {gain:e}").into());
    }
    Ok(())
}

#[test]
fn steep_box_resolves_where_its_full_outlet_head_reaches_the_crown_at_the_inlet() -> TestResult {
    let base = steep_box(12.0, 0.0)?;
    let (_, found) = neighbourhood(
        |tailwater| Lane { tailwater, ..base },
        [RISE, 2.5],
        |row| row.regime == OutletRegime::Full,
        "full outlet head at the crown",
    )?;
    let (Some(first), Some(last)) = (found.first(), found.last()) else {
        return Err("no tailwater rows".into());
    };
    let gap = last.headwater_m.1 - first.headwater_m.0;
    if gap > WIDTH * RISE {
        return Err(format!("headwater moves by {gap:e} across the crown").into());
    }
    Ok(())
}
