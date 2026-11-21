//! Tailwater and discharge sweeps: no certified fall of the headwater, and continuity where the
//! outlet turns full, the tailwater crosses critical depth, a profile reaches the crown, and a
//! steep barrel's drowned profile reaches critical depth at the inlet.

use super::lanes::{Lane, lane};
use super::outlet_rows::{OutletRow, PROFILE_ROWS};
use super::runtime::{Runtime, client};
use super::support::{TestResult, headwater};
use hydr::structures::{
    CulvertBarrelBatch, CulvertBarrelShape, CulvertOutletHeadwaterRow, OutletRegime,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// Points per sweep and per refinement round.
const POINTS: usize = 24;

/// Refinement rounds of a transition search. Each round shrinks the bracket by `POINTS + 1`.
const ROUNDS: usize = 8;

fn base(name: &str, circular: bool) -> Outcome<(Lane, f64)> {
    let row: &OutletRow = PROFILE_ROWS
        .iter()
        .find(|row| row.name == name && row.inputs.circular == circular)
        .ok_or_else(|| format!("no profile row {name}"))?;
    let made = lane(row)?;
    Ok((made, made.row.barrel.rise_m()))
}

/// `POINTS` evenly spaced values over `[low, high]`.
fn spread(low: f64, high: f64) -> Vec<f64> {
    (0..POINTS)
        .map(|i| low + (high - low) * i as f64 / (POINTS - 1) as f64)
        .collect()
}

fn resolved(rows: &[CulvertOutletHeadwaterRow], what: &str) -> TestResult {
    for (index, row) in rows.iter().enumerate() {
        if row.regime == OutletRegime::Unresolved {
            return Err(format!("{what}: point {index} is unresolved").into());
        }
    }
    Ok(())
}

/// No headwater falls certifiably as the swept input rises, except where `allowed` admits the
/// pair.
fn rising(
    rows: &[CulvertOutletHeadwaterRow],
    allowed: impl Fn(&CulvertOutletHeadwaterRow, &CulvertOutletHeadwaterRow) -> bool,
    what: &str,
) -> TestResult {
    resolved(rows, what)?;
    for (index, pair) in rows.windows(2).enumerate() {
        if let [before, after] = pair
            && after.headwater_m.1 < before.headwater_m.0
            && !allowed(before, after)
        {
            return Err(format!(
                "{what}: headwater falls from {:?} to {:?} at point {index}",
                before.headwater_m, after.headwater_m
            )
            .into());
        }
    }
    Ok(())
}

/// `(1 + ke) Fr^2` at a free-surface depth, from closed-form circle and box geometry.
fn loaded_froude(lane: &Lane, depth: f64) -> f64 {
    let barrel = lane.row.barrel;
    let rise = barrel.rise_m();
    let (area, top) = if barrel.shape() == CulvertBarrelShape::Circular {
        let phi = (1.0 - 2.0 * (depth / rise).min(1.0)).acos();
        (
            rise * rise * (phi - phi.sin() * phi.cos()) / 4.0,
            rise * phi.sin(),
        )
    } else {
        (barrel.span_m() * depth, barrel.span_m())
    };
    let flow = lane.discharge;
    (1.0 + lane.row.entrance_loss) * flow * flow * top
        / (lane.row.gravity_m_per_s2 * area * area * area)
}

/// A headwater may fall with rising tailwater only where `y + (1 + ke) V^2 / (2 g)` falls with
/// the inlet depth, so where `(1 + ke) Fr^2 > 1` at the lower inlet head of the pair.
fn energy_falls(
    lane: Lane,
) -> impl Fn(&CulvertOutletHeadwaterRow, &CulvertOutletHeadwaterRow) -> bool {
    move |before, after| {
        let depth = before.inlet_head_m.0.min(after.inlet_head_m.0);
        depth < lane.row.barrel.rise_m() && loaded_froude(&lane, depth) > 1.0
    }
}

#[test]
fn headwater_never_falls_as_tailwater_or_discharge_rises() -> TestResult {
    let bases = [
        ("m1", true),
        ("steep_jump", true),
        ("steep_drowned", false),
        ("horizontal", false),
        ("adverse", true),
        ("full_outlet_free_inlet", false),
    ];
    let mut lanes = Vec::new();
    for (name, circular) in bases {
        let (made, rise) = base(name, circular)?;
        for tailwater in spread(0.0, 1.6 * rise) {
            lanes.push(Lane { tailwater, ..made });
        }
        for scale in spread(0.1, 2.5) {
            lanes.push(Lane {
                discharge: made.discharge * scale,
                ..made
            });
        }
    }
    let found = headwater(&lanes)?;
    for ((name, circular), sweeps) in bases.iter().zip(found.chunks(2 * POINTS)) {
        let (made, _) = base(name, *circular)?;
        let (tailwaters, discharges) = sweeps.split_at(POINTS);
        rising(
            tailwaters,
            energy_falls(made),
            &format!("{name} circular={circular} tailwater sweep"),
        )?;
        rising(
            discharges,
            |_, _| false,
            &format!("{name} circular={circular} discharge sweep"),
        )?;
    }
    Ok(())
}

/// Brackets the input where `side` flips between `low` and `high` and returns the headwaters at
/// the two ends of the final bracket.
fn transition(
    make: impl Fn(f64) -> Lane,
    mut low: f64,
    mut high: f64,
    side: impl Fn(&CulvertOutletHeadwaterRow) -> bool,
    what: &str,
) -> Outcome<[CulvertOutletHeadwaterRow; 2]> {
    let ends = headwater(&[make(low), make(high)])?;
    let (Some(first), Some(last)) = (ends.first().copied(), ends.get(1).copied()) else {
        return Err(format!("{what}: no end rows").into());
    };
    if side(&first) == side(&last) {
        return Err(format!(
            "{what}: {:?} and {:?} lie on one side",
            first.regime, last.regime
        )
        .into());
    }
    let (mut below, mut above) = (first, last);
    for _ in 0..ROUNDS {
        let points: Vec<f64> = (1..=POINTS)
            .map(|i| low + (high - low) * i as f64 / (POINTS + 1) as f64)
            .collect();
        let rows = headwater(&points.iter().map(|&p| make(p)).collect::<Vec<_>>())?;
        resolved(&rows, what)?;
        for (&point, row) in points.iter().zip(&rows) {
            if side(row) == side(&below) {
                low = point;
                below = *row;
            } else {
                high = point;
                above = *row;
                break;
            }
        }
    }
    Ok([below, above])
}

/// Both headwater enclosures lie within `width` of each other and each is at most `width`
/// wide. The two inputs lie `(POINTS + 1)^-ROUNDS` of the search span apart.
fn continuous(pair: [CulvertOutletHeadwaterRow; 2], width: f64, what: &str) -> TestResult {
    let [before, after] = pair;
    let (low, high) = (before.headwater_m, after.headwater_m);
    let gap = (high.0 - low.1).max(low.0 - high.1);
    let widest = (low.1 - low.0).max(high.1 - high.0);
    if gap > width || widest > width {
        return Err(format!(
            "{what}: headwater moves from {low:?} ({:?}) to {high:?} ({:?})",
            before.regime, after.regime
        )
        .into());
    }
    Ok(())
}

/// Width and gap allowed where the inlet depth sits at a critical-depth limit, relative to the
/// rise. There the profile length varies with the square of the depth offset, so the headwater
/// moves with the square root of an input offset `e` and a certified length of relative width
/// `e` leaves a headwater width near `sqrt(e)`.
const CRITICAL_WIDTH: f64 = 1e-5;

/// Width and gap allowed at every other transition, relative to the rise.
const WIDTH: f64 = 1e-8;

#[test]
fn headwater_is_continuous_where_the_outlet_turns_full_and_at_critical_depth() -> TestResult {
    // Mild circle: the outlet depth max(TW, yc) moves from yc to TW and then the outlet turns full.
    let (mild, rise) = base("m1", true)?;
    let tail = |tailwater: f64| Lane { tailwater, ..mild };
    let barrels = CulvertBarrelBatch::<Runtime>::new(client(), vec![mild.row.barrel])?;
    let flow = Tensor::<Runtime>::from_slice(&[mild.discharge], &[1], client().device())?;
    let critical = barrels
        .critical_depth(&flow, mild.row.gravity_m_per_s2)?
        .project()?;
    let (low, high) = critical
        .first()
        .and_then(|row| row.depth_m)
        .ok_or("no critical depth enclosure")?;
    // At and below critical depth the outlet depth is yc, so the headwater does not move.
    let rows = headwater(&[
        tail(0.0),
        tail(low * (1.0 - 1e-9)),
        tail(high * (1.0 + 1e-9)),
    ])?;
    resolved(&rows, "tailwater crossing critical depth")?;
    if let [dry, below, above] = rows.as_slice() {
        if dry.headwater_m != below.headwater_m {
            return Err(format!(
                "tailwater below critical depth moves the headwater from {:?} to {:?}",
                dry.headwater_m, below.headwater_m
            )
            .into());
        }
        continuous(
            [*below, *above],
            WIDTH * rise,
            "tailwater crossing critical depth",
        )?;
    }
    let full = transition(
        tail,
        0.9 * rise,
        1.1 * rise,
        |row| row.regime == OutletRegime::Full,
        "outlet turning full",
    )?;
    continuous(full, WIDTH * rise, "outlet turning full")?;
    Ok(())
}

#[test]
fn headwater_is_continuous_where_a_rising_profile_reaches_the_crown() -> TestResult {
    // A rising profile that reaches the crown runs full to the inlet.
    for circular in [true, false] {
        let (crown, rise) = base("rising_to_crown", circular)?;
        let flow = |scale: f64| Lane {
            discharge: crown.discharge * scale,
            ..crown
        };
        let pair = transition(
            flow,
            0.5,
            1.0,
            |row| row.inlet_head_m.0 >= rise,
            "profile reaching the crown",
        )?;
        continuous(pair, WIDTH * rise, &format!("circular={circular} crown"))?;
    }
    Ok(())
}

#[test]
fn headwater_is_continuous_where_a_drowned_steep_profile_reaches_critical_depth() -> TestResult {
    // A drowned steep profile falls to critical depth at the inlet as the tailwater drops.
    for circular in [true, false] {
        let (steep, rise) = base("steep_drowned", circular)?;
        let tail = |tailwater: f64| Lane { tailwater, ..steep };
        let pair = transition(
            tail,
            0.5 * rise,
            steep.tailwater,
            |row| row.regime == OutletRegime::InletCritical,
            "drowned profile reaching critical depth",
        )?;
        continuous(
            pair,
            CRITICAL_WIDTH * rise,
            &format!("circular={circular} critical inlet"),
        )?;
    }
    Ok(())
}

#[test]
fn headwater_is_continuous_where_a_full_outlet_falls_to_the_crown_inside_the_barrel() -> TestResult
{
    // A full outlet whose head falls to the crown inside the barrel.
    for circular in [true, false] {
        let (falling, rise) = base("full_outlet_free_inlet", circular)?;
        let tail = |tailwater: f64| Lane {
            tailwater,
            ..falling
        };
        let pair = transition(
            tail,
            falling.tailwater,
            3.0 * rise,
            |row| row.regime == OutletRegime::Full,
            "full outlet reaching the inlet full",
        )?;
        continuous(
            pair,
            WIDTH * rise,
            &format!("circular={circular} full inlet"),
        )?;
    }
    Ok(())
}
