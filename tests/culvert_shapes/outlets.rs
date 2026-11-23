//! Arc barrel outlet control: the frozen outlet rows, continuity where the tailwater crosses
//! critical depth, a junction depth or the crown, and monotone sweeps.

use super::lanes::{Outcome, OutletLane as Lane, outlet_lane as lane};
use super::outlet_rows::{
    BARRELS, FULL_RELATIVE_TOLERANCE, FULL_ROWS, JUMP_POSITION_TOLERANCE, OUTLET_SHA256, Outlet,
    OutletRow, PROFILE_RELATIVE_TOLERANCE, PROFILE_ROWS,
};
use super::runtime::{Runtime, client};
use hydr::structures::{
    CulvertOutletBatch, CulvertOutletHeadwaterRow, CulvertOutletJumpRow, JumpStatus, OutletRegime,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

type TestResult = Outcome<()>;

/// Points of each monotone sweep.
const POINTS: usize = 16;

fn profile(barrel: usize, name: &str) -> Outcome<&'static OutletRow> {
    PROFILE_ROWS
        .iter()
        .find(|row| row.inputs.barrel == barrel && row.name == name)
        .ok_or_else(|| format!("no profile row {name} for barrel {barrel}").into())
}

/// Headwater and jump rows of one batch over `lanes`.
fn evaluate(
    lanes: &[Lane],
) -> Outcome<(Vec<CulvertOutletHeadwaterRow>, Vec<CulvertOutletJumpRow>)> {
    let client = client();
    let batch = CulvertOutletBatch::<Runtime>::new(
        client.clone(),
        lanes.iter().map(|lane| lane.row).collect(),
    )?;
    let flows: Vec<f64> = lanes.iter().map(|lane| lane.discharge).collect();
    let tails: Vec<f64> = lanes.iter().map(|lane| lane.tailwater).collect();
    let discharge = Tensor::<Runtime>::from_slice(&flows, &[flows.len()], client.device())?;
    let tailwater = Tensor::<Runtime>::from_slice(&tails, &[tails.len()], client.device())?;
    let found = batch.jump(&discharge, &tailwater)?;
    Ok((found.headwater().project()?, found.project()?))
}

/// Headwater rows of one batch over `lanes`.
fn headwater(lanes: &[Lane]) -> Outcome<Vec<CulvertOutletHeadwaterRow>> {
    let client = client();
    let batch = CulvertOutletBatch::<Runtime>::new(
        client.clone(),
        lanes.iter().map(|lane| lane.row).collect(),
    )?;
    let flows: Vec<f64> = lanes.iter().map(|lane| lane.discharge).collect();
    let tails: Vec<f64> = lanes.iter().map(|lane| lane.tailwater).collect();
    let discharge = Tensor::<Runtime>::from_slice(&flows, &[flows.len()], client.device())?;
    let tailwater = Tensor::<Runtime>::from_slice(&tails, &[tails.len()], client.device())?;
    Ok(batch.headwater(&discharge, &tailwater)?.project()?)
}

/// Requires `bounds` to hold the reference bits and to lie within `allowed` of them.
fn holds_within(bounds: (f64, f64), reference: [u64; 2], allowed: f64, what: &str) -> TestResult {
    let (low, high) = (f64::from_bits(reference[0]), f64::from_bits(reference[1]));
    if !(bounds.0 <= low && high <= bounds.1) {
        return Err(format!(
            "{what}: [{:e}, {:e}] misses the reference [{low:e}, {high:e}]",
            bounds.0, bounds.1
        )
        .into());
    }
    let reach = (high - bounds.0).max(bounds.1 - low);
    if reach > allowed {
        return Err(format!(
            "{what}: enclosure reaches {reach:e} from the reference, above {allowed:e}"
        )
        .into());
    }
    Ok(())
}

/// [`holds_within`] with a tolerance relative to the reference.
fn holds(bounds: (f64, f64), reference: [u64; 2], relative: f64, what: &str) -> TestResult {
    let scale = f64::from_bits(reference[0])
        .abs()
        .max(f64::from_bits(reference[1]).abs());
    holds_within(bounds, reference, relative * scale, what)
}

fn label(row: &OutletRow) -> String {
    let name = BARRELS
        .get(row.inputs.barrel)
        .map_or("?", |b| b.barrel.name);
    format!("{name} {}", row.name)
}

#[test]
fn outlet_fixture_is_frozen() {
    assert_eq!(
        OUTLET_SHA256,
        "b6985091fe6cee91c927c078f610adc6a05c90f450f81375b24ce6f9513e20f3"
    );
    assert_eq!(BARRELS.len(), 4);
    assert_eq!(FULL_ROWS.len(), 36);
    assert_eq!(PROFILE_ROWS.len(), 52);
}

#[test]
fn full_barrel_rows_lie_inside_their_enclosures() -> TestResult {
    let lanes = FULL_ROWS.iter().map(lane).collect::<Outcome<Vec<_>>>()?;
    let found = headwater(&lanes)?;
    for (row, result) in FULL_ROWS.iter().zip(&found) {
        let what = label(row);
        if result.regime != OutletRegime::Full {
            return Err(format!("{what}: regime {:?}", result.regime).into());
        }
        holds(
            result.headwater_m,
            row.headwater,
            FULL_RELATIVE_TOLERANCE,
            &what,
        )?;
    }
    Ok(())
}

fn regime(row: &OutletRow) -> OutletRegime {
    match row.regime {
        Outlet::Full => OutletRegime::Full,
        Outlet::PartlyFull => OutletRegime::PartlyFull,
        Outlet::Supercritical | Outlet::Jump | Outlet::CriticalWithFullExit => {
            OutletRegime::InletCritical
        }
    }
}

/// Checks the jump fields of one profile row.
fn jump_holds(row: &OutletRow, jump: &CulvertOutletJumpRow) -> TestResult {
    let what = label(row);
    let tolerance = PROFILE_RELATIVE_TOLERANCE;
    match (row.regime, row.jump, row.outlet_depth) {
        (Outlet::Jump, Some(expected), _) => {
            if jump.status != JumpStatus::InBarrel {
                return Err(format!("{what}: jump status {:?}", jump.status).into());
            }
            let allowed = JUMP_POSITION_TOLERANCE * f64::from_bits(row.inputs.length);
            holds_within(
                jump.position_m,
                expected.position,
                allowed,
                &format!("{what} jump position"),
            )?;
            holds(
                jump.upstream_depth_m,
                expected.upstream_depth,
                tolerance,
                &format!("{what} upstream depth"),
            )?;
            holds(
                jump.downstream_depth_m,
                expected.downstream_depth,
                tolerance,
                &format!("{what} downstream depth"),
            )
        }
        (Outlet::Supercritical, _, Some(outlet)) => {
            if jump.status != JumpStatus::Supercritical {
                return Err(format!("{what}: jump status {:?}", jump.status).into());
            }
            holds(
                jump.outlet_depth_m,
                outlet,
                tolerance,
                &format!("{what} outlet depth"),
            )
        }
        (Outlet::CriticalWithFullExit, _, _) => {
            if jump.status == JumpStatus::Unresolved {
                Ok(())
            } else {
                Err(format!("{what}: jump status {:?}", jump.status).into())
            }
        }
        (Outlet::Full | Outlet::PartlyFull, _, _) => {
            if jump.status == JumpStatus::NotSteep {
                Ok(())
            } else {
                Err(format!("{what}: jump status {:?}", jump.status).into())
            }
        }
        _ => Err(format!("{what}: reference row lacks its jump fields").into()),
    }
}

#[test]
fn profile_and_jump_rows_lie_inside_their_enclosures() -> TestResult {
    let lanes = PROFILE_ROWS.iter().map(lane).collect::<Outcome<Vec<_>>>()?;
    let (found, jumps) = evaluate(&lanes)?;
    let tolerance = PROFILE_RELATIVE_TOLERANCE;
    let mut errors = Vec::new();
    for ((row, result), jump) in PROFILE_ROWS.iter().zip(&found).zip(&jumps) {
        let what = label(row);
        if result.regime != regime(row) {
            errors.push(format!(
                "{what}: regime {:?}, reference {:?}",
                result.regime, row.regime
            ));
            continue;
        }
        let mut checks = vec![holds(
            result.headwater_m,
            row.headwater,
            tolerance,
            &format!("{what} headwater"),
        )];
        if let Some(inlet) = row.inlet_head {
            checks.push(holds(
                result.inlet_head_m,
                inlet,
                tolerance,
                &format!("{what} inlet head"),
            ));
        }
        checks.push(jump_holds(row, jump));
        errors.extend(
            checks
                .into_iter()
                .filter_map(Result::err)
                .map(|e| e.to_string()),
        );
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n").into())
    }
}

/// `lane` at tailwater `tailwater`.
fn at_tail(lane: Lane, tailwater: f64) -> Lane {
    Lane { tailwater, ..lane }
}

#[test]
fn headwater_is_continuous_where_the_tailwater_crosses_critical_depth_a_junction_or_the_crown()
-> TestResult {
    let mut lanes = Vec::new();
    let mut names = Vec::new();
    for (index, entry) in BARRELS.iter().enumerate() {
        let mild = lane(profile(index, "m1")?)?;
        let critical = profile(index, "m2_critical_outlet")?
            .critical_depth
            .ok_or("m2 row lacks its critical depth")?;
        let rise = mild.row.barrel.rise_m();
        let mut seams = vec![
            ("critical depth", f64::from_bits(critical[1])),
            ("crown", rise),
        ];
        for junction in entry.junctions {
            seams.push(("junction", f64::from_bits(junction[1])));
        }
        for (name, depth) in seams {
            for tail in [depth.next_down().next_down(), depth.next_up()] {
                lanes.push(at_tail(mild, tail));
            }
            names.push(format!("{} {name}", entry.barrel.name));
        }
    }
    let rows = headwater(&lanes)?;
    for (pair, name) in rows.chunks(2).zip(&names) {
        let [below, above] = pair else { continue };
        for row in [below, above] {
            if row.regime == OutletRegime::Unresolved {
                return Err(format!("{name}: unresolved").into());
            }
        }
        let low = below.headwater_m.0.min(above.headwater_m.0);
        let high = below.headwater_m.1.max(above.headwater_m.1);
        let allowed = 2.0 * PROFILE_RELATIVE_TOLERANCE * high.abs();
        if high - low > allowed {
            return Err(format!(
                "{name}: headwater {:?} below and {:?} above differ by more than {allowed:e}",
                below.headwater_m, above.headwater_m
            )
            .into());
        }
    }
    Ok(())
}

/// No headwater falls certifiably along a sweep of rising input.
fn rising(rows: &[CulvertOutletHeadwaterRow], what: &str) -> TestResult {
    for (index, pair) in rows.windows(2).enumerate() {
        let [before, after] = pair else { continue };
        if before.regime == OutletRegime::Unresolved || after.regime == OutletRegime::Unresolved {
            return Err(format!("{what}: point {index} is unresolved").into());
        }
        if after.headwater_m.1 < before.headwater_m.0 {
            return Err(format!(
                "{what}: headwater falls from {:?} to {:?} at point {index}",
                before.headwater_m, after.headwater_m
            )
            .into());
        }
    }
    Ok(())
}

#[test]
fn headwater_never_falls_as_discharge_or_tailwater_rises() -> TestResult {
    let fraction = |k: usize| (k + 1) as f64 / POINTS as f64;
    let mut by_flow = Vec::new();
    let mut by_tail = Vec::new();
    for index in 0..BARRELS.len() {
        let mild = lane(profile(index, "m1")?)?;
        let rise = mild.row.barrel.rise_m();
        for k in 0..POINTS {
            by_flow.push(Lane {
                discharge: mild.discharge * 2.0 * fraction(k),
                tailwater: 0.5 * rise,
                ..mild
            });
            by_tail.push(at_tail(mild, 1.5 * rise * fraction(k)));
        }
    }
    for (lanes, what) in [(&by_flow, "discharge"), (&by_tail, "tailwater")] {
        let rows = headwater(lanes)?;
        for (index, chunk) in rows.chunks(POINTS).enumerate() {
            let name = BARRELS.get(index).map_or("?", |b| b.barrel.name);
            rising(chunk, &format!("{name} {what} sweep"))?;
        }
    }
    Ok(())
}
