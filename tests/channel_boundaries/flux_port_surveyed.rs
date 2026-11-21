//! Flux ports on standalone surveyed reaches: uniform flow, admission and the water ledger.
//!
//! Criteria are frozen before any run. A test that misses one fails.
//!
//! Frozen criteria:
//!
//! - Flux port, standalone: rectangle and three bands, uniform state, exact physical flux at
//!   both ends. Relative change of every `A` and `Q` `<= 1e-12` over 4 intervals. Each port
//!   rejection fires.
//! - Mass, standalone: every standalone row. Each interval: ledger residual within its bound.
//!   `Discharge` face volume equals `RN(σ Q dt)` bit for bit. End face low parts equal `+0.0`.
//!
//! The tests below refine those rows:
//!
//! - Uniform flow: the `rectangle-3` and `three-bands` profiles at a uniform state, with
//!   ports at both ends that supply the exact physical flux of that state. After `INTERVALS`
//!   accepted intervals, the relative change of every cell area and discharge is at most
//!   `UNIFORM_RELATIVE`.
//! - Rejections: a port slower than the end cell's `|u| + sqrt(g D_h)` rejects with the
//!   `flux-port-speed-below-interior` text. Export from a dry end cell rejects with the
//!   `flux-port-dry-export` text. Each rejection leaves the reach unchanged. Import into a dry
//!   end cell is admitted.
//! - Mass: every interval, the ledger residual plus its charges stays within the volume
//!   tolerance. Each end face volume and momentum equals `RN(rate · dt)` bit for bit, with a
//!   `+0.0` low part. The end face transfer bound covers the product tail.

use super::common::TestResult;
use crate::surveyed_profiles::{Profile, RECTANGLE_3, THREE_BANDS};
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelError, ChannelFluxPort, ChannelForcingInterval,
    ChannelGeometry, ChannelInitialState, ChannelReach, ChannelReachDefinition,
    ChannelSurveyedConfig, CrossSectionDefinition, CrossSectionPoint, SurveyedChannelCell,
};

type Reach = ChannelReach<crate::runtime::Runtime>;

/// Largest relative change of any cell area or discharge under uniform flow after `INTERVALS`.
const UNIFORM_RELATIVE: f64 = 1e-12;

/// Accepted intervals per uniform-flow run. Each interval feeds the last state back in, so a
/// drift shows in a few intervals.
const INTERVALS: usize = 4;

/// Longest requested interval in seconds.
const SPAN: f64 = 0.25;

/// Gravity in m/s² of every fixture.
const GRAVITY: f64 = 9.81;

/// Cell length in metres.
const LENGTH: f64 = 4.0;

/// Cells per reach.
const CELLS: usize = 4;

/// Uniform depth in metres: inside the second band of `three-bands`.
const DEPTH: f64 = 1.5;

/// Simpson panels for the invariant `φ`, which no wet uniform run reads.
const PHI_PANELS: usize = 2000;

const PROFILES: [Profile; 2] = [RECTANGLE_3, THREE_BANDS];

/// Survey polyline of `profile` as `(station, elevation)`.
fn points(profile: Profile) -> Vec<(f64, f64)> {
    profile
        .points
        .iter()
        .map(|p| (f64::from_bits(p[0]), f64::from_bits(p[1])))
        .collect()
}

/// Station where one monotone bank reaches elevation `y`.
fn station(bank: &[(f64, f64)], y: f64) -> f64 {
    for pair in bank.windows(2) {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        let (low, high) = (y0.min(y1), y0.max(y1));
        if y >= low && y <= high && high > low {
            return x0 + (x1 - x0) * (y - y0) / (y1 - y0);
        }
    }
    bank[0].0
}

/// Top width at depth `y` above the bed at elevation zero.
fn width(profile: Profile, y: f64) -> f64 {
    let points = points(profile);
    let first = points.iter().position(|p| p.1 == 0.0).unwrap_or(0);
    let last = points.iter().rposition(|p| p.1 == 0.0).unwrap_or(0);
    station(&points[last..], y) - station(&points[..=first], y)
}

/// Depth intervals between band knots, cut at `d`.
fn pieces(profile: Profile, d: f64) -> Vec<(f64, f64)> {
    profile
        .knots
        .windows(2)
        .filter(|k| k[0] < d)
        .map(|k| (k[0], k[1].min(d)))
        .collect()
}

/// Flow area `A(d)`. Width is linear on each band, so the trapezoid rule is exact.
fn area(profile: Profile, d: f64) -> f64 {
    pieces(profile, d)
        .into_iter()
        .map(|(a, b)| 0.5 * (width(profile, a) + width(profile, b)) * (b - a))
        .sum()
}

/// Hydrostatic integral `I(d) = ∫ (d - y) T(y) dy`. Its integrand is quadratic on each band,
/// so Simpson's rule is exact.
fn hydrostatic(profile: Profile, d: f64) -> f64 {
    let f = |y: f64| (d - y) * width(profile, y);
    pieces(profile, d)
        .into_iter()
        .map(|(a, b)| (b - a) / 6.0 * (f(a) + 4.0 * f(0.5 * (a + b)) + f(b)))
        .sum()
}

/// Sound speed `sqrt(g A / T)` at depth `d`.
fn sound(profile: Profile, d: f64) -> f64 {
    (GRAVITY * area(profile, d) / width(profile, d)).sqrt()
}

/// Riemann invariant `φ(d) = ∫ sqrt(g T / A) dy`, with `y = d x²` to remove the bed singularity.
fn invariant(profile: Profile, d: f64) -> f64 {
    let f = |x: f64| {
        if x == 0.0 {
            return 2.0 * (GRAVITY * d * width(profile, 0.0) / width(profile, 0.0)).sqrt();
        }
        let y = d * x * x;
        (GRAVITY * width(profile, y) / area(profile, y)).sqrt() * 2.0 * d * x
    };
    let h = 1.0 / PHI_PANELS as f64;
    let mut sum = f(0.0) + f(1.0);
    for i in 1..PHI_PANELS {
        sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

fn definition(profile: Profile) -> ChannelReachDefinition {
    let datum = "Synthetic SI datum";
    ChannelReachDefinition {
        vertical_datum: datum.into(),
        provenance: "Surveyed flux port acceptance".into(),
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile: CrossSectionDefinition {
                id: profile.name.into(),
                vertical_datum: datum.into(),
                points: points(profile)
                    .into_iter()
                    .map(|(station_m, elevation_m)| CrossSectionPoint {
                        station_m,
                        elevation_m,
                    })
                    .collect(),
                maximum_stage_m: f64::from_bits(profile.cap),
                profile_provenance: "Frozen surveyed endpoint reference profile".into(),
            },
            cells: (0..CELLS)
                .map(|i| SurveyedChannelCell {
                    id: format!("cell-{i}"),
                    length_m: LENGTH,
                    manning_n: 0.0,
                })
                .collect(),
            numerical: ChannelSurveyedConfig::default(),
        },
        lateral_bindings: Vec::new(),
        config: ChannelConfig {
            gravity: GRAVITY,
            max_step: 1.0,
            ..ChannelConfig::default()
        },
    }
}

/// Reach of `profile` with every cell at area `a` and discharge `q`.
fn uniform_reach(profile: Profile, a: f64, q: f64) -> Result<Reach, Box<dyn std::error::Error>> {
    Ok(ChannelReach::new(
        crate::runtime::client(),
        definition(profile),
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![a * LENGTH; CELLS],
            longitudinal_momentum_m4_per_s: vec![q * LENGTH; CELLS],
        },
    )?)
}

fn port(
    water: f64,
    momentum: f64,
    speed: f64,
    invariant: Option<f64>,
) -> Result<ChannelBoundary, ChannelError> {
    ChannelFluxPort::new(
        water,
        momentum,
        speed,
        invariant,
        "test provider of exact surveyed flux",
    )
    .map(ChannelBoundary::FluxPort)
}

/// Ports at both ends with the exact physical flux of the uniform state at `DEPTH` and `u`.
///
/// The speed is `|u| + c` raised by two ulps, so it meets the reach's certified lower bound.
fn uniform_ports(profile: Profile, u: f64) -> Result<[ChannelBoundary; 2], ChannelError> {
    let q = area(profile, DEPTH) * u;
    let momentum = q * u + GRAVITY * hydrostatic(profile, DEPTH);
    let speed = (u.abs() + sound(profile, DEPTH)).next_up().next_up();
    let phi = invariant(profile, DEPTH);
    Ok([
        port(-q, momentum, speed, Some(u + phi))?,
        port(q, momentum, speed, Some(-u + phi))?,
    ])
}

fn held(reach: &Reach, left: &ChannelBoundary, right: &ChannelBoundary) -> ChannelForcingInterval {
    ChannelForcingInterval {
        start_s: reach.time(),
        end_s: reach.time() + SPAN,
        left: left.clone(),
        right: right.clone(),
        lateral_inputs: Vec::new(),
    }
}

/// Ledger residual plus every charge stays within the volume tolerance.
fn check_ledger(reach: &Reach) -> TestResult {
    let ledger = reach.ledger();
    let scale = ledger
        .initial_storage_m3
        .max(ledger.storage_m3)
        .max(ledger.boundary_in_m3)
        .max(ledger.boundary_out_m3)
        .max(1.0);
    let charged = ledger.residual_m3.abs()
        + ledger.numerical_transfer_underflow_bound_m3
        + ledger.lateral_counter_abs_bound_m3
        + ledger.lateral_assembly_abs_bound_m3
        + ledger.storage_selection_abs_bound_m3;
    let tolerance = reach.config().volume_absolute + reach.config().volume_relative * scale;
    assert!(
        charged <= tolerance,
        "charged={charged:e}, tolerance={tolerance:e}"
    );
    Ok(())
}

/// End face volumes and momenta of the last interval are the one-part products of the rates.
fn check_end_faces(reach: &Reach, ports: &[ChannelBoundary; 2]) -> TestResult {
    let receipt = reach.last_step()?.ok_or("missing interval receipt")?;
    let dt = receipt.end_s - receipt.start_s;
    for (endpoint, face) in [(0, 0), (1, CELLS)] {
        let ChannelBoundary::FluxPort(port) = &ports[endpoint] else {
            return Err("end policy is not a FluxPort".into());
        };
        let sign = if endpoint == 0 { -1.0 } else { 1.0 };
        let water = sign * port.outward_water_rate_m3_per_s();
        let momentum = port.outward_momentum_rate_m4_per_s2();
        let volume = receipt.face_volume_m3[face];
        assert_eq!(volume.to_bits(), (water * dt).to_bits(), "face={face}");
        assert_eq!(
            receipt.face_momentum_m4_per_s[face].to_bits(),
            (momentum * dt).to_bits(),
            "face={face}"
        );
        assert_eq!(
            receipt.face_volume_low_m3[face].to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            receipt.face_momentum_low_m4_per_s[face].to_bits(),
            0.0_f64.to_bits()
        );
        let tail = water.mul_add(dt, -volume).abs();
        assert!(
            receipt.face_transfer_underflow_bound_m3[face] >= tail,
            "face={face}"
        );
    }
    let left = receipt.face_volume_m3[0];
    let right = receipt.face_volume_m3[CELLS];
    let incoming = left.max(0.0) + (-right).max(0.0);
    let outgoing = (-left).max(0.0) + right.max(0.0);
    assert_eq!(receipt.boundary_in_m3.to_bits(), incoming.to_bits());
    assert_eq!(receipt.boundary_out_m3.to_bits(), outgoing.to_bits());
    Ok(())
}

/// Rejection text of one step, which must leave the reach unchanged.
fn rejection(
    origin: &Reach,
    left: &ChannelBoundary,
    right: &ChannelBoundary,
) -> Result<(ChannelError, String), Box<dyn std::error::Error>> {
    let before = origin.snapshot_data()?;
    let forcing = held(origin, left, right);
    let Err(error) = origin.step(forcing.end_s, &forcing) else {
        return Err("the port was admitted".into());
    };
    assert_eq!(origin.snapshot_data()?, before, "rejection changed state");
    let message = error.to_string();
    Ok((error, message))
}

#[test]
fn exact_uniform_flux_keeps_uniform_surveyed_flow() -> TestResult {
    for profile in PROFILES {
        let a = area(profile, DEPTH);
        let c = sound(profile, DEPTH);
        // Subcritical Fr 0.5 and supercritical Fr 2, in both directions.
        for u in [0.5 * c, -0.5 * c, 2.0 * c, -2.0 * c] {
            let q = a * u;
            let ports = uniform_ports(profile, u)?;
            let mut current = uniform_reach(profile, a, q)?;
            for _ in 0..INTERVALS {
                let forcing = held(&current, &ports[0], &ports[1]);
                current = current.step(forcing.end_s, &forcing)?.next;
                check_ledger(&current)?;
                check_end_faces(&current, &ports)?;
            }
            for cell in current.state()? {
                let relative_area = (cell.volume_m3 / LENGTH - a).abs() / a;
                let relative_discharge = (cell.discharge_m3_per_s - q).abs() / q.abs();
                assert!(
                    relative_area <= UNIFORM_RELATIVE && relative_discharge <= UNIFORM_RELATIVE,
                    "{} u={u}: area={relative_area:e}, discharge={relative_discharge:e}, {cell:?}",
                    profile.name
                );
            }
        }
    }
    Ok(())
}

#[test]
fn a_port_slower_than_the_hydraulic_depth_characteristic_rejects() -> TestResult {
    for profile in PROFILES {
        let a = area(profile, DEPTH);
        let c = sound(profile, DEPTH);
        let u = 0.5 * c;
        let [left, right] = uniform_ports(profile, u)?;
        let origin = uniform_reach(profile, a, a * u)?;
        let slow_speed = (u + c) * (1.0 - 1e-9);
        let pressure = GRAVITY * hydrostatic(profile, DEPTH);
        let slow = port(0.0, pressure, slow_speed, Some(slow_speed))?;
        for (l, r, cell) in [(&slow, &right, 0), (&left, &slow, CELLS - 1)] {
            let (error, message) = rejection(&origin, l, r)?;
            assert!(
                matches!(error, ChannelError::InvalidInput { cell: Some(found), .. } if found == cell),
                "{}: {error:?}",
                profile.name
            );
            for fragment in [
                "is below the end cell |u|+c",
                &format!("speed={slow_speed}"),
                "supply a signal-speed bound of at least the end cell characteristic speed",
            ] {
                assert!(message.contains(fragment), "{message} lacks {fragment}");
            }
        }
    }
    Ok(())
}

#[test]
fn export_from_a_dry_surveyed_end_rejects_and_import_fills_it() -> TestResult {
    for profile in PROFILES {
        let origin = uniform_reach(profile, 0.0, 0.0)?;
        let export = port(0.5, 0.0, 1.0, None)?;
        let wall = ChannelBoundary::Wall;
        for (left, right, side) in [(&export, &wall, "left"), (&wall, &export, "right")] {
            let (error, message) = rejection(&origin, left, right)?;
            assert!(
                matches!(error, ChannelError::InvalidInput { .. }),
                "{error:?}"
            );
            let expected = format!(
                "flux port {side} exports water=0.5 m³/s from a dry end cell: supply zero or entering water at a dry end"
            );
            assert!(message.contains(&expected), "{message}");
        }
        let import = port(-0.01, 0.0, 1.0, Some(7.0))?;
        for (left, right, cell) in [(&import, &wall, 0), (&wall, &import, CELLS - 1)] {
            let forcing = held(&origin, left, right);
            let next = origin.step(forcing.end_s, &forcing)?.next;
            check_ledger(&next)?;
            let state = next.state()?;
            assert!(state[cell].volume_m3 > 0.0, "{}: {state:?}", profile.name);
            let receipt = next.last_step()?.ok_or("missing fill receipt")?;
            let dt = receipt.end_s - receipt.start_s;
            assert_eq!(receipt.boundary_in_m3.to_bits(), (0.01 * dt).to_bits());
            assert_eq!(next.ledger().boundary_out_m3, 0.0);
        }
    }
    Ok(())
}
