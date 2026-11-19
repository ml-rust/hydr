//! Channel reach declarations checked against their admission code.

use crate::channel_fixture::{
    Case, END, KINDS, Kind, Outcome, Reach, TestResult, Variant, admits, definition, exterior,
    initial, interval, lateral_inputs, pool_outcome, reach, step_outcome,
};
use crate::channel_variants::{
    ENDPOINTS, PORT_SPEED, RECONSTRUCTIONS, boundary_kind, port, reconstruction_id, samples,
    source_kind,
};
use hydr::capability::{
    BoundaryKind, EndpointId, Regime, RegimeScope, RegimeSupport, UnsupportedReason,
};
use hydr::channel::{ChannelBoundary, ChannelConfig};
use hydr::forcing::WaterAmount;

/// Outflow speed in m/s that exceeds the gravity-wave speed of every fixture at unit depth.
const FAST: f64 = 5.0;

/// Places `boundary` at `endpoint` and a wall at the other end.
fn at(endpoint: EndpointId, boundary: ChannelBoundary) -> [ChannelBoundary; 2] {
    if endpoint == EndpointId::ReachLeft {
        [boundary, ChannelBoundary::Wall]
    } else {
        [ChannelBoundary::Wall, boundary]
    }
}

/// One cell with velocity `u` and `boundary` at `endpoint`.
fn one_cell(
    kind: Kind,
    h: f64,
    u: f64,
    endpoint: EndpointId,
    boundary: ChannelBoundary,
) -> Outcome {
    let [left, right] = at(endpoint, boundary);
    step_outcome(reach(kind, &[h], &[u]), &interval(left, right))
}

/// Outward velocity sign at `endpoint`.
fn outward(endpoint: EndpointId) -> f64 {
    if endpoint == EndpointId::ReachLeft {
        -1.0
    } else {
        1.0
    }
}

/// Full-section pressure force per unit density at depth `h`, in m⁴/s².
fn pressure(kind: Kind, h: f64) -> f64 {
    let g = ChannelConfig::default().gravity;
    match kind {
        // Width 2 m: g b h² / 2.
        Kind::Rectangular => g * h * h,
        // Unit bank slopes: g h³ / 3.
        Kind::Triangular | Kind::Surveyed => g * h * h * h / 3.0,
    }
}

/// Gravity-wave speed at depth `h`: `sqrt(g h)` for the rectangle, `sqrt(g h / 2)` otherwise.
fn sound(kind: Kind, h: f64) -> f64 {
    let g = ChannelConfig::default().gravity;
    match kind {
        Kind::Rectangular => (g * h).sqrt(),
        Kind::Triangular | Kind::Surveyed => (g * h / 2.0).sqrt(),
    }
}

/// One cell with velocity `u` and a flux port at `endpoint`.
fn port_cell(
    kind: Kind,
    h: f64,
    u: f64,
    endpoint: EndpointId,
    rates: (f64, f64, f64, Option<f64>),
) -> Outcome {
    let (water, momentum, speed, invariant) = rates;
    one_cell(
        kind,
        h,
        u,
        endpoint,
        port(water, momentum, speed, invariant)?,
    )
}

/// Minimal case at `endpoint` in which `scope` holds flow of `regime`.
fn regime_case(
    kind: Kind,
    endpoint: EndpointId,
    scope: RegimeScope,
    regime: Regime,
) -> Option<Outcome> {
    let fast = FAST * outward(endpoint);
    let stage = ChannelBoundary::Stage { elevation_m: 1.0 };
    let discharge = |q| ChannelBoundary::Discharge {
        discharge_m3_per_s: q,
    };
    let reservoir = ChannelBoundary::QuiescentReservoir {
        elevation_m: 1.0,
        entrance_loss_coefficient: 0.0,
    };
    let two = |depths: [f64; 2], velocities: [f64; 2], left, right| {
        step_outcome(reach(kind, &depths, &velocities), &interval(left, right))
    };
    Some(match (scope, regime) {
        (RegimeScope::Boundary(BoundaryKind::Stage), Regime::WetSubcritical) => {
            one_cell(kind, 1.0, 0.0, endpoint, stage)
        }
        (RegimeScope::Boundary(BoundaryKind::Stage), Regime::Supercritical) => {
            one_cell(kind, 1.0, fast, endpoint, stage)
        }
        (RegimeScope::Boundary(BoundaryKind::Stage), Regime::Dry) => {
            one_cell(kind, 0.0, 0.0, endpoint, stage)
        }
        (RegimeScope::Boundary(BoundaryKind::Discharge), Regime::WetSubcritical) => {
            one_cell(kind, 1.0, 0.0, endpoint, discharge(0.0))
        }
        (RegimeScope::Boundary(BoundaryKind::Discharge), Regime::Supercritical) => {
            one_cell(kind, 1.0, fast, endpoint, discharge(fast * kind.area(1.0)))
        }
        (RegimeScope::Boundary(BoundaryKind::Discharge), Regime::Dry) => {
            one_cell(kind, 0.0, 0.0, endpoint, discharge(0.0))
        }
        (RegimeScope::Boundary(BoundaryKind::QuiescentReservoir), Regime::WetSubcritical) => {
            one_cell(kind, 1.0, 0.0, endpoint, reservoir)
        }
        (RegimeScope::Boundary(BoundaryKind::QuiescentReservoir), Regime::Supercritical) => {
            one_cell(kind, 1.0, fast, endpoint, reservoir)
        }
        (RegimeScope::Boundary(BoundaryKind::QuiescentReservoir), Regime::Dry) => {
            one_cell(kind, 0.0, 0.0, endpoint, reservoir)
        }
        (RegimeScope::Boundary(BoundaryKind::Exterior), Regime::WetSubcritical) => {
            one_cell(kind, 1.0, 0.0, endpoint, exterior(1.0, 0.0))
        }
        // The exterior supplies supercritical inflow into resting water.
        (RegimeScope::Boundary(BoundaryKind::Exterior), Regime::Supercritical) => {
            one_cell(kind, 1.0, 0.0, endpoint, exterior(1.0, -fast))
        }
        (RegimeScope::Boundary(BoundaryKind::Exterior), Regime::Dry) => {
            one_cell(kind, 1.0, 0.0, endpoint, exterior(0.0, 0.0))
        }
        // Resting water against a port that supplies only the end cell's pressure.
        (RegimeScope::Boundary(BoundaryKind::FluxPort), Regime::WetSubcritical) => port_cell(
            kind,
            1.0,
            0.0,
            endpoint,
            (0.0, pressure(kind, 1.0), PORT_SPEED, Some(PORT_SPEED)),
        ),
        // Supercritical outflow at the end cell's own physical flux.
        (RegimeScope::Boundary(BoundaryKind::FluxPort), Regime::Supercritical) => {
            let q = FAST * kind.area(1.0);
            port_cell(
                kind,
                1.0,
                fast,
                endpoint,
                (
                    q,
                    q * FAST + pressure(kind, 1.0),
                    FAST + sound(kind, 1.0) + 1.0,
                    Some(0.0),
                ),
            )
        }
        // Outflow onto a dry exterior.
        (RegimeScope::Boundary(BoundaryKind::FluxPort), Regime::Dry) => port_cell(
            kind,
            1.0,
            0.0,
            endpoint,
            (0.125, pressure(kind, 1.0), PORT_SPEED, None),
        ),
        // A wet exterior fills a dry end cell.
        (RegimeScope::Boundary(BoundaryKind::FluxPort), Regime::WetDryFront) => port_cell(
            kind,
            0.0,
            0.0,
            endpoint,
            (-0.125, 0.0, PORT_SPEED, Some(PORT_SPEED)),
        ),
        (RegimeScope::Interior, Regime::WetSubcritical) => two(
            [1.0, 1.0],
            [0.0, 0.0],
            ChannelBoundary::Wall,
            ChannelBoundary::Wall,
        ),
        (RegimeScope::Interior, Regime::Supercritical) => two(
            [1.0, 1.0],
            [FAST, FAST],
            exterior(1.0, FAST),
            exterior(1.0, FAST),
        ),
        (RegimeScope::Interior, Regime::Dry) => two(
            [0.0, 0.0],
            [0.0, 0.0],
            ChannelBoundary::Wall,
            ChannelBoundary::Wall,
        ),
        (RegimeScope::Interior, Regime::WetDryFront) => two(
            [1.0, 0.0],
            [0.0, 0.0],
            ChannelBoundary::Wall,
            ChannelBoundary::Wall,
        ),
        _ => return None,
    })
}

/// Undeclared boundary regimes of every reach geometry that admission rejects.
const SCALAR_REJECTED_REGIMES: [(BoundaryKind, Regime); 6] = [
    (BoundaryKind::Stage, Regime::Supercritical),
    (BoundaryKind::Stage, Regime::Dry),
    (BoundaryKind::Discharge, Regime::Supercritical),
    (BoundaryKind::Discharge, Regime::Dry),
    (BoundaryKind::QuiescentReservoir, Regime::Supercritical),
    (BoundaryKind::QuiescentReservoir, Regime::Dry),
];

/// Cases, expected variants and message fragments for each `RejectedByAdmission` entry.
fn rejection_cases(kind: Kind, id: &str) -> Option<Vec<(Outcome, Variant, &'static str)>> {
    // Supercritical outflow leaves no incoming characteristic. Both endpoints are checked.
    let scalar = |boundary| {
        [
            (
                Regime::Supercritical,
                "incoming characteristic is unresolved",
            ),
            (Regime::Dry, "interior depth is dry"),
        ]
        .into_iter()
        .flat_map(|(regime, fragment)| {
            ENDPOINTS.into_iter().filter_map(move |endpoint| {
                let scope = RegimeScope::Boundary(boundary);
                regime_case(kind, endpoint, scope, regime)
                    .map(|outcome| (outcome, Variant::Numerical, fragment))
            })
        })
        .collect::<Vec<_>>()
    };
    Some(match id {
        "stage-outside-wet-subcritical" => scalar(BoundaryKind::Stage),
        "discharge-outside-wet-subcritical" => {
            let mut cases = scalar(BoundaryKind::Discharge);
            // 100 m³/s into resting unit-depth water lies beyond the subcritical branch.
            for endpoint in ENDPOINTS {
                let boundary = ChannelBoundary::Discharge {
                    discharge_m3_per_s: -100.0 * outward(endpoint),
                };
                cases.push((
                    one_cell(kind, 1.0, 0.0, endpoint, boundary),
                    Variant::Numerical,
                    "Supply a resolved wet subcritical target",
                ));
            }
            cases
        }
        "quiescent-reservoir-outside-wet-subcritical" => scalar(BoundaryKind::QuiescentReservoir),
        "flux-port-speed-below-interior" => ENDPOINTS
            .into_iter()
            .map(|endpoint| {
                (
                    port_cell(
                        kind,
                        1.0,
                        0.0,
                        endpoint,
                        (0.0, pressure(kind, 1.0), 1.0, Some(1.0)),
                    ),
                    Variant::Input,
                    "is below the end cell |u|+c",
                )
            })
            .collect(),
        "flux-port-dry-export" => ENDPOINTS
            .into_iter()
            .map(|endpoint| {
                (
                    port_cell(kind, 0.0, 0.0, endpoint, (0.125, 0.0, PORT_SPEED, None)),
                    Variant::Input,
                    "from a dry end cell",
                )
            })
            .collect(),
        "surveyed-channel-pool-attachment" => vec![(
            pool_outcome(kind),
            Variant::Input,
            "surveyed coordinator attachment is unsupported",
        )],
        _ => return None,
    })
}

#[test]
fn channel_reaches_admit_declared_boundaries() -> TestResult {
    for kind in KINDS {
        let capabilities = kind.capabilities();
        for id in ENDPOINTS {
            let endpoint = capabilities
                .endpoint(id)
                .ok_or(format!("{kind:?} declares no `{}`", id.id()))?;
            for admitted in endpoint.admitted {
                let boundary = samples()?
                    .into_iter()
                    .find(|sample| boundary_kind(sample) == *admitted)
                    .ok_or(format!(
                        "{kind:?}: no public variant for `{}`",
                        admitted.id()
                    ))?;
                let [left, right] = at(id, boundary);
                admits(reach(kind, &[1.0], &[0.0]), &interval(left, right)).map_err(|error| {
                    format!("{kind:?} `{}` `{}`: {error}", id.id(), admitted.id())
                })?;
            }
        }
    }
    Ok(())
}

#[test]
fn channel_reaches_reject_declared_rejections() -> TestResult {
    for kind in KINDS {
        let rejected = kind
            .capabilities()
            .unsupported
            .iter()
            .filter(|u| u.reason == UnsupportedReason::RejectedByAdmission);
        let mut count = 0;
        for entry in rejected {
            let cases = rejection_cases(kind, entry.id)
                .ok_or(format!("{kind:?}: no case for `{}`", entry.id))?;
            for (outcome, variant, fragment) in cases {
                let (found, reason) =
                    outcome?.ok_or(format!("{kind:?}: `{}` admitted", entry.id))?;
                assert_eq!(found, variant, "{kind:?} `{}`: {reason}", entry.id);
                assert!(
                    reason.contains(fragment),
                    "{kind:?} `{}`: `{reason}` lacks `{fragment}`",
                    entry.id
                );
                count += 1;
            }
        }
        assert!(count > 0, "{kind:?} declares no admission rejection");
    }
    // Control: the same pool connection admits rectangular and triangular reaches.
    for kind in [Kind::Rectangular, Kind::Triangular] {
        if let Some((_, reason)) = pool_outcome(kind)? {
            return Err(format!("{kind:?} pool connection: {reason}").into());
        }
    }
    Ok(())
}

#[test]
fn channel_reach_regimes_match_admission() -> TestResult {
    for kind in KINDS {
        let declared = kind.capabilities().regimes;
        for claim in declared {
            for endpoint in ENDPOINTS {
                let outcome = regime_case(kind, endpoint, claim.scope, claim.regime)
                    .ok_or(format!("{kind:?}: no case for {claim:?}"))?;
                if let Some((_, reason)) = outcome? {
                    return Err(format!("{kind:?} {claim:?} at {endpoint:?}: {reason}").into());
                }
            }
        }
        for (boundary, regime) in SCALAR_REJECTED_REGIMES {
            let scope = RegimeScope::Boundary(boundary);
            assert!(
                !declared.contains(&RegimeSupport { scope, regime }),
                "{kind:?} {boundary:?} claims {regime:?}"
            );
            for endpoint in ENDPOINTS {
                let outcome = regime_case(kind, endpoint, scope, regime)
                    .ok_or(format!("{kind:?}: no case for {boundary:?} {regime:?}"))?;
                assert!(
                    outcome?.is_some(),
                    "{kind:?} {boundary:?} admitted {regime:?}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn channel_reaches_admit_declared_sources() -> TestResult {
    for kind in KINDS {
        let declared = kind.capabilities().sources;
        let bound = definition(kind, 1, true);
        for binding in &bound.lateral_bindings {
            let source = source_kind(binding.direction);
            assert!(
                declared.contains(&source),
                "{kind:?} omits `{}`",
                source.id()
            );
        }
        assert_eq!(
            declared.len(),
            bound.lateral_bindings.len(),
            "{kind:?} sources"
        );
        for amount in [
            WaterAmount::Rate {
                rate_m3_per_s: 1e-3,
            },
            WaterAmount::IntervalVolume { volume_m3: 1e-6 },
        ] {
            let origin = Reach::new(
                crate::runtime::client(),
                definition(kind, 1, true),
                initial(kind, &[1.0], &[0.0]),
            )?;
            let mut forcing = interval(ChannelBoundary::Wall, ChannelBoundary::Wall);
            forcing.lateral_inputs = lateral_inputs(amount);
            let next = origin.step(END, &forcing)?.next;
            let receipt = next.last_step()?.ok_or("step recorded no receipt")?;
            assert_eq!(receipt.lateral_receipts.len(), 2, "{kind:?} {amount:?}");
        }
    }
    Ok(())
}

#[test]
fn channel_reach_options_are_admitted() -> TestResult {
    let triangular = Kind::Triangular.capabilities();
    let option = triangular
        .option("triangular-reconstruction")
        .ok_or("triangular reach declares no reconstruction option")?;
    assert_eq!(
        option.default,
        Some(reconstruction_id(
            ChannelConfig::default().triangular_reconstruction
        ))
    );
    assert_eq!(option.values.len(), RECONSTRUCTIONS.len());
    for value in option.values {
        let reconstruction = RECONSTRUCTIONS
            .into_iter()
            .find(|r| reconstruction_id(*r) == *value)
            .ok_or(format!("`{value}` names no reconstruction"))?;
        let mut layout = definition(Kind::Triangular, 2, false);
        layout.config.triangular_reconstruction = reconstruction;
        let case: Case = Reach::new(
            crate::runtime::client(),
            layout,
            initial(Kind::Triangular, &[1.0, 0.5], &[0.0, 0.0]),
        )
        .map_err(Into::into);
        admits(
            case,
            &interval(ChannelBoundary::Wall, ChannelBoundary::Wall),
        )
        .map_err(|error| format!("`{value}`: {error}"))?;
    }
    for kind in [Kind::Rectangular, Kind::Surveyed] {
        assert!(kind.capabilities().options.is_empty(), "{kind:?} options");
    }
    Ok(())
}
