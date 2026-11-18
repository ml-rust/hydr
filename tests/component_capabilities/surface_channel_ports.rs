//! Surface/channel reach-end port declaration checked against port runs and admission.

use crate::surface_channel_support::fixture::Result;
use crate::surface_channel_support::ports_fixture::{
    ALPHA, BETA, Batch, CELLS, GRAVITY, REACH_WIDTH, Reach, Scenario, east, forcing, port,
    rectangular, step, still, west, x_face, y_face,
};
use crate::surface_channels_fixture::{Outcome, construction_outcome, trial_outcome};
use crate::surface_channels_variants::{Variant, port_regime};
use hydr::capability::{BoundaryKind, EndpointId, Regime, RegimeScope};
use hydr::channel::{
    ChannelBoundary, ChannelEnd, ChannelGeometry, ChannelReachDefinition, ChannelSurveyedConfig,
    CrossSectionDefinition, CrossSectionPoint, SurveyedChannelCell, TriangularChannelCell,
};
use hydr::coupling::{
    SurfaceChannelPort, SurfaceChannelPortTransfer, SurfaceChannelSetup,
    surface_channel_capabilities,
};
use hydr::model::CellState;

/// The only port receipt of the last interval of `batch`.
fn transfer(batch: &Batch) -> Result<SurfaceChannelPortTransfer> {
    let step = batch.last_step()?.ok_or("an accepted interval")?;
    Ok(step.ports.first().ok_or("the port receipt")?.clone())
}

/// A reach above a still floodplain, draining through `port` with no bank link.
fn draining(port: SurfaceChannelPort) -> Scenario {
    let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![port]);
    case.link = false;
    case.alpha = Reach::rest(0.5, 1.8);
    case
}

/// Every surface cell at depth `h` with discharge per width `[qx, qy]`.
fn moving(h: f64, [qx, qy]: [f64; 2]) -> Vec<CellState> {
    vec![CellState { h, qx, qy }; CELLS]
}

/// `alpha` in state `alpha` against `surface` through the east port, on a flat 1 m bed.
fn east_case(alpha: Reach, surface: Vec<CellState>) -> Scenario {
    let mut case = Scenario::rest(1.0, 1.5, 1.0, vec![east("p")]);
    case.surface = surface;
    case.alpha = alpha;
    case.link = false;
    case
}

/// Receipts of one accepted interval of a case that holds each declared port regime.
fn regime_receipts(regime: Regime) -> Result<Vec<SurfaceChannelPortTransfer>> {
    let one = |case: Scenario, dt: f64| -> Result<Vec<SurfaceChannelPortTransfer>> {
        Ok(vec![transfer(&step(&case.build()?, dt)?)?])
    };
    let h = 0.5;
    let q = h * 2.0 * (GRAVITY * h).sqrt();
    Ok(match regime {
        Regime::WetSubcritical => one(Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]), 0.1)?,
        Regime::Supercritical => {
            let mut both = one(
                east_case(Reach::uniform(1.0, h, q), moving(h, [-q, 0.0])),
                0.01,
            )?;
            both.extend(one(
                east_case(Reach::uniform(1.0, h, -q), moving(h, [q, 0.0])),
                0.01,
            )?);
            both
        }
        Regime::Critical => one(
            east_case(Reach::uniform(1.0, 1.0, 1.0), moving(0.1, [-0.2, 0.0])),
            0.01,
        )?,
        Regime::Dry => {
            let mut case = Scenario::rest(1.0, 0.8, 0.5, vec![west("p")]);
            case.link = false;
            one(case, 0.1)?
        }
        Regime::WetDryFront => one(
            east_case(Reach::uniform(1.0, 0.5, 0.0), vec![still(0.0); CELLS]),
            0.01,
        )?,
        Regime::ReversalInBranch => {
            // The reach drains through the port and the bank link, then refills.
            let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
            case.alpha = Reach::rest(0.5, 1.75);
            let mut batch = case.build()?;
            let mut run = Vec::new();
            for _ in 0..120 {
                batch = step(&batch, 0.25)?;
                run.push(transfer(&batch)?);
            }
            run
        }
        other => return Err(format!("surface/channel port: no case for {other:?}").into()),
    })
}

#[test]
fn surface_channel_ports_admit_declared_endpoints_and_regimes() -> Result {
    let declared = surface_channel_capabilities();
    let mut ported = 0;
    for endpoint in declared.endpoints {
        if !endpoint.admits(BoundaryKind::FluxPort) {
            continue;
        }
        ported += 1;
        let moved = match endpoint.id {
            EndpointId::ReachLeft | EndpointId::ReachRight => {
                let case = match endpoint.id {
                    EndpointId::ReachLeft => draining(west("p")),
                    _ => draining(east("p")),
                };
                let batch = step(&case.build()?, 0.1)?;
                let reach = batch.reach(ALPHA).ok_or("alpha")?.ledger();
                // The port end carries the reach boundary water, with no caller policy there.
                transfer(&batch)?.signed_volume_m3 < 0.0 && reach.boundary_out_m3 > 0.0
            }
            EndpointId::ExteriorFace => {
                let batch = step(&draining(west("p")).build()?, 0.1)?;
                let surface = batch.surface().last_step()?.ok_or("a surface receipt")?;
                let port_faces = [x_face(0, 0), x_face(0, 1)];
                // Grid-edge faces carry the port water into the surface exchange totals.
                surface
                    .faces
                    .iter()
                    .filter(|face| port_faces.contains(&face.face))
                    .all(|face| face.volume.abs() > 0.0)
                    && batch.surface().ledger().exchange_in_m3 > 0.0
            }
            other => return Err(format!("surface/channel port: no case for {other:?}").into()),
        };
        assert!(moved, "{:?} moved no water", endpoint.id);
    }
    assert_eq!(
        ported, 3,
        "ports attach at both reach ends and exterior faces"
    );
    let mut claims = 0;
    for claim in declared.regimes {
        if claim.scope != RegimeScope::Boundary(BoundaryKind::FluxPort) {
            continue;
        }
        claims += 1;
        let receipts = regime_receipts(claim.regime)?;
        let held = if claim.regime == Regime::ReversalInBranch {
            receipts.iter().any(|t| t.reversed)
        } else {
            receipts
                .iter()
                .all(|t| port_regime(t.regime) == claim.regime)
        };
        assert!(held, "{:?}: {receipts:?}", claim.regime);
    }
    assert_eq!(claims, 6);
    assert!(declared.unsupported_by_id("reach-end-port").is_none());
    Ok(())
}

#[test]
fn surface_channel_port_limits_hold_in_port_runs() -> Result {
    let declared = surface_channel_capabilities();
    for id in [
        "port-tangential-closure",
        "port-bed-step",
        "port-explicit-flux",
    ] {
        assert!(declared.limit_by_id(id).is_some(), "`{id}` is undeclared");
    }
    // port-explicit-flux: the start states fix the rates, whatever the duration.
    let origin = draining(west("p")).build()?;
    let rate = |dt: f64| -> Result<(f64, SurfaceChannelPortTransfer)> {
        let next = step(&origin, dt)?;
        let receipt = next.last_step()?.ok_or("an accepted interval")?;
        assert_eq!(receipt.retries, 0, "{receipt:?}");
        let transfer = transfer(&next)?;
        Ok((transfer.signed_volume_m3 / next.time(), transfer))
    };
    let (short_rate, short) = rate(0.01)?;
    let (long_rate, long) = rate(0.02)?;
    assert_eq!(
        short.signal_speed_m_per_s.to_bits(),
        long.signal_speed_m_per_s.to_bits()
    );
    assert!((short_rate - long_rate).abs() <= 4.0 * f64::EPSILON * short_rate.abs());
    // port-tangential-closure: reach water enters the surface without tangential momentum.
    assert_eq!(
        short.surface_momentum_impulse_m4_per_s[1].to_bits(),
        0.0f64.to_bits()
    );
    // Tangential surface momentum leaving through the port ends at the junction.
    let mut inflow = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
    inflow.link = false;
    inflow.alpha = Reach::rest(0.5, 1.4);
    inflow.surface = moving(0.6, [-0.1, 0.2]);
    let entering = transfer(&step(&inflow.build()?, 0.01)?)?;
    assert!(entering.signed_volume_m3 > 0.0, "{entering:?}");
    assert!(
        entering.junction_reaction_m4_per_s[1] != 0.0,
        "{entering:?}"
    );
    // port-bed-step: a lake at rest stays at rest across a step, which the junction reacts.
    let mut batch = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]).build()?;
    for _ in 0..10 {
        batch = step(&batch, 0.1)?;
    }
    let rest = transfer(&batch)?;
    assert!(rest.junction_reaction_m4_per_s[0].abs() > 0.0, "{rest:?}");
    for cell in batch.surface().state()?.cells() {
        assert!(cell.qx.abs() <= 1e-12 && cell.qy.abs() <= 1e-12, "{cell:?}");
    }
    Ok(())
}

/// A `beta` setup that rejects one port: its geometry or width replaced by `edit`.
fn beta_edited(
    edit: impl FnOnce(&mut SurfaceChannelSetup),
    ports: Vec<SurfaceChannelPort>,
) -> Outcome {
    let (mut setup, initial) = Scenario::rest(1.0, 1.1, 0.5, Vec::new()).parts()?;
    edit(&mut setup);
    setup.ports = ports;
    construction_outcome(setup, initial)
}

/// A caller policy at a ported end, or none at an unported end, for one interval.
fn endpoint_policy(owned: bool) -> Outcome {
    let batch = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]).build()?;
    let mut held = forcing(batch.setup(), 0.0, 0.1);
    if owned {
        held.reaches[0].left = Some(ChannelBoundary::Wall);
    } else {
        held.reaches[1].left = None;
    }
    trial_outcome(&batch, 0.1, &held)
}

/// One rejection case: outcome, expected variant and message fragment.
type Rejection = (Outcome, Variant, &'static str);

/// Cases for each port `RejectedByAdmission` entry, or `None` for an undeclared id.
pub fn rejection_cases(id: &str) -> Option<Vec<Rejection>> {
    let beta_right = || {
        vec![port(
            "p",
            BETA,
            ChannelEnd::Right,
            vec![y_face(0, 0), y_face(1, 0)],
        )]
    };
    Some(match id {
        "port-section-mismatch" => vec![(
            beta_edited(
                |s| {
                    s.reaches[1].definition.geometry = ChannelGeometry::PrismaticTriangular {
                        left_bank_slope_h_per_v: 1.0,
                        right_bank_slope_h_per_v: 1.0,
                        bed_elevation_m: 0.25,
                        cells: vec![TriangularChannelCell {
                            id: "cell-0".into(),
                            length_m: 2.0,
                            maximum_depth_m: 4.0,
                            manning_n: 0.0,
                        }],
                    };
                },
                beta_right(),
            ),
            Variant::Input,
            "a port needs a rectangular end cell of the face width",
        )],
        "surveyed-port-end" => vec![(
            beta_edited(|s| s.reaches[1].definition = surveyed(0.25), beta_right()),
            Variant::Input,
            "surveyed reaches couple through bank links only",
        )],
        "port-width-mismatch" => vec![(
            beta_edited(
                |s| s.reaches[1].definition = rectangular(3, 0.25),
                vec![port("p", BETA, ChannelEnd::Right, vec![y_face(0, 0)])],
            ),
            Variant::Input,
            "set the end cell width to the face width sum",
        )],
        "port-face-ownership" => vec![
            (
                beta_edited(
                    |_| {},
                    vec![port(
                        "p",
                        BETA,
                        ChannelEnd::Right,
                        vec![x_face(1, 0), x_face(1, 1)],
                    )],
                ),
                Variant::Input,
                "has two active cells",
            ),
            (
                beta_edited(
                    |_| {},
                    vec![
                        port(
                            "p",
                            BETA,
                            ChannelEnd::Right,
                            vec![y_face(0, 0), y_face(1, 0)],
                        ),
                        port(
                            "q",
                            BETA,
                            ChannelEnd::Right,
                            vec![y_face(2, 0), y_face(3, 0)],
                        ),
                    ],
                ),
                Variant::Input,
                "bind one port per reach end",
            ),
        ],
        "port-endpoint-policy" => vec![
            (
                endpoint_policy(true),
                Variant::Input,
                "supply no caller policy at a ported end",
            ),
            (
                endpoint_policy(false),
                Variant::Input,
                "supply an endpoint policy",
            ),
        ],
        _ => return None,
    })
}

/// Three surveyed cells of one rectangular section of the port width on `bed`.
fn surveyed(bed: f64) -> ChannelReachDefinition {
    let rectangular = rectangular(3, bed);
    ChannelReachDefinition {
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile: CrossSectionDefinition {
                id: "section".into(),
                vertical_datum: rectangular.vertical_datum.clone(),
                points: [
                    (0.0, 4.0),
                    (0.0, bed),
                    (REACH_WIDTH, bed),
                    (REACH_WIDTH, 4.0),
                ]
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
                maximum_stage_m: 4.0,
                profile_provenance: "Analytical rectangular survey".into(),
            },
            cells: (0..3)
                .map(|i| SurveyedChannelCell {
                    id: format!("cell-{i}"),
                    length_m: 2.0,
                    manning_n: 0.0,
                })
                .collect(),
            numerical: ChannelSurveyedConfig::default(),
        },
        ..rectangular
    }
}
