//! Pool declarations checked against their admission code.

use crate::coupled_pool_fixture::orifice_case;
use crate::inflow_fixture::{binding, forcing, span};
use crate::level_pool_fixture::{element, input, single, unit};
use crate::pools_fixture::{
    Connected, Coupled, KINDS, Kind, Outcome, Pool, STEP, TestResult, Variant, channel_pool_case,
    channel_pool_outcome, channel_pool_step, construction_outcome, coupled_outcome,
    coupled_rejection, coupled_system, exchange_outcome, laws, operating_points, policy,
    surveyed_geometry,
};
use crate::pools_variants::{
    CHANNEL_POOL_ENDPOINTS, endpoint_id, inflow_kind, inflow_samples, lateral_kind,
    operation_fragment, structure_kind,
};
use hydr::capability::{
    BoundaryKind, EndpointId, Regime, RegimeScope, SourceKind, UnsupportedReason,
};
use hydr::channel::{ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput};
use hydr::coupling::{
    ChannelPoolEndpoint, ChannelPoolInitialState, CoupledLevelPoolForcingInterval,
    CoupledPoolInflow,
};
use hydr::forcing::WaterAmount;
use hydr::storage::{LevelPoolConfig, LevelPoolSources, LevelPoolState};

/// Asserts that `outcome` admits its case.
fn admitted(outcome: Outcome, case: &str) -> TestResult {
    match outcome? {
        None => Ok(()),
        Some((variant, reason)) => Err(format!("{case}: {variant:?} {reason}").into()),
    }
}

/// Coupled orifice case with one inflow boundary on pool `a` and one trial to `STEP`.
fn coupled_inflow_outcome(value: WaterAmount) -> Outcome {
    let (mut setup, initial) = orifice_case([4.0, 2.0])?;
    setup.external_inflows = vec![binding("in", "a")];
    let inflow = CoupledPoolInflow {
        boundary_id: "in".into(),
        input_id: "selected-input-in".into(),
        value,
    };
    // Integrated volumes cover the exact requested interval.
    let interval = match value {
        WaterAmount::Rate { .. } => forcing(vec![inflow]),
        WaterAmount::IntervalVolume { .. } => span(0.0, STEP, vec![inflow]),
    };
    let origin = Coupled::new(crate::runtime::client(), setup, initial)?;
    let next = match origin.trial_with_forcing_to(STEP, &interval) {
        Ok(trial) => trial.into_next(),
        Err(error) => return coupled_rejection(error),
    };
    let ledgers = next.external_inflows()?;
    assert!(
        ledgers.iter().all(|l| l.cumulative_volume_m3 > 0.0),
        "{value:?}"
    );
    Ok(None)
}

/// Coupled orifice case with local sources; returns cumulative rain and loss volumes.
fn coupled_source_volumes(rates: LevelPoolSources) -> Result<[f64; 3], Box<dyn std::error::Error>> {
    let (mut setup, initial) = orifice_case([4.0, 2.0])?;
    setup.source_policies = Some(vec![policy(); 2]);
    let interval = CoupledLevelPoolForcingInterval {
        start_s: 0.0,
        end_s: 2.0,
        sources: Some(vec![rates; 2]),
        inflows: vec![],
    };
    let next = Coupled::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(STEP, &interval)?
        .into_next();
    let ledger = next.pools().totals()?.sources.ok_or("no source counters")?;
    Ok([
        ledger.cumulative_rain_volume_m3,
        ledger.cumulative_evaporation_volume_m3,
        ledger.cumulative_bottom_loss_volume_m3,
    ])
}

/// Standalone pool with sources; returns cumulative rain and loss volumes.
fn pool_source_volumes(rates: LevelPoolSources) -> Result<[f64; 3], Box<dyn std::error::Error>> {
    let pool = Pool::new_with_sources(
        crate::runtime::client(),
        vec![element("pool", unit(16.0)?)],
        vec![LevelPoolState { volume_m3: 4.0 }],
        0.0,
        LevelPoolConfig::default(),
        vec![policy()],
    )?;
    let next = pool
        .trial_with_sources_to(1.0, &[input(0.0, 0.0)], &[rates])?
        .into_next();
    let ledger = next.totals()?.sources.ok_or("no source counters")?;
    Ok([
        ledger.cumulative_rain_volume_m3,
        ledger.cumulative_evaporation_volume_m3,
        ledger.cumulative_bottom_loss_volume_m3,
    ])
}

/// Rain, evaporation and bottom-loss rates that each move water in every fixture.
const RATES: LevelPoolSources = LevelPoolSources {
    rain_rate_m_per_s: 0.25,
    potential_evaporation_rate_m_per_s: 0.125,
    bottom_loss_rate_m3_per_s: 0.125,
};

/// Asserts that every declared local pool source moved water.
fn assert_sources(kind: Kind, volumes: [f64; 3]) {
    let [rain, evaporation, bottom] = volumes;
    for source in kind.capabilities().sources {
        match source {
            SourceKind::LocalPoolRain => assert!(rain > 0.0, "{kind:?} rain"),
            SourceKind::LocalPoolLoss => {
                assert!(evaporation > 0.0 && bottom > 0.0, "{kind:?} losses")
            }
            _ => {}
        }
    }
}

/// Channel/pool case whose reach binds one inflow and one withdrawal of `amount`.
fn channel_lateral_outcome(amount: WaterAmount) -> Outcome {
    let (mut setup, initial) = channel_pool_case(false, ChannelPoolEndpoint::Right)?;
    setup.reaches[0].definition.lateral_bindings = [
        ("in", ChannelLateralDirection::Inflow),
        ("out", ChannelLateralDirection::Withdrawal),
    ]
    .into_iter()
    .map(|(id, direction)| ChannelLateralBinding {
        boundary_id: id.into(),
        cell_id: "cell-0".into(),
        direction,
        ownership_assertion: format!("Independent synthetic lateral {id}"),
    })
    .collect();
    let mut held =
        crate::channel_pool_fixture::forcing(&setup, 0.0, crate::pools_fixture::CHANNEL_STEP);
    held.reaches[0].lateral_inputs = vec![
        ChannelLateralInput::Inflow {
            boundary_id: "in".into(),
            amount,
            longitudinal_velocity_m_per_s: 0.0,
        },
        ChannelLateralInput::Withdrawal {
            boundary_id: "out".into(),
            amount,
        },
    ];
    let origin = Connected::new(crate::runtime::client(), setup.clone(), initial.clone())?;
    let next = origin
        .trial_to(crate::pools_fixture::CHANNEL_STEP, &held)?
        .into_next();
    let ledger = next.ledger()?;
    for (direction, volume) in [
        (ChannelLateralDirection::Inflow, ledger.lateral_in_m3),
        (ChannelLateralDirection::Withdrawal, ledger.lateral_out_m3),
    ] {
        assert!(
            volume > 0.0,
            "{amount:?} `{}`",
            lateral_kind(direction).id()
        );
    }
    channel_pool_outcome(setup, initial, &held)
}

#[test]
fn pools_admit_declared_boundaries_and_sources() -> TestResult {
    let pool = Kind::LevelPool.capabilities();
    assert!(pool.endpoints.is_empty(), "level pool endpoints");
    admitted(
        exchange_outcome(&single(unit(4.0)?, 1.0)?, input(0.5, 0.25)),
        "exchange",
    )?;
    assert!(pool.sources.contains(&SourceKind::ExchangeVolume));
    assert_sources(Kind::LevelPool, pool_source_volumes(RATES)?);
    for claim in pool.regimes {
        assert_eq!(claim.scope, RegimeScope::Interior, "{claim:?}");
        let volume = match claim.regime {
            Regime::Dry => 0.0,
            Regime::WetSubcritical => 1.0,
            other => return Err(format!("level pool: no case for {other:?}").into()),
        };
        let batch = single(unit(4.0)?, volume)?;
        assert_eq!(batch.outputs()?[0].stage_m.is_some(), volume > 0.0);
        admitted(exchange_outcome(&batch, input(0.0, 0.0)), claim.regime.id())?;
    }

    let coupled = Kind::Coupled.capabilities();
    for endpoint in coupled.endpoints {
        for &boundary in endpoint.admitted {
            let outcome = match endpoint.id {
                EndpointId::PoolConnection => {
                    let (law, operating_point) = laws()?
                        .into_iter()
                        .flat_map(|law| operating_points().map(move |op| (law.clone(), op)))
                        .find(|(law, op)| structure_kind(law, *op) == Ok(boundary))
                        .ok_or(format!("no structure for `{}`", boundary.id()))?;
                    let (setup, initial) = coupled_system(law, operating_point, [4.0, 2.0])?;
                    coupled_outcome(setup, initial)
                }
                EndpointId::Lateral => {
                    let value = inflow_samples()
                        .into_iter()
                        .find(|value| inflow_kind(*value) == boundary)
                        .ok_or(format!("no inflow for `{}`", boundary.id()))?;
                    coupled_inflow_outcome(value)
                }
                other => return Err(format!("coupled pools: no case for {other:?}").into()),
            };
            admitted(outcome, boundary.id())?;
        }
    }
    assert_sources(Kind::Coupled, coupled_source_volumes(RATES)?);
    for claim in coupled.regimes {
        assert_eq!(claim.scope, RegimeScope::Interior, "{claim:?}");
        let (law, operating_point, volumes) = match claim.regime {
            // A stopped isolated pump leaves the receiving pool dry.
            Regime::Dry => (
                laws()?[1].clone(),
                hydr::structures::StructureOperatingPoint::PumpStoppedIsolated,
                [4.0, 0.0],
            ),
            Regime::WetSubcritical => (
                laws()?[2].clone(),
                hydr::structures::StructureOperatingPoint::Passive,
                [4.0, 2.0],
            ),
            other => return Err(format!("coupled pools: no case for {other:?}").into()),
        };
        let (setup, initial) = coupled_system(law, operating_point, volumes)?;
        admitted(coupled_outcome(setup, initial), claim.regime.id())?;
    }

    let channel_pools = Kind::ChannelPools.capabilities();
    for triangle in [false, true] {
        for endpoint in CHANNEL_POOL_ENDPOINTS {
            assert!(channel_pools.admits(endpoint_id(endpoint), BoundaryKind::PoolConnection));
            let (setup, initial) = channel_pool_case(triangle, endpoint)?;
            admitted(
                channel_pool_step(setup, initial),
                &format!("triangle={triangle} {endpoint:?}"),
            )?;
        }
    }
    for claim in channel_pools.regimes {
        assert!(
            matches!(
                claim.scope,
                RegimeScope::Interior | RegimeScope::Boundary(BoundaryKind::PoolConnection)
            ) && claim.regime == Regime::WetSubcritical,
            "{claim:?} has no case"
        );
    }
    for amount in [
        WaterAmount::Rate {
            rate_m3_per_s: 1e-3,
        },
        WaterAmount::IntervalVolume { volume_m3: 1e-4 },
    ] {
        admitted(channel_lateral_outcome(amount), &format!("{amount:?}"))?;
    }
    Ok(())
}

/// One rejection case: outcome, expected variant and message fragment.
type Rejection = (Outcome, Variant, &'static str);

/// Cases for each `RejectedByAdmission` entry, or `None` for an undeclared id.
fn rejection_cases(
    kind: Kind,
    id: &str,
) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    Ok(Some(match (kind, id) {
        (Kind::LevelPool, "curve-extrapolation") => vec![
            (
                exchange_outcome(&single(unit(4.0)?, 1.0)?, input(8.0, 0.0)),
                Variant::Numerical,
                "candidate volume outside closed curve support",
            ),
            (
                construction_outcome(Pool::new(
                    crate::runtime::client(),
                    vec![element("pool", unit(4.0)?)],
                    vec![LevelPoolState { volume_m3: 8.0 }],
                    0.0,
                    LevelPoolConfig::default(),
                )),
                Variant::Input,
                "supply explicit finite closed-range storage",
            ),
        ],
        (Kind::LevelPool, "absorbed-positive-exchange") => vec![
            (
                exchange_outcome(
                    &single(unit(2.0_f64.powi(54))?, 2.0_f64.powi(53))?,
                    input(1.0, 0.0),
                ),
                Variant::Numerical,
                "positive incoming volume is absorbed into unchanged storage",
            ),
            (
                exchange_outcome(
                    &single(unit(2.0_f64.powi(55))?, 2.0_f64.powi(54))?,
                    input(0.0, 1.0),
                ),
                Variant::Numerical,
                "positive outgoing volume is absorbed into unchanged storage",
            ),
        ],
        (Kind::LevelPool, "donor-overdraft") => vec![(
            exchange_outcome(&single(unit(16.0)?, 1.0)?, input(10.0, 1.1)),
            Variant::Numerical,
            "outgoing volume exceeds initial donor storage without incoming credit",
        )],
        (Kind::Coupled, "implicit-pump-operation" | "mismatched-structure-operation") => {
            let mut cases = Vec::new();
            for law in laws()? {
                for operating_point in operating_points() {
                    if structure_kind(&law, operating_point) == Err(id) {
                        let fragment = operation_fragment(&law);
                        let (setup, initial) =
                            coupled_system(law.clone(), operating_point, [4.0, 2.0])?;
                        cases.push((
                            coupled_outcome(setup, initial),
                            Variant::Operation,
                            fragment,
                        ));
                    }
                }
            }
            cases
        }
        (Kind::ChannelPools, "surveyed-reach-attachment") => {
            let (mut setup, initial) = channel_pool_case(false, ChannelPoolEndpoint::Right)?;
            setup.reaches[0].definition.geometry = surveyed_geometry(&setup.vertical_datum);
            vec![(
                channel_pool_step(setup, initial),
                Variant::Input,
                "surveyed coordinator attachment is unsupported",
            )]
        }
        (Kind::ChannelPools, "dry-connections") => {
            let mut cases = Vec::new();
            for triangle in [false, true] {
                let dry_pool = channel_edge(triangle, |initial| initial.pools[0].volume_m3 = 0.0);
                let dry_reach = channel_edge(triangle, |initial| {
                    initial.reaches[0].volume_m3 = vec![0.0];
                });
                cases.push((
                    dry_pool,
                    Variant::Unresolved,
                    "reservoir depth bracket is empty",
                ));
                cases.push((dry_reach, Variant::Unresolved, "interior depth is dry"));
            }
            cases
        }
        (Kind::ChannelPools, "supercritical-connections") => {
            let mut cases = Vec::new();
            // Velocity `±FAST` at unit depth exceeds every reference case's wave speed.
            for (velocity, fragment) in [
                (FAST, "incoming characteristic is unresolved"),
                (-FAST, "outgoing characteristic is unresolved"),
            ] {
                for triangle in [false, true] {
                    let outcome = channel_edge(triangle, |initial| {
                        let reach = &mut initial.reaches[0];
                        reach.longitudinal_momentum_m4_per_s =
                            reach.volume_m3.iter().map(|v| velocity * v).collect();
                    });
                    cases.push((outcome, Variant::Unresolved, fragment));
                }
            }
            cases
        }
        _ => return Ok(None),
    }))
}

/// Outflow speed in m/s toward the right-end pool, supercritical in every reference case.
const FAST: f64 = 5.0;

/// Resting reference case connected at its right end, with `edit` applied to the initial state.
fn channel_edge(triangle: bool, edit: impl FnOnce(&mut ChannelPoolInitialState)) -> Outcome {
    let (setup, mut initial) = channel_pool_case(triangle, ChannelPoolEndpoint::Right)?;
    edit(&mut initial);
    channel_pool_step(setup, initial)
}

#[test]
fn pools_reject_declared_rejections() -> TestResult {
    for kind in KINDS {
        let mut count = 0;
        for entry in kind
            .capabilities()
            .unsupported
            .iter()
            .filter(|u| u.reason == UnsupportedReason::RejectedByAdmission)
        {
            let cases = rejection_cases(kind, entry.id)?
                .ok_or(format!("{kind:?}: no case for `{}`", entry.id))?;
            assert!(!cases.is_empty(), "{kind:?} `{}` has no case", entry.id);
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
    Ok(())
}
