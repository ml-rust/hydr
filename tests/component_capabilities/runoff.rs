//! Runoff declarations checked against their admission code.

use crate::land_fixture::domain;
use crate::pools_fixture::policy;
use crate::reservoir_fixture::single as reservoir;
use crate::runoff_fixture::{
    KINDS, Kind, Outcome, Reservoir, RunoffPools, Soil, Subcatchment, TestResult, Variant,
    admitted, built, drainage_branch, drainage_samples, reservoir_outcome, reservoir_variant,
    runoff_pool_outcome, soil_input, soil_outcome, soil_parameters, soil_variant,
    subcatchment_input, subcatchment_outcome, subcatchment_variant,
};
use crate::runoff_pools::fixture::{
    exact_case, forcing, impervious, inactive_pools, setup, shortened_case,
};
use crate::runoff_variants::{ROUTINGS, routing_id};
use crate::subcatchment_fixture::{geometry, parameters, single as subcatchment};
use hydr::capability::{BoundaryKind, ComponentId, EndpointId, SourceKind, UnsupportedReason};
use hydr::runoff::{
    ReservoirConfig, ReservoirInput, ReservoirParameters, SubcatchmentConfig,
    SubcatchmentDefinition, SubcatchmentParameters, SubcatchmentRouting, SubcatchmentState,
};
use hydr::soil::{LandCell, LandModel, ModifiedGreenAmptParameters, SoilCellState, SoilState};
use hydr::storage::LevelPoolSources;
use std::collections::BTreeSet;

/// Rain rate of 1/1024 ft/s in m/s.
const RAIN: f64 = 0.3048 / 1024.0;

/// Asserts that the declared sources of `kind` equal the sources a case moved water through.
fn assert_sources(kind: Kind, covered: &[SourceKind]) {
    let declared: BTreeSet<_> = kind.capabilities().sources.iter().map(|s| s.id()).collect();
    let covered: BTreeSet<_> = covered.iter().map(|s| s.id()).collect();
    assert_eq!(declared, covered, "{kind:?} sources");
}

/// Runs every drainage branch through the reservoir outlet.
fn reservoir_branches() -> TestResult {
    let capabilities = Kind::Reservoir.capabilities();
    assert!(capabilities.admits(EndpointId::Outlet, BoundaryKind::RunoffOutletNoFeedback));
    let limit = capabilities
        .limit_by_id("parameter-selected-drainage")
        .ok_or("reservoir declares no drainage limit")?;
    for parameters in drainage_samples()? {
        let branch = drainage_branch(parameters)?;
        let batch = reservoir(parameters, 0.0, ReservoirConfig::default())?;
        let input = [ReservoirInput {
            incoming_rate_m_per_s: RAIN,
        }];
        let trial = batch.trial_to(256.0, &input)?;
        let output = &trial.outputs()?[0];
        let (exports, discharge) = match branch {
            "nonlinear" => (true, true),
            "closed" => (false, true),
            "instantaneous-overflow" => (true, false),
            other => return Err(format!("reservoir: no case for `{other}`").into()),
        };
        assert_eq!(output.export_volume_m3 > 0.0, exports, "{branch} export");
        assert_eq!(
            output.endpoint_discharge_m3_per_s.is_some(),
            discharge,
            "{branch} discharge"
        );
        let words = branch.replace('-', " ");
        assert!(limit.summary.contains(&words), "limit omits `{words}`");
    }
    Ok(())
}

/// Runs every declared routing value and source through one pervious subcatchment.
fn subcatchment_sources() -> TestResult {
    let option = Kind::Subcatchment
        .capabilities()
        .option("subcatchment-routing")
        .ok_or("subcatchment declares no routing option")?;
    assert_eq!(option.default, None, "routing has no default");
    let mut covered = Vec::new();
    for value in option.values {
        let routing = ROUTINGS
            .into_iter()
            .find(|r| routing_id(*r) == *value)
            .ok_or(format!("`{value}` names no routing"))?;
        let p = parameters(0.5, 0.5, routing)?;
        let mut state = SubcatchmentState::initialized(&p)?;
        state.depths_m = [0.125 * 0.3048, 0.25 * 0.3048, 0.125 * 0.3048];
        let batch = subcatchment(p, state, SubcatchmentConfig::default())?;
        let trial = batch.trial_to(64.0, &[subcatchment_input(RAIN, 1e-6, [0.0; 3])])?;
        let step = trial.diagnostics()?;
        if routing == SubcatchmentRouting::Direct {
            assert!(step.transfers.is_empty(), "direct routing transfers water");
        }
        let ledger = trial.into_next().ledger().clone();
        assert!(ledger.outlet_volume_m3 > 0.0, "{routing:?} outlet");
        for (source, volume) in [
            (SourceKind::Rain, ledger.applied_rain_volume_m3),
            (SourceKind::ModifiedGreenAmpt, ledger.infiltration_volume_m3),
            (SourceKind::Evaporation, ledger.evaporation_volume_m3),
        ] {
            assert!(volume > 0.0, "{routing:?} `{}`", source.id());
            covered.push(source);
        }
    }
    // Impervious land exports rain without a soil call.
    let p = parameters(1.0, 1.0, SubcatchmentRouting::Direct)?;
    let state = SubcatchmentState::initialized(&p)?;
    let next = subcatchment(p, state, SubcatchmentConfig::default())?
        .trial_to(64.0, &[subcatchment_input(RAIN, 0.0, [0.0; 3])])?
        .into_next();
    assert!(next.states()?[0].soil.is_none());
    assert_eq!(next.ledger().infiltration_volume_m3, 0.0);
    assert!(next.ledger().outlet_volume_m3 > 0.0);
    covered.push(SourceKind::ImperviousLand);
    assert_sources(Kind::Subcatchment, &covered);
    Ok(())
}

/// Runs runoff sources, pool sources and the paired outlet through the coordinator.
fn runoff_pool_sources() -> TestResult {
    let capabilities = Kind::RunoffPools.capabilities();
    assert!(capabilities.admits(EndpointId::Outlet, BoundaryKind::RunoffOutletNoFeedback));
    let mut covered = Vec::new();

    let (exact, initial) = exact_case()?;
    let next = RunoffPools::new(crate::runtime::client(), exact, initial)?
        .trial_with_forcing_to(2.0, &forcing(1, 0.25, 0.0))?
        .into_next();
    let step = next.last_step()?.ok_or("missing exact receipt")?;
    assert!(!step.outlets.is_empty(), "no paired outlet");
    assert!(
        step.balance.internal_outlet_volume_m3 > 0.0,
        "no outlet volume"
    );
    assert!(step.balance.rain_volume_m3 > 0.0, "no rain");
    assert_eq!(step.balance.infiltration_volume_m3, 0.0);
    covered.extend([SourceKind::Rain, SourceKind::ImperviousLand]);

    let (pervious, initial, mut losses) = shortened_case()?;
    losses.subcatchments[0].potential_evaporation_m_per_s = 1e-6;
    let next = RunoffPools::new(crate::runtime::client(), pervious, initial)?
        .trial_with_forcing_to(1.0, &losses)?
        .into_next();
    let balance = next.last_step()?.ok_or("missing pervious receipt")?.balance;
    assert!(balance.infiltration_volume_m3 > 0.0, "no infiltration");
    assert!(balance.evaporation_volume_m3 > 0.0, "no evaporation");
    covered.extend([SourceKind::ModifiedGreenAmpt, SourceKind::Evaporation]);

    let (mut pools, states) = inactive_pools([0.0, 1.0])?;
    pools.config.maximum_step_s = 2.0;
    pools.source_policies = Some(vec![policy(); 2]);
    let (sourced, initial) = setup(vec![impervious(2.0, 1.0)?], pools, states)?;
    let mut local = forcing(1, 0.25, 0.0);
    let none = LevelPoolSources {
        rain_rate_m_per_s: 0.0,
        potential_evaporation_rate_m_per_s: 0.0,
        bottom_loss_rate_m3_per_s: 0.0,
    };
    let rates = LevelPoolSources {
        rain_rate_m_per_s: 0.25,
        potential_evaporation_rate_m_per_s: 0.125,
        bottom_loss_rate_m3_per_s: 0.125,
    };
    local.pool_sources = Some(vec![none, rates]);
    let next = RunoffPools::new(crate::runtime::client(), sourced, initial)?
        .trial_with_forcing_to(2.0, &local)?
        .into_next();
    let balance = next
        .last_step()?
        .ok_or("missing pool-source receipt")?
        .balance;
    let runoff = next.runoff().ledger();
    assert!(
        balance.rain_volume_m3 > runoff.applied_rain_volume_m3,
        "no pool rain"
    );
    assert!(balance.bottom_loss_volume_m3 > 0.0, "no bottom loss");
    assert!(
        balance.evaporation_volume_m3 > runoff.evaporation_volume_m3,
        "no pool evaporation"
    );
    covered.extend([SourceKind::LocalPoolRain, SourceKind::LocalPoolLoss]);
    assert_sources(Kind::RunoffPools, &covered);
    Ok(())
}

#[test]
fn runoff_components_admit_declared_options_and_sources() -> TestResult {
    for kind in KINDS {
        let capabilities = kind.capabilities();
        if kind != Kind::Subcatchment {
            assert!(capabilities.options.is_empty(), "{kind:?} options");
        }
        assert!(capabilities.regimes.is_empty(), "{kind:?} regimes");
    }
    assert_eq!(Kind::Subcatchment.capabilities().options.len(), 1);
    assert!(Kind::Soil.capabilities().endpoints.is_empty());
    let batch = Soil::initialized(crate::runtime::client(), &[soil_parameters()?])?;
    admitted(soil_outcome(&batch, 60.0, soil_input(1e-3)), "soil")?;
    assert_sources(Kind::Soil, &[SourceKind::ModifiedGreenAmpt]);

    reservoir_branches()?;
    assert_sources(Kind::Reservoir, &[]);
    let p = drainage_samples()?[0];
    admitted(
        reservoir_outcome(
            &reservoir(p, 0.25 * 0.3048, ReservoirConfig::default())?,
            0.0,
        ),
        "reservoir recession",
    )?;

    let subcatchment_outlet = Kind::Subcatchment.capabilities();
    assert!(subcatchment_outlet.admits(EndpointId::Outlet, BoundaryKind::RunoffOutletNoFeedback));
    subcatchment_sources()?;
    runoff_pool_sources()
}

/// One rejection case: outcome, expected variant and message fragment.
type Rejection = (Outcome, Variant, &'static str);

/// Cases for each `RejectedByAdmission` entry, or `None` for an undeclared id.
fn rejection_cases(
    kind: Kind,
    id: &str,
) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    let soil = soil_parameters()?;
    // Subcatchment fixture soil for positive pervious support.
    let pervious = ModifiedGreenAmptParameters::from_native(0.125, 1.0 / 1024.0, 1.0)?;
    Ok(Some(match (kind, id) {
        (Kind::Soil, "unrepresentable-derived-parameters") => vec![(
            built(
                ModifiedGreenAmptParameters::from_native(0.4, 1e-5, f64::from_bits(1)),
                soil_variant,
            ),
            Variant::Input,
            "derived parameter or unit bridge is unrepresentable",
        )],
        (Kind::Soil, "inactive-pervious-cell") => vec![(
            built(
                LandModel::new(
                    &domain(2, 1, vec![true, false])?,
                    vec![LandCell::ModifiedGreenAmpt(soil); 2],
                ),
                soil_variant,
            ),
            Variant::Input,
            "inactive cells require explicit impervious bookkeeping",
        )],
        (Kind::Soil, "mismatched-land-state") => {
            let land = LandModel::new(
                &domain(1, 1, vec![true])?,
                vec![LandCell::ModifiedGreenAmpt(soil)],
            )?;
            vec![(
                built(
                    SoilState::new(&land, vec![SoilCellState::Impervious]),
                    soil_variant,
                ),
                Variant::Input,
                "soil variant must match the explicit land variant",
            )]
        }
        (Kind::Soil, "collapsed-positive-infiltration") => {
            let batch = Soil::initialized(crate::runtime::client(), &[soil])?;
            vec![(
                soil_outcome(&batch, f64::from_bits(1), soil_input(1e-10)),
                Variant::Numerical,
                "Positive infiltration became zero",
            )]
        }
        (Kind::Reservoir, "ambiguous-closure") => [(0.0, 1.0 / 64.0), (32.0, 0.0)]
            .into_iter()
            .map(|(width, slope)| {
                let result =
                    ReservoirParameters::from_native(1024.0, 1024.0, width, slope, 0.0, 0.125);
                (
                    built(result, reservoir_variant),
                    Variant::Input,
                    "zero roughness requires positive width and slope",
                )
            })
            .collect(),
        (Kind::Reservoir, "absorbed-drainage-excess") => {
            let depression = 1e6_f64;
            let p = ReservoirParameters::from_si(1.0, 1.0, 1.0, 1.0, 1e-8, depression)?;
            let batch: Reservoir = reservoir(p, depression.next_up(), ReservoirConfig::default())?;
            vec![(
                reservoir_outcome(&batch, 0.0),
                Variant::Numerical,
                "represented implicit excess",
            )]
        }
        (Kind::Subcatchment, "soil-support-mismatch") => [(1.0, Some(pervious)), (0.0, None)]
            .into_iter()
            .map(|(impervious, soil)| {
                let result = SubcatchmentParameters::new(
                    geometry(impervious, 0.5),
                    soil,
                    SubcatchmentRouting::Direct,
                );
                (
                    built(result, subcatchment_variant),
                    Variant::Input,
                    "soil parameters must be present exactly when pervious area is positive",
                )
            })
            .collect(),
        (Kind::Subcatchment, "routing-without-receiver") => [
            (
                1.0,
                None,
                SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
            ),
            (
                0.0,
                Some(pervious),
                SubcatchmentRouting::PerviousToImpervious { fraction: 0.5 },
            ),
        ]
        .into_iter()
        .map(|(impervious, soil, routing)| {
            let result = SubcatchmentParameters::new(geometry(impervious, 0.5), soil, routing);
            (
                built(result, subcatchment_variant),
                Variant::Input,
                "positive internal routing requires a positive-area destination",
            )
        })
        .collect(),
        (Kind::Subcatchment, "runon-to-absent-subarea") => {
            let p = parameters(1.0, 0.0, SubcatchmentRouting::Direct)?;
            let state = SubcatchmentState::initialized(&p)?;
            let batch = subcatchment(p, state, SubcatchmentConfig::default())?;
            vec![(
                subcatchment_outcome(&batch, 1.0, subcatchment_input(0.0, 0.0, [0.0, 0.0, 1.0])),
                Variant::Input,
                "supply finite nonnegative volume over positive support",
            )]
        }
        (Kind::RunoffPools, "unmapped-outlet" | "duplicate-outlet") => {
            let (mut changed, initial) = exact_case()?;
            let fragment = if id == "unmapped-outlet" {
                changed.outlets.clear();
                "pair every potential native outlet and every receiving boundary exactly once"
            } else {
                changed.outlets.push(changed.outlets[0].clone());
                "outlet bindings require unique potential native lanes and unique known receiving boundaries"
            };
            vec![(
                runoff_pool_outcome(changed, initial, 2.0, &forcing(1, 0.25, 0.0)),
                Variant::Input,
                fragment,
            )]
        }
        _ => return Ok(None),
    }))
}

#[test]
fn runoff_components_reject_declared_rejections() -> TestResult {
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

#[test]
fn runoff_parents_run_nested_admission() -> TestResult {
    let soil = Kind::Soil.capabilities();
    for parent in [
        hydr::solver::surface_solver_capabilities(),
        Kind::Subcatchment.capabilities(),
        hydr::coupling::surface_channel_capabilities(),
    ] {
        assert!(
            parent.nested.contains(&ComponentId::Soil),
            "{:?}",
            parent.component
        );
        for source in soil.sources {
            assert!(parent.sources.contains(source), "{:?}", parent.component);
        }
    }
    // Pervious subcatchment support runs nested soil admission.
    let p = parameters(0.5, 0.5, SubcatchmentRouting::Direct)?;
    let mut state = SubcatchmentState::initialized(&p)?;
    let memory = state
        .soil
        .as_mut()
        .ok_or("pervious support holds no soil")?;
    memory.fu_ft = -1.0;
    let result = Subcatchment::new(
        crate::runtime::client(),
        vec![SubcatchmentDefinition {
            id: "catchment".into(),
            parameters: p,
        }],
        vec![state],
        0.0,
        SubcatchmentConfig::default(),
    );
    let (variant, reason) =
        built(result, subcatchment_variant)?.ok_or("negative soil memory admitted")?;
    assert_eq!(variant, Variant::Input);
    assert!(reason.contains("require finite state, IMD in"), "{reason}");
    // Subcatchment support runs nested reservoir closure admission.
    let mut ambiguous = geometry(1.0, 1.0);
    ambiguous.impervious_manning = 0.0;
    ambiguous.width_m = 0.0;
    let (variant, reason) = built(
        SubcatchmentParameters::new(ambiguous, None, SubcatchmentRouting::Direct),
        subcatchment_variant,
    )?
    .ok_or("ambiguous subarea closure admitted")?;
    assert_eq!(variant, Variant::Input);
    assert!(
        reason.contains("zero roughness requires positive width and slope"),
        "{reason}"
    );
    // The coordinator reports nested subcatchment and pool rejections as their own variants.
    let pools = Kind::RunoffPools.capabilities();
    for (nested, variant) in [
        (ComponentId::Subcatchment, Variant::Runoff),
        (ComponentId::CoupledLevelPools, Variant::Pools),
    ] {
        assert!(pools.nested.contains(&nested), "{}", nested.id());
        let (setup, mut initial) = exact_case()?;
        match variant {
            Variant::Runoff => initial.subcatchments[0].depths_m[0] = -1.0,
            Variant::Pools => initial.pools[0].volume_m3 = -1.0,
            other => return Err(format!("no nested case for {other:?}").into()),
        }
        let (found, reason) = runoff_pool_outcome(setup, initial, 2.0, &forcing(1, 0.25, 0.0))?
            .ok_or(format!("invalid `{}` state admitted", nested.id()))?;
        assert_eq!(found, variant, "{reason}");
    }
    Ok(())
}
