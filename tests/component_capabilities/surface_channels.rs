//! Surface/channel declaration checked against its admission code.

use crate::surface_channel_support::fixture::{
    INACTIVE_CELL, Result, SURFACE_BED, link, reach_definition,
};
use crate::surface_channel_support::scenario::{Batch, ReachSpec, Spec, forcing, weir};
use crate::surface_channels_fixture::{
    CREST, Outcome, REACH, STEP, accept, construction_outcome, edited, green_ampt, pair,
    trial_outcome, triangular, with_rates,
};
use crate::surface_channels_variants::{
    Variant, WEIR_REGIMES, geometry_component, lateral_kind, weir_regime,
};
use hydr::capability::{
    BoundaryKind, ComponentId, EndpointId, Regime, RegimeScope, SourceKind, UnsupportedReason,
};
use hydr::channel::{ChannelBoundary, ChannelLateralDirection, ChannelLateralInput};
use hydr::coupling::surface_channel_capabilities;
use hydr::forcing::WaterAmount;

/// Paired surface and channel stages that hold each declared regime at the weir.
fn regime_stages(regime: Regime) -> Option<[f64; 2]> {
    Some(match regime {
        Regime::FreeWeirFlow => [1.5, 0.5],
        Regime::SubmergedWeirFlow => [1.5, 1.3],
        Regime::BelowCrest => [1.1, 0.5],
        _ => return None,
    })
}

/// Accepts one interval of `spec` with walls and zero caller rates.
fn one_interval(spec: &Spec) -> Result<(Batch, hydr::coupling::SurfaceChannelStepDiagnostics)> {
    let batch = spec.build()?;
    let held = forcing(&batch, STEP, &[]);
    accept(&batch, STEP, &held)
}

#[test]
fn surface_channels_admit_declared_endpoints_regimes_and_sources() -> Result {
    let declared = surface_channel_capabilities();
    // A spill and its reversal move water through both sides of one bank weir.
    let (_, spill) = one_interval(&pair(1.5, 0.5))?;
    let (_, reverse) = one_interval(&pair(1.1, 1.5))?;
    assert!(spill.links[0].signed_volume_m3 > 0.0, "{spill:?}");
    assert!(reverse.links[0].signed_volume_m3 < 0.0, "{reverse:?}");
    // Port endpoints and regimes carry their own evidence in `surface_channel_ports`, pool,
    // structure, and pool-end endpoints in `surface_channel_pools`, and the runoff outlet in
    // `surface_channel_runoff`.
    for endpoint in declared.endpoints {
        for &kind in endpoint.admitted {
            if kind != BoundaryKind::SuppressedRectangularWeir
                || !matches!(endpoint.id, EndpointId::SurfaceCell | EndpointId::Lateral)
            {
                continue;
            }
            let moved = match endpoint.id {
                EndpointId::SurfaceCell => [
                    spill.surface_exchange_out_m3,
                    reverse.surface_exchange_in_m3,
                ],
                EndpointId::Lateral => [
                    spill.channel_exchange_in_m3,
                    reverse.channel_exchange_out_m3,
                ],
                other => return Err(format!("surface/channel: no case for {other:?}").into()),
            };
            assert!(
                moved.iter().all(|v| *v > 0.0),
                "{:?}: {moved:?}",
                endpoint.id
            );
        }
    }
    for claim in declared.regimes {
        if claim.scope != RegimeScope::Boundary(BoundaryKind::SuppressedRectangularWeir) {
            continue;
        }
        let [surface, channel] =
            regime_stages(claim.regime).ok_or(format!("no case for {claim:?}"))?;
        let (_, step) = one_interval(&pair(surface, channel))?;
        assert_eq!(weir_regime(step.links[0].regime), claim.regime, "{step:?}");
    }
    // Equal heads above the crest carry no water inside the submerged regime.
    let (_, equal) = one_interval(&pair(1.5, 1.5))?;
    let link = &equal.links[0];
    assert_eq!(weir_regime(link.regime), Regime::SubmergedWeirFlow);
    assert_eq!(link.signed_volume_m3, 0.0, "{link:?}");
    for source in declared.sources {
        assert!(
            source_moves_water(*source)?,
            "`{}` moved no water",
            source.id()
        );
    }
    // Reach ends keep the nested reach's endpoint policies.
    let batch = pair(1.1, 0.5).build()?;
    let mut held = forcing(&batch, STEP, &[]);
    held.reaches[0].right = Some(ChannelBoundary::Stage { elevation_m: 0.5 });
    accept(&batch, STEP, &held)?;
    Ok(())
}

/// Runs one interval with `source` active and reports whether it moved water.
fn source_moves_water(source: SourceKind) -> Result<bool> {
    let below = pair(1.1, 0.5);
    let surface = |rain, evaporation, withdrawal| -> Result<hydr::solver::WaterLedger> {
        let mut spec = below.clone();
        spec.withdrawal = vec![withdrawal];
        let (setup, initial) = with_rates(&spec, rain, evaporation)?;
        let batch = Batch::new(crate::runtime::client(), setup, initial)?;
        let held = forcing(&batch, STEP, &[]);
        Ok(accept(&batch, STEP, &held)?.0.surface().ledger().clone())
    };
    Ok(match source {
        SourceKind::Rain => surface(1e-3, 0.0, 0.0)?.rain > 0.0,
        // Impervious land keeps all rain on the surface.
        SourceKind::ImperviousLand => {
            let ledger = surface(1e-3, 0.0, 0.0)?;
            ledger.rain > 0.0 && ledger.infiltration == 0.0
        }
        SourceKind::ModifiedGreenAmpt => {
            let (batch, _) = green_ampt_overflow()?;
            batch.surface().ledger().infiltration > 0.0
        }
        SourceKind::Evaporation => surface(0.0, 1e-3, 0.0)?.evaporation > 0.0,
        SourceKind::Withdrawal => surface(0.0, 0.0, 1e-3)?.withdrawal > 0.0,
        SourceKind::LateralInflow | SourceKind::LateralWithdrawal => lateral_moves_water(source)?,
        SourceKind::LocalPoolRain | SourceKind::LocalPoolLoss => {
            crate::surface_channel_pools::pool_source_moves_water(source)?
        }
        other => return Err(format!("surface/channel: no case for `{}`", other.id()).into()),
    })
}

/// Green-Ampt soil under a spilling surface cell, advanced through `GREEN_AMPT_END`.
///
/// Returns the final batch and whether every interval spilled into the reach.
fn green_ampt_overflow() -> Result<(Batch, bool)> {
    let (setup, initial) = green_ampt(&pair(1.5, 0.5))?;
    let mut batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let mut spilled = true;
    while batch.time() < GREEN_AMPT_END {
        let held = forcing(&batch, GREEN_AMPT_END, &[]);
        let (next, step) = accept(&batch, GREEN_AMPT_END, &held)?;
        spilled &= step.links[0].signed_volume_m3 > 0.0;
        batch = next;
    }
    Ok((batch, spilled))
}

/// End of the Green-Ampt overflow case, in seconds.
const GREEN_AMPT_END: f64 = 0.5;

#[test]
fn surface_channels_close_balance_with_green_ampt_infiltration() -> Result {
    let (batch, spilled) = green_ampt_overflow()?;
    assert!(spilled, "every interval must spill into the reach");
    let surface = batch.surface().ledger();
    assert!(surface.infiltration > 0.0, "{surface:?}");
    let balance = batch.ledger()?;
    // Infiltration is the only water leaving the composite system.
    assert!(
        (balance.external_out_m3 - surface.infiltration).abs() <= balance.bound_m3,
        "{balance:?}"
    );
    let change = balance.storage_m3 - balance.initial_storage_m3;
    let external = balance.external_in_m3 - balance.external_out_m3 + balance.water_adjustment_m3;
    assert_eq!(change - external, balance.residual_m3, "{balance:?}");
    assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
    assert!(balance.channel_exchange_in_m3 > 0.0, "{balance:?}");
    Ok(())
}

/// Runs one interval with a caller lateral input of the kind `source` names.
fn lateral_moves_water(source: SourceKind) -> Result<bool> {
    let amount = WaterAmount::Rate {
        rate_m3_per_s: 1e-2,
    };
    let direction = [
        ChannelLateralDirection::Inflow,
        ChannelLateralDirection::Withdrawal,
    ]
    .into_iter()
    .find(|d| lateral_kind(*d) == source)
    .ok_or("no lateral direction")?;
    let input = match direction {
        ChannelLateralDirection::Inflow => ChannelLateralInput::Inflow {
            boundary_id: "caller".into(),
            amount,
            longitudinal_velocity_m_per_s: 0.0,
        },
        ChannelLateralDirection::Withdrawal => ChannelLateralInput::Withdrawal {
            boundary_id: "caller".into(),
            amount,
        },
    };
    let mut spec = pair(1.1, 0.5);
    spec.reaches[0] = ReachSpec::new(REACH, 0.0, &[0.5; 2]).binding("caller", 1, direction);
    let batch = spec.build()?;
    let held = forcing(&batch, STEP, &[(REACH, input)]);
    let (next, _) = accept(&batch, STEP, &held)?;
    let ledger = next.reach(REACH).ok_or("missing reach")?.ledger();
    Ok(match direction {
        ChannelLateralDirection::Inflow => ledger.lateral_in_m3 > 0.0,
        ChannelLateralDirection::Withdrawal => ledger.lateral_out_m3 > 0.0,
    })
}

#[test]
fn surface_channels_nest_and_attach_declared_components() -> Result {
    let declared = surface_channel_capabilities();
    let rectangular = pair(1.5, 0.5);
    let (setup, initial) = triangular(&rectangular)?;
    let triangle = setup.reaches[0].definition.geometry.clone();
    if let Some((variant, reason)) = construction_outcome(setup, initial)? {
        return Err(format!("triangular link: {variant:?} {reason}").into());
    }
    // A surveyed reach with Wall endpoints spills through the same bank weir.
    let surveyed = Spec::new(1, 1, SURFACE_BED, 0.5)
        .reach(ReachSpec::new(REACH, 0.0, &[0.5; 2]).surveyed())
        .link(weir("bank", 0, REACH, 0, CREST, 2.0));
    let (surveyed_setup, _) = surveyed.parts()?;
    let (_, spill) = one_interval(&surveyed)?;
    assert!(spill.links[0].signed_volume_m3 > 0.0, "{spill:?}");
    let (setup, _) = rectangular.parts()?;
    for geometry in [
        setup.reaches[0].definition.geometry.clone(),
        triangle,
        surveyed_setup.reaches[0].definition.geometry.clone(),
    ] {
        let component = geometry_component(&geometry);
        assert!(declared.nested.contains(&component), "{component:?}");
    }
    assert!(declared.nested.contains(&ComponentId::SurfaceSolver));
    assert!(declared.attachable_to.is_empty());
    for nested in declared.nested {
        let child = match nested {
            ComponentId::SurfaceSolver => hydr::solver::surface_solver_capabilities(),
            ComponentId::Soil => hydr::soil::soil_capabilities(),
            ComponentId::ChannelReachRectangular => hydr::channel::rectangular_reach_capabilities(),
            ComponentId::ChannelReachTriangular => hydr::channel::triangular_reach_capabilities(),
            ComponentId::ChannelReachSurveyed => hydr::channel::surveyed_reach_capabilities(),
            ComponentId::LevelPool => hydr::storage::level_pool_capabilities(),
            ComponentId::OperatingControl => hydr::structures::operating_control_capabilities(),
            ComponentId::GateActuator => hydr::structures::gate_actuator_capabilities(),
            ComponentId::ControlledGate => hydr::structures::controlled_gate_capabilities(),
            ComponentId::ControlledPump => hydr::structures::controlled_pump_capabilities(),
            ComponentId::Culvert => hydr::structures::culvert_capabilities(),
            ComponentId::Subcatchment => hydr::runoff::subcatchment_capabilities(),
            other => return Err(format!("surface/channel: no case for {other:?}").into()),
        };
        assert!(child.attachable_to.contains(&ComponentId::SurfaceChannels));
        assert!(child.unsupported_by_id("surface-solver").is_none());
        assert!(child.unsupported_by_id("channel-reaches").is_none());
    }
    assert_eq!(WEIR_REGIMES.len(), 4);
    Ok(())
}

/// One rejection case: outcome, expected variant and message fragment.
type Rejection = (Outcome, Variant, &'static str);

/// Integrated caller inflow beside one bank link, with the trial span and forcing end.
fn integrated(requested: f64, forcing_end: f64) -> Outcome {
    let spec = Spec::new(2, 1, 1.0, 0.0)
        .reach(ReachSpec::new("river", 0.0, &[0.5, 0.5]).binding(
            "tributary",
            1,
            ChannelLateralDirection::Inflow,
        ))
        .link(weir("bank", 0, "river", 0, CREST, 2.0));
    let batch = spec.build()?;
    let inflow = ChannelLateralInput::Inflow {
        boundary_id: "tributary".into(),
        amount: WaterAmount::IntervalVolume { volume_m3: 0.5 },
        longitudinal_velocity_m_per_s: 0.0,
    };
    let held = forcing(&batch, forcing_end, &[("river", inflow)]);
    trial_outcome(&batch, requested, &held)
}

/// A caller lateral input under the reserved prefix in an otherwise valid interval.
fn reserved_input() -> Outcome {
    let batch = pair(1.5, 0.5).build()?;
    let mut held = forcing(&batch, STEP, &[]);
    held.reaches[0]
        .lateral_inputs
        .push(ChannelLateralInput::Withdrawal {
            boundary_id: "surface-exchange:bank:withdrawal".into(),
            amount: WaterAmount::Rate { rate_m3_per_s: 0.0 },
        });
    trial_outcome(&batch, STEP, &held)
}

/// A common interval shorter than `minimum_step_s`.
fn below_minimum_step() -> Outcome {
    let batch = pair(1.5, 0.5).build()?;
    let end = 0.5 * batch.setup().config.minimum_step_s;
    let held = forcing(&batch, STEP, &[]);
    trial_outcome(&batch, end, &held)
}

/// Cases for each `RejectedByAdmission` entry, or `None` for an undeclared id.
fn rejection_cases(id: &str) -> Option<Vec<Rejection>> {
    use Variant::{Input, Recompute, Unresolved};
    Some(match id {
        "unobstructed-connection" => vec![
            (
                edited(|s| s.links[0].crest_elevation_m = SURFACE_BED - 0.1),
                Input,
                "an unobstructed connection is unsupported",
            ),
            (
                edited(|s| s.reaches[0].definition = reach_definition(4, 1.5)),
                Input,
                "an unobstructed connection is unsupported",
            ),
        ],
        "duplicate-links" => vec![
            (
                edited(|s| s.links[1].id = "link-b".into()),
                Input,
                "give each link a unique identity",
            ),
            (
                edited(|s| s.links.push(link("link-d", 1, 1))),
                Input,
                "merge the duplicate links into one",
            ),
        ],
        "inactive-surface-cell" => [INACTIVE_CELL, 8]
            .map(|cell| {
                (
                    edited(|s| s.links[0].surface_cell = cell),
                    Input,
                    "link an active cell inside the grid",
                )
            })
            .into(),
        "non-unit-tangent" => [[1.0, 1.0], [f64::NAN, 1.0]]
            .map(|tangent| {
                (
                    edited(|s| s.links[0].reach_tangent = tangent),
                    Input,
                    "supply a finite unit tangent",
                )
            })
            .into(),
        "datum-or-gravity-mismatch" => vec![
            (
                edited(|s| s.reaches[0].definition.config.gravity = 9.8),
                Input,
                "supply bit-identical gravity",
            ),
            (
                edited(|s| s.reaches[1].definition.vertical_datum = "Other datum".into()),
                Input,
                "express both in one vertical datum",
            ),
        ],
        "step-and-retry-limits" => vec![
            (
                edited(|s| s.config.retry_limit = 65),
                Input,
                "lower retry_limit to at most the surface max_retries",
            ),
            (
                edited(|s| s.reaches[1].definition.config.max_step = 0.5),
                Input,
                "lower maximum_step_s to at most every reach max_step",
            ),
            (
                edited(|s| s.surface.config.min_step = 1e-6),
                Input,
                "raise minimum_step_s to at least the surface min_step",
            ),
            (
                below_minimum_step(),
                Unresolved,
                "common duration falls below minimum_step_s",
            ),
        ],
        "reserved-binding-prefix" => vec![
            (
                edited(|s| {
                    s.reaches[0].definition.lateral_bindings[0].boundary_id =
                        "surface-exchange:x".into();
                }),
                Input,
                "uses the reserved prefix",
            ),
            (reserved_input(), Input, "omit surface-exchange inputs"),
        ],
        "non-symmetric-weir" => vec![(
            edited(|s| s.links[0].applicability.symmetric_reversal = false),
            Input,
            "supply a coefficient that holds under reversed flow",
        )],
        "missing-link-evidence" => vec![
            (
                edited(|s| s.storage_exclusion_evidence = String::new()),
                Input,
                "supply nonempty identity and evidence",
            ),
            (
                edited(|s| s.links[2].mapping_evidence = "\n".into()),
                Input,
                "supply nonempty identity and evidence",
            ),
        ],
        "integrated-lateral-span" => vec![
            (
                integrated(0.5, 3.0),
                Input,
                "integrated lateral volumes require the exact requested span",
            ),
            (
                integrated(2.0, 2.0),
                Recompute,
                "maximum_step_s or the surface forcing interval ends before the integrated span",
            ),
        ],
        id => {
            return crate::surface_channel_ports::rejection_cases(id)
                .or_else(|| crate::surface_channel_pools::rejection_cases(id))
                .or_else(|| crate::surface_channel_runoff::rejection_cases(id));
        }
    })
}

#[test]
fn surface_channels_reject_declared_rejections() -> Result {
    let mut count = 0;
    for entry in surface_channel_capabilities()
        .unsupported
        .iter()
        .filter(|u| u.reason == UnsupportedReason::RejectedByAdmission)
    {
        let cases = rejection_cases(entry.id).ok_or(format!("no case for `{}`", entry.id))?;
        assert!(!cases.is_empty(), "`{}` has no case", entry.id);
        for (outcome, variant, fragment) in cases {
            let (found, reason) = outcome?.ok_or(format!("`{}` admitted", entry.id))?;
            assert_eq!(found, variant, "`{}`: {reason}", entry.id);
            assert!(
                reason.contains(fragment),
                "`{}`: `{reason}` lacks `{fragment}`",
                entry.id
            );
            count += 1;
        }
    }
    assert!(count > 0, "surface/channel declares no admission rejection");
    // Control: the unedited fixture admits.
    if let Some((variant, reason)) = edited(|_| {})? {
        return Err(format!("fixture: {variant:?} {reason}").into());
    }
    Ok(())
}
