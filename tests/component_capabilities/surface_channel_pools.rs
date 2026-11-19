//! Surface/channel pool, structure, pool-end, and operating declarations checked against runs
//! and admission.

use crate::surface_channel_support::pools_fixture::{
    Batch, Case, PoolForcing, Result, channel, controlled_gate, controlled_pump, forcing, gate_law,
    inflow_binding, inflow_input, manual, operating, orifice_law, pond, pool, pool_end, pump_law,
    signal, step, structure, surface, trajectory, transfer, weir_law,
};
use crate::surface_channels_fixture::{Outcome, construction_outcome, trial_outcome};
use crate::surface_channels_variants::{Variant, structure_regime};
use hydr::capability::{BoundaryKind, EndpointId, Regime, RegimeScope, SourceKind};
use hydr::channel::ChannelEnd;
use hydr::coupling::{
    SurfaceChannelNode, SurfaceChannelOperating, SurfaceChannelOperationOwner,
    SurfaceChannelStructure, surface_channel_capabilities,
};
use hydr::forcing::WaterAmount;
use hydr::storage::LevelPoolSources;
use hydr::structures::{ControlCommand, DirectionPolicy, StructureOperatingPoint};

/// Requested duration of every one-interval case, in s.
const DT: f64 = 0.1;

/// A full pond at stage 1 m with its structure `id` of `kind` from the pond to `node`.
///
/// The node stands at `stage`. Dynamic kinds take their operating child, held open.
fn kind_case(kind: BoundaryKind, node: SurfaceChannelNode, stage: f64) -> Result<Case> {
    let mut case = match node {
        SurfaceChannelNode::SurfaceCell(_) => Case::tanks(stage, 0.5),
        _ => Case::tanks(0.0, stage),
    };
    case.ponds = vec![(pond("pond", 0.0, 8.0, 2.0)?, 8.0)];
    let schedule = (0.0, &[1.0][..], 10.0);
    let gate = |owner| -> Result<SurfaceChannelStructure> {
        Ok(structure(
            "s",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            owner,
            (pool("pond"), 0.0),
            (node.clone(), 0.0),
        ))
    };
    let pump = |owner| -> Result<SurfaceChannelStructure> {
        Ok(structure(
            "s",
            pump_law(0.1, Some(1.0))?,
            DirectionPolicy::AtoBOnly,
            owner,
            (pool("pond"), 0.0),
            (node.clone(), 0.0),
        ))
    };
    let running = StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 };
    let structure = match kind {
        BoundaryKind::SuppressedRectangularWeir => structure(
            "s",
            weir_law(0.5, 1.0)?,
            DirectionPolicy::Bidirectional,
            manual(StructureOperatingPoint::Passive),
            (pool("pond"), 0.0),
            (node.clone(), 0.0),
        ),
        BoundaryKind::SubmergedOrifice => structure(
            "s",
            orifice_law(0.0, 0.2, true)?,
            DirectionPolicy::Bidirectional,
            manual(StructureOperatingPoint::Passive),
            (pool("pond"), 0.0),
            (node.clone(), 0.0),
        ),
        BoundaryKind::GateFixedOpening => gate(manual(StructureOperatingPoint::GateOpening {
            opening_m: 0.2,
        }))?,
        BoundaryKind::GatePrescribedOpening => {
            case.operating = Some(SurfaceChannelOperating {
                trajectories: vec![trajectory("s", &[(0.0, 0.2), (10.0, 0.2)])?],
                ..operating(Vec::new())
            });
            gate(SurfaceChannelOperationOwner::Prescribed)?
        }
        BoundaryKind::GateActuatedOpening => {
            case.operating = Some(SurfaceChannelOperating {
                gates: Some(controlled_gate(
                    "s",
                    [0.1, 3.0],
                    ([0.2, 0.2], 0.1),
                    schedule,
                )),
                ..operating(vec![signal("s", pool("pond"))])
            });
            gate(SurfaceChannelOperationOwner::Actuated)?
        }
        BoundaryKind::PumpRunning => pump(manual(running))?,
        BoundaryKind::PumpStoppedIsolated => {
            pump(manual(StructureOperatingPoint::PumpStoppedIsolated))?
        }
        BoundaryKind::PumpSampledControl => {
            case.operating = Some(SurfaceChannelOperating {
                pumps: Some(controlled_pump(
                    "s",
                    [0.1, 3.0],
                    [running_command(), running_command()],
                    schedule,
                )),
                ..operating(vec![signal("s", pool("pond"))])
            });
            pump(SurfaceChannelOperationOwner::ControlledPump)?
        }
        other => return Err(format!("no structure case for {other:?}").into()),
    };
    case.structures = vec![structure];
    Ok(case)
}

fn running_command() -> ControlCommand {
    ControlCommand::PumpRunning { speed_ratio: 1.0 }
}

/// One accepted interval of `case` without pool forcing.
fn once(case: &Case) -> Result<Batch> {
    step(&case.build()?, DT, &PoolForcing::default())
}

/// True when `kind` at `endpoint` moves water, or, for a stopped pump, isolates its sides.
fn kind_moves(endpoint: EndpointId, kind: BoundaryKind) -> Result<bool> {
    let node = match endpoint {
        EndpointId::SurfaceCell | EndpointId::PoolConnection => surface(),
        EndpointId::Lateral => channel(),
        other => return Err(format!("no structure endpoint case for {other:?}").into()),
    };
    if matches!(
        kind,
        BoundaryKind::PrescribedInflowRate | BoundaryKind::PrescribedInflowVolume
    ) {
        return inflow_moves(kind);
    }
    // An orifice needs both sides above its crown.
    let stage = if kind == BoundaryKind::SubmergedOrifice {
        0.5
    } else {
        0.0
    };
    let batch = once(&kind_case(kind, node, stage)?)?;
    let moved = transfer(&batch, "s")?;
    Ok(match kind {
        BoundaryKind::PumpStoppedIsolated => {
            moved.signed_volume_m3.to_bits() == 0.0f64.to_bits()
                && structure_regime(moved.regime, moved.law_volume_m3) == Regime::PumpIsolated
        }
        _ => moved.signed_volume_m3 > 0.0,
    })
}

/// True when an external inflow of `kind` raises its pond.
fn inflow_moves(kind: BoundaryKind) -> Result<bool> {
    let mut case = kind_case(BoundaryKind::SuppressedRectangularWeir, surface(), 0.0)?;
    case.ponds[0].1 = 2.0;
    case.inflows = vec![inflow_binding("feed", "pond")];
    let value = match kind {
        BoundaryKind::PrescribedInflowRate => WaterAmount::Rate { rate_m3_per_s: 0.1 },
        _ => WaterAmount::IntervalVolume { volume_m3: 0.01 },
    };
    let pools = PoolForcing {
        sources: None,
        inflows: vec![inflow_input("feed", value)],
    };
    let batch = step(&case.build()?, DT, &pools)?;
    let ledger = batch.pool_inflow_ledgers()?;
    Ok(ledger.first().is_some_and(|l| l.cumulative_volume_m3 > 0.0))
}

/// True when a pool end on `endpoint` exchanges water with its pond.
fn pool_end_moves(endpoint: EndpointId) -> Result<bool> {
    let end = match endpoint {
        EndpointId::ReachLeft => ChannelEnd::Left,
        _ => ChannelEnd::Right,
    };
    let mut case = Case::tanks(0.0, 0.5);
    case.ponds = vec![(pond("pond", 0.0, 8.0, 2.0)?, 0.4 * 8.0)];
    case.pool_ends = vec![pool_end("outfall", end, "pond")];
    let batch = once(&case)?;
    let step = batch.last_step()?.ok_or("an accepted interval")?;
    Ok(step
        .pool_ends
        .first()
        .is_some_and(|e| e.signed_volume_m3 > 0.0))
}

/// The node stage of a case that holds `regime` under `kind`.
fn regime_holds(kind: BoundaryKind, regime: Regime) -> Result<bool> {
    if kind == BoundaryKind::QuiescentReservoir {
        return pool_end_moves(EndpointId::ReachRight);
    }
    let (stage, edit): (f64, fn(&mut Case) -> Result) = match regime {
        Regime::SubmergedOrificeFlow | Regime::SubmergedGateAperture => (0.5, |_| Ok(())),
        Regime::PumpCurve | Regime::PumpIsolated => (0.0, |_| Ok(())),
        // The receiver fills from its bed but stays below a raised sill.
        Regime::FreeGateAperture => (0.0, |c| {
            c.structures[0].element.law = gate_law(0.1)?;
            Ok(())
        }),
        Regime::MixedGateAperture => (0.1, |_| Ok(())),
        Regime::GateClosed => (0.0, |c| {
            c.structures[0].operation =
                manual(StructureOperatingPoint::GateOpening { opening_m: 0.0 });
            Ok(())
        }),
        Regime::PumpShutoff => (0.0, |c| {
            c.structures[0].element.law = pump_law(0.1, Some(4.0))?;
            Ok(())
        }),
        other => return Err(format!("no regime case for {other:?}").into()),
    };
    let mut case = kind_case(kind, surface(), stage)?;
    edit(&mut case)?;
    let moved = transfer(&once(&case)?, "s")?;
    let held = structure_regime(moved.regime, moved.law_volume_m3) == regime;
    assert!(held, "{regime:?}: {moved:?}");
    Ok(held)
}

/// Scope kinds the pool, structure, and pool-end cases cover.
fn covered(scope: RegimeScope) -> Option<BoundaryKind> {
    match scope {
        RegimeScope::Boundary(
            kind @ (BoundaryKind::SubmergedOrifice
            | BoundaryKind::GateFixedOpening
            | BoundaryKind::PumpRunning
            | BoundaryKind::PumpStoppedIsolated
            | BoundaryKind::QuiescentReservoir),
        ) => Some(kind),
        _ => None,
    }
}

#[test]
fn surface_channel_pools_admit_declared_endpoints_and_regimes() -> Result {
    let declared = surface_channel_capabilities();
    let mut cases = 0;
    for endpoint in declared.endpoints {
        for &kind in endpoint.admitted {
            let moved = match (endpoint.id, kind) {
                (_, BoundaryKind::FluxPort) => continue,
                (
                    EndpointId::ReachLeft | EndpointId::ReachRight,
                    BoundaryKind::QuiescentReservoir,
                ) => pool_end_moves(endpoint.id)?,
                (id, kind) => kind_moves(id, kind)?,
            };
            assert!(moved, "{:?} {kind:?} moved no water", endpoint.id);
            cases += 1;
        }
    }
    assert!(cases > 0);
    for claim in declared.regimes {
        let Some(kind) = covered(claim.scope) else {
            continue;
        };
        assert!(
            regime_holds(kind, claim.regime)?,
            "{:?} at {kind:?} does not hold",
            claim.regime
        );
    }
    Ok(())
}

/// True when local pool `source` moves water.
pub fn pool_source_moves_water(source: SourceKind) -> Result<bool> {
    let mut case = kind_case(BoundaryKind::SuppressedRectangularWeir, surface(), 0.0)?;
    case.ponds[0].1 = 2.0;
    case.sources = true;
    let pools = PoolForcing {
        sources: Some(vec![LevelPoolSources {
            rain_rate_m_per_s: 1e-3,
            potential_evaporation_rate_m_per_s: 0.0,
            bottom_loss_rate_m3_per_s: 1e-3,
        }]),
        inflows: Vec::new(),
    };
    let batch = step(&case.build()?, DT, &pools)?;
    let balance = batch.ledger()?;
    Ok(match source {
        SourceKind::LocalPoolRain => balance.external_in_m3 > 0.0,
        SourceKind::LocalPoolLoss => balance.external_out_m3 > 0.0,
        other => return Err(format!("no pool source case for `{}`", other.id()).into()),
    })
}

/// One rejection case: outcome, expected variant and message fragment.
type Rejection = (Outcome, Variant, &'static str);

/// The construction outcome of `case` after `edit`.
fn edited(case: Result<Case>, edit: impl FnOnce(&mut Case)) -> Outcome {
    let mut case = case?;
    edit(&mut case);
    let (setup, initial) = case.parts()?;
    construction_outcome(setup, initial)
}

/// The first-trial outcome of `case`.
fn first_trial(case: Result<Case>) -> Outcome {
    let batch = case?.build()?;
    let end = batch.time() + DT;
    trial_outcome(&batch, end, &forcing(&batch, end, &PoolForcing::default()))
}

/// A pond above the dry surface cell through a free weir.
fn weir_case() -> Result<Case> {
    kind_case(BoundaryKind::SuppressedRectangularWeir, surface(), 0.0)
}

/// Cases for each pool and structure `RejectedByAdmission` entry, or `None` for an unknown id.
pub fn rejection_cases(id: &str) -> Option<Vec<Rejection>> {
    use Variant::Input;
    Some(match id {
        "structure-endpoint-identity" => vec![(
            edited(weir_case(), |c| {
                c.structures[0].element.endpoint_b = "surface-cell:1".into();
            }),
            Input,
            "structure-endpoint-identity",
        )],
        "unobstructed-pool-connection" => vec![(
            edited(weir_case(), |c| {
                c.structures[0].endpoint_a.bed_elevation_m = -0.1
            }),
            Input,
            "unobstructed-pool-connection",
        )],
        "non-symmetric-structure" => vec![(
            edited(kind_case(BoundaryKind::PumpRunning, surface(), 0.0), |c| {
                c.structures[0].element.direction = DirectionPolicy::Bidirectional;
            }),
            Input,
            "non-symmetric-structure",
        )],
        "reach-end-ownership" => vec![(
            edited(weir_case(), |c| {
                c.pool_ends = vec![
                    pool_end("one", ChannelEnd::Right, "pond"),
                    pool_end("two", ChannelEnd::Right, "pond"),
                ];
            }),
            Input,
            "reach-end-ownership",
        )],
        "pool-support" => vec![(
            edited(weir_case(), |c| {
                c.structures[0].endpoint_a.node = channel();
                c.structures[0].element.endpoint_a = channel().identity();
            }),
            Input,
            "pool-support",
        )],
        "operation-ownership" => vec![(
            edited(
                kind_case(BoundaryKind::GateActuatedOpening, surface(), 0.0),
                |c| {
                    c.operating = None;
                },
            ),
            Input,
            "operation-ownership",
        )],
        "pool-capacity-exceeded" => vec![(capacity(), Input, "pool-capacity-exceeded")],
        "pump-without-suction" => vec![(
            first_trial(
                kind_case(BoundaryKind::PumpRunning, surface(), 0.0).and_then(|mut c| {
                    c.structures[0].element.law = pump_law(1.0, Some(1.0))?;
                    Ok(c)
                }),
            ),
            Input,
            "pump-without-suction",
        )],
        "pump-dry-delivery" => vec![(
            first_trial(
                kind_case(BoundaryKind::PumpRunning, surface(), 0.0).and_then(|mut c| {
                    c.structures[0].element.law = pump_law(0.1, None)?;
                    Ok(c)
                }),
            ),
            Input,
            "pump-dry-delivery",
        )],
        "pool-end-regime" => vec![(
            first_trial(Case::tanks(0.0, 0.5).pipe(|c| {
                c.ponds = vec![(pond("pond", 0.0, 8.0, 2.0)?, 0.0)];
                c.pool_ends = vec![pool_end("outfall", ChannelEnd::Right, "pond")];
                Ok(())
            })),
            Input,
            "pool-end-regime",
        )],
        "orifice-partial-submergence" => vec![(
            first_trial(kind_case(BoundaryKind::SubmergedOrifice, surface(), 0.0)),
            Input,
            "orifice",
        )],
        _ => return None,
    })
}

/// A full pond with a pipe that overfills it within the first interval.
fn capacity() -> Outcome {
    let mut case = weir_case()?;
    case.structures[0].element.law = weir_law(1.99, 1.0)?;
    case.ponds[0].1 = 15.99;
    case.inflows = vec![inflow_binding("feed", "pond")];
    let batch = case.build()?;
    let end = batch.time() + DT;
    let pools = PoolForcing {
        sources: None,
        inflows: vec![inflow_input(
            "feed",
            WaterAmount::Rate { rate_m3_per_s: 1.0 },
        )],
    };
    trial_outcome(&batch, end, &forcing(&batch, end, &pools))
}

/// `self` after a fallible edit.
trait Pipe: Sized {
    fn pipe(self, edit: impl FnOnce(&mut Self) -> Result) -> Result<Self>;
}

impl Pipe for Case {
    fn pipe(mut self, edit: impl FnOnce(&mut Self) -> Result) -> Result<Self> {
        edit(&mut self)?;
        Ok(self)
    }
}
