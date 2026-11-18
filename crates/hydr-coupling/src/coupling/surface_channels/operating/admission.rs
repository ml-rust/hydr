//! Admission of operating children: ownership, common time, and signals.
//!
//! Every rejection names `operation-ownership` first, except a signal outside the surface datum,
//! which names `datum-or-gravity-mismatch`.

use super::interval::interval_operation;
use super::signals::bind_signals;
use super::{
    ACTUATED, IntervalOperation, MANUAL, OperatingBatches, OperatingBinding, OperatingChildren,
    PRESCRIBED, PUMPED, SurfaceChannelOperating, build_children,
};
use crate::coupling::surface_channels::context::Context;
use crate::coupling::surface_channels::structures::SurfaceChannelOperationOwner;
use crate::coupling::surface_channels::{SurfaceChannelDefinition, SurfaceChannelError};
use crate::structures::{OperatingSourceBinding, StructureBatch, StructureElement};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::LinearInterp1dDiagnostics;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Rejects with `operation-ownership`.
fn rejected(detail: String) -> SurfaceChannelError {
    SurfaceChannelError::input(format!("operation-ownership: {detail}"))
}

/// Registry source of `owner`, or `None` for a manual owner.
fn source_of(owner: &SurfaceChannelOperationOwner) -> Option<&'static str> {
    match owner {
        SurfaceChannelOperationOwner::Manual(_) => None,
        SurfaceChannelOperationOwner::Prescribed => Some(PRESCRIBED),
        SurfaceChannelOperationOwner::Actuated => Some(ACTUATED),
        SurfaceChannelOperationOwner::ControlledPump => Some(PUMPED),
    }
}

/// Rejects a structure with a dynamic owner in a coordinator without operating children.
pub(in super::super) fn check_manual(
    definition: &SurfaceChannelDefinition,
) -> Result<(), SurfaceChannelError> {
    match definition
        .structures
        .iter()
        .find(|s| s.structure.operation.is_dynamic())
    {
        Some(structure) => Err(rejected(format!(
            "structure={:?} owner={:?} has no operating child: supply its operating child or give it a manual operation",
            structure.id(),
            structure.structure.operation
        ))),
        None => Ok(()),
    }
}

/// Builds the children `declared` holds on the structure links of `definition`, then checks them
/// against `definition` and its `context` at common time `time_s`.
///
/// # Errors
/// Returns [`SurfaceChannelError::Input`] naming the child for every child rejection, and
/// naming `operation-ownership` for:
/// - a declaration without any child
/// - a structure with zero owners or two: its declared owner has no claiming child, or a child
///   claims a structure of another owner
/// - a child whose initial time differs from `time_s`, a trajectory whose support excludes
///   `time_s`, or a common operating support that excludes `time_s`
/// - a controller identity in both children, a signal naming no controller signal, a signal
///   bound twice, a signal on a node outside the node table, or a controller signal without a
///   binding
///
/// A signal outside the surface datum rejects with `datum-or-gravity-mismatch`.
pub(in super::super) fn admit<R: Runtime<DType = DType>>(
    definition: &SurfaceChannelDefinition,
    context: &Context<R>,
    time_s: f64,
    declared: &SurfaceChannelOperating,
) -> Result<OperatingChildren<R>, SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + LinearInterp1dDiagnostics<R>,
{
    if declared.gates.is_none() && declared.pumps.is_none() && declared.trajectories.is_empty() {
        return Err(rejected(
            "operating declares no child: omit operating when every structure has a manual owner"
                .into(),
        ));
    }
    let expected: Vec<StructureElement> = definition
        .structures
        .iter()
        .map(|s| s.structure.element.clone())
        .collect();
    let OperatingBatches {
        gate,
        pump,
        trajectory,
    } = build_children::<R>(context.client.clone(), expected.clone(), declared)?;
    let mut claims: Vec<(&str, &str)> = Vec::new();
    if let Some(gate) = &gate {
        for actuator in gate.actuators().definitions() {
            claims.push((actuator.structure_id.as_str(), ACTUATED));
        }
    }
    if let Some(pump) = &pump {
        for response in pump.definitions() {
            claims.push((response.structure_id.as_str(), PUMPED));
        }
    }
    if let Some(trajectory) = &trajectory {
        for assignment in trajectory.assignments() {
            claims.push((assignment.structure_id.as_str(), PRESCRIBED));
        }
    }
    let mut claimed: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (id, source) in claims {
        claimed.entry(id).or_default().push(source);
    }
    for structure in &definition.structures {
        let id = structure.id();
        let declared = source_of(&structure.structure.operation);
        let children = claimed.get(id).map_or(&[][..], Vec::as_slice);
        let owned = match declared {
            None => children.is_empty(),
            Some(source) => children == [source],
        };
        if !owned {
            return Err(rejected(format!(
                "structure={id:?} owner={:?} claimed by children={children:?}: give each structure exactly one owner, held by the operating child of its declared owner",
                structure.structure.operation
            )));
        }
    }
    check_time(gate.as_ref(), pump.as_ref(), trajectory.as_ref(), time_s)?;
    let gate_controls = gate
        .as_ref()
        .map_or(&[][..], |g| g.controls().definitions());
    let pump_controls = pump
        .as_ref()
        .map_or(&[][..], |p| p.controls().definitions());
    let (gate_signals, pump_signals) = bind_signals(
        definition,
        context,
        (gate_controls, pump_controls),
        &declared.signals,
    )?;
    let client = context.client.clone();
    let structures = StructureBatch::<R>::new(client, expected)
        .map_err(|error| rejected(format!("operating structures: {error}")))?;
    let manual: Vec<String> = definition
        .structures
        .iter()
        .filter(|s| !s.structure.operation.is_dynamic())
        .map(|s| s.id().to_string())
        .collect();
    let mut sources = Vec::new();
    if !manual.is_empty() {
        sources.push(OperatingSourceBinding::Manual {
            source_id: MANUAL.into(),
            structure_ids: manual,
        });
    }
    if let Some(trajectory) = &trajectory {
        sources.push(OperatingSourceBinding::Prescribed {
            source_id: PRESCRIBED.into(),
            trajectory,
        });
    }
    if let Some(gate) = &gate {
        sources.push(OperatingSourceBinding::Actuated {
            source_id: ACTUATED.into(),
            actuator: gate.actuators(),
        });
    }
    if let Some(pump) = &pump {
        sources.push(OperatingSourceBinding::ControlledPump {
            source_id: PUMPED.into(),
            pump,
        });
    }
    let registry = structures
        .with_operating_sources(sources)
        .map_err(|error| rejected(format!("operating sources: {error}")))?;
    if let Some(support) = registry.support()
        && !(support.start_s <= time_s && time_s < support.end_s)
    {
        return Err(rejected(format!(
            "time_s={time_s} lies outside the common operating support=[{},{}): start every operating child inside its support",
            support.start_s, support.end_s
        )));
    }
    let links = definition
        .structures
        .iter()
        .filter(|s| s.structure.operation.is_dynamic())
        .map(|s| {
            let id = s.id();
            context
                .link_ids
                .iter()
                .position(|link| link == id)
                .map(|link| (id.to_string(), link))
                .ok_or_else(|| {
                    SurfaceChannelError::Numerical(format!(
                        "structure={id:?} lacks its link table position"
                    ))
                })
        })
        .collect::<Result<_, _>>()?;
    let interval: IntervalOperation<R> = interval_operation::<R>;
    Ok(OperatingChildren {
        binding: Arc::new(OperatingBinding {
            registry,
            interval,
            links,
            gate_signals,
            pump_signals,
        }),
        gate,
        pump,
    })
}

/// Rejects a fresh child whose initial time stands off `time_s`.
fn check_time<R: Runtime<DType = DType>>(
    gate: Option<&crate::structures::ControlledGateBatch<R>>,
    pump: Option<&crate::structures::ControlledPumpBatch<R>>,
    trajectory: Option<&crate::structures::GateTrajectoryBatch<R>>,
    time_s: f64,
) -> Result<(), SurfaceChannelError> {
    let mut children: Vec<(&str, f64)> = Vec::new();
    if let Some(gate) = gate {
        children.push(("controlled-gate", gate.time()));
        children.push(("controlled-gate actuators", gate.actuators().time()));
    }
    if let Some(pump) = pump {
        children.push(("controlled-pump", pump.time()));
    }
    for (child, time) in children {
        if time.to_bits() != time_s.to_bits() {
            return Err(rejected(format!(
                "child={child} time={time}, common time_s={time_s}: start the child at the common time"
            )));
        }
    }
    if let Some(trajectory) = trajectory {
        let support = trajectory.support();
        if !(support.start_s <= time_s && time_s < support.end_s) {
            return Err(rejected(format!(
                "child=trajectory support=[{},{}] excludes common time_s={time_s}: start the trajectory at or before the common time",
                support.start_s, support.end_s
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{build, controlled_gate, operating, signal, trajectory};
    use super::super::{SurfaceChannelControlledGates, SurfaceChannelOperating};
    use crate::coupling::surface_channels::pools::SurfaceChannelPools;
    use crate::coupling::surface_channels::pools::scenario::{
        build as manual_build, channel, declaration, link, long_weir, nodes_setup, pond, pool,
        surface,
    };
    use crate::coupling::surface_channels::structures::scenario::{Tanks, TestResult, gate_law};
    use crate::coupling::surface_channels::structures::{
        SurfaceChannelOperationOwner, SurfaceChannelStructure,
    };
    use crate::coupling::surface_channels::{SurfaceChannelDefinition, SurfaceChannelError};
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    const TIMES: [f64; 3] = [1.0, 2.0, 3.0];

    /// An actuated outlet from a pond onto the surface beside a manual spill weir.
    fn structures(owner: SurfaceChannelOperationOwner) -> TestResult<Vec<SurfaceChannelStructure>> {
        let mut outlet = link(
            "outlet",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::GateOpening { opening_m: 0.0 },
            (pool("pond"), 0.0),
            (surface(), 0.0),
        );
        outlet.operation = owner;
        let spill = link(
            "spill",
            long_weir(0.9, 1.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            (pool("pond"), 0.0),
            (surface(), 0.0),
        );
        Ok(vec![outlet, spill])
    }

    fn pools() -> TestResult<SurfaceChannelPools> {
        Ok(declaration(
            vec![pond("pond", 0.0, 4.0, 2.0)?],
            false,
            Vec::new(),
        ))
    }

    fn gate(start: f64) -> SurfaceChannelControlledGates {
        controlled_gate("outlet", [0.4, 0.6], (0.3, 0.1), (start, &TIMES, 4.0))
    }

    /// The input message of a coordinator with `structures` and `operating`.
    fn rejection(
        structures: Vec<SurfaceChannelStructure>,
        operating: SurfaceChannelOperating,
    ) -> TestResult<String> {
        match build(Tanks::at(0.0, 0.2), pools()?, &[2.0], structures, operating) {
            Err(SurfaceChannelError::Input(message)) => Ok(message),
            Err(other) => Err(format!("expected an input rejection, got {other}").into()),
            Ok(_) => Err("the coordinator admitted invalid operating children".into()),
        }
    }

    /// Operating children holding `gate` and the pond signal.
    fn with_gate(gate: SurfaceChannelControlledGates) -> SurfaceChannelOperating {
        SurfaceChannelOperating {
            gates: Some(gate),
            ..operating(vec![signal("outlet", pool("pond"))])
        }
    }

    /// Frozen criteria, fixed before the first run:
    /// - The valid coordinator builds.
    /// - Each `operation-ownership` row of the plan that a declaration can hold rejects, and its
    ///   message names the entry first: a declaration without a child, zero owners, two owners,
    ///   a dynamic owner on another law, an unknown signal node, an unbound or unknown controller
    ///   signal, a signal bound twice, and a child off the common time.
    /// - The coordinator builds every child on its own structure links, fresh, so a child
    ///   context that differs and a child that is not fresh have no representation.
    #[test]
    fn each_operation_ownership_row_rejects() -> TestResult {
        let actuated = || structures(SurfaceChannelOperationOwner::Actuated);
        build(
            Tanks::at(0.0, 0.2),
            pools()?,
            &[2.0],
            actuated()?,
            with_gate(gate(0.0)),
        )?;
        let mut cases: Vec<(&str, String)> = Vec::new();
        cases.push((
            "operating declares no child",
            rejection(actuated()?, operating(Vec::new()))?,
        ));
        let mut wider = actuated()?;
        let mut extra = link(
            "extra",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::GateOpening { opening_m: 0.0 },
            (pool("pond"), 0.0),
            (channel(), 0.0),
        );
        extra.operation = SurfaceChannelOperationOwner::Prescribed;
        wider.push(extra);
        cases.push((
            "owner=Actuated claimed by children=[]",
            rejection(
                wider,
                SurfaceChannelOperating {
                    trajectories: vec![trajectory("extra", &[(0.0, 0.0), (4.0, 0.2)])?],
                    ..operating(Vec::new())
                },
            )?,
        ));
        let manual = structures(SurfaceChannelOperationOwner::Manual(
            StructureOperatingPoint::GateOpening { opening_m: 0.1 },
        ))?;
        cases.push((
            "claimed by children=[\"actuated\"]",
            rejection(manual, with_gate(gate(0.0)))?,
        ));
        let prescribed = structures(SurfaceChannelOperationOwner::Prescribed)?;
        cases.push((
            "children=[\"actuated\", \"prescribed\"]",
            rejection(
                prescribed,
                SurfaceChannelOperating {
                    trajectories: vec![trajectory("outlet", &[(0.0, 0.0), (4.0, 0.2)])?],
                    ..with_gate(gate(0.0))
                },
            )?,
        ));
        let mut pumped = actuated()?;
        pumped[0].operation = SurfaceChannelOperationOwner::ControlledPump;
        let (mut setup, _) = Tanks::at(0.0, 0.2).parts()?;
        setup.operating = Some(operating(Vec::new()));
        let law = match SurfaceChannelDefinition::new(nodes_setup(setup, pools()?, pumped)) {
            Err(SurfaceChannelError::Input(message)) => message,
            other => return Err(format!("a pump owner on a gate rejects: {other:?}").into()),
        };
        cases.push(("does not drive its law", law));
        let unknown = SurfaceChannelOperating {
            signals: vec![signal("outlet", channel())],
            ..with_gate(gate(0.0))
        };
        cases.push((
            "node=\"reach:r:cell:0\" is no coordinator node",
            rejection(actuated()?, unknown)?,
        ));
        let unbound = SurfaceChannelOperating {
            signals: Vec::new(),
            ..with_gate(gate(0.0))
        };
        cases.push(("has no binding", rejection(actuated()?, unbound)?));
        let stranger = SurfaceChannelOperating {
            signals: vec![signal("spill", pool("pond"))],
            ..with_gate(gate(0.0))
        };
        cases.push((
            "names no controller signal",
            rejection(actuated()?, stranger)?,
        ));
        let twice = SurfaceChannelOperating {
            signals: vec![signal("outlet", pool("pond")), signal("outlet", surface())],
            ..with_gate(gate(0.0))
        };
        cases.push(("is bound twice", rejection(actuated()?, twice)?));
        cases.push((
            "time=0.5, common time_s=0",
            rejection(actuated()?, with_gate(gate(0.5)))?,
        ));
        for (reason, message) in &cases {
            assert!(message.starts_with("operation-ownership"), "{message}");
            assert!(message.contains(reason), "{reason}: {message}");
        }
        Ok(())
    }

    /// Frozen criteria, fixed before the first run:
    /// - A dynamic owner without operating children rejects with `operation-ownership` at the
    ///   definition.
    #[test]
    fn a_dynamic_owner_needs_its_operating_child() -> TestResult {
        let error = manual_build(
            Tanks::at(0.0, 0.2),
            pools()?,
            &[2.0],
            structures(SurfaceChannelOperationOwner::Actuated)?,
        )
        .err()
        .ok_or("a dynamic owner without its child rejects")?;
        let SurfaceChannelError::Input(message) = error else {
            return Err(format!("expected an input rejection, got {error}").into());
        };
        assert!(message.starts_with("operation-ownership"), "{message}");
        Ok(())
    }
}
