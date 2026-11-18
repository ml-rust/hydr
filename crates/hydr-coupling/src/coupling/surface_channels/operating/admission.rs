//! Admission of operating children: structure context, ownership, freshness, time, and signals.
//!
//! Every rejection names `operation-ownership` first, except a signal outside the surface datum,
//! which names `datum-or-gravity-mismatch`.

use super::interval::interval_operation;
use super::signals::bind_signals;
use super::{
    ACTUATED, IntervalOperation, MANUAL, OperatingBinding, OperatingChildren, PRESCRIBED, PUMPED,
    SurfaceChannelOperating,
};
use crate::coupling::surface_channels::context::Context;
use crate::coupling::surface_channels::structures::SurfaceChannelOperationOwner;
use crate::coupling::surface_channels::{SurfaceChannelDefinition, SurfaceChannelError};
use crate::structures::{OperatingSourceBinding, StructureBatch, StructureElement};
use hydr_structures::internal::same_elements;
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

/// Checks `operating` against `definition` and its `context` at common time `time_s`.
///
/// # Errors
/// Returns [`SurfaceChannelError::Input`] naming `operation-ownership` for:
/// - a child whose structure context differs from the setup structures in ascending identity
/// - a structure with zero owners or two: its declared owner has no claiming child, or a child
///   claims a structure of another owner
/// - a child that has accepted an interval, or whose time differs from `time_s`, or a trajectory
///   whose support excludes `time_s`
/// - a controller identity in both children, a signal naming no controller signal, a signal
///   bound twice, a signal on a node outside the node table, or a controller signal without a
///   binding
///
/// A signal outside the surface datum rejects with `datum-or-gravity-mismatch`.
pub(in super::super) fn admit<R: Runtime<DType = DType>>(
    definition: &SurfaceChannelDefinition,
    context: &Context<R>,
    time_s: f64,
    operating: SurfaceChannelOperating<R>,
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
    let SurfaceChannelOperating {
        gate,
        pump,
        trajectory,
        signals,
    } = operating;
    let expected: Vec<StructureElement> = definition
        .structures
        .iter()
        .map(|s| s.structure.element.clone())
        .collect();
    let mut contexts: Vec<(&str, &[StructureElement])> = Vec::new();
    let mut claims: Vec<(&str, &str)> = Vec::new();
    if let Some(gate) = &gate {
        contexts.push(("controlled-gate controls", gate.controls().elements()));
        contexts.push(("controlled-gate actuators", gate.actuators().elements()));
        for actuator in gate.actuators().definitions() {
            claims.push((actuator.structure_id.as_str(), ACTUATED));
        }
    }
    if let Some(pump) = &pump {
        contexts.push(("controlled-pump", pump.controls().elements()));
        for response in pump.definitions() {
            claims.push((response.structure_id.as_str(), PUMPED));
        }
    }
    if let Some(trajectory) = &trajectory {
        contexts.push(("trajectory", trajectory.elements()));
        for assignment in trajectory.assignments() {
            claims.push((assignment.structure_id.as_str(), PRESCRIBED));
        }
    }
    for (child, elements) in &contexts {
        if !same_elements(elements, &expected) {
            return Err(rejected(format!(
                "child={child} structure context differs from the setup structure links: build every child on the structure links in ascending identity"
            )));
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
    check_fresh(gate.as_ref(), pump.as_ref(), trajectory.as_ref(), time_s)?;
    let gate_controls = gate
        .as_ref()
        .map_or(&[][..], |g| g.controls().definitions());
    let pump_controls = pump
        .as_ref()
        .map_or(&[][..], |p| p.controls().definitions());
    let (gate_signals, pump_signals) =
        bind_signals(definition, context, (gate_controls, pump_controls), signals)?;
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

/// Rejects a child that has accepted an interval or stands off `time_s`.
fn check_fresh<R: Runtime<DType = DType>>(
    gate: Option<&crate::structures::ControlledGateBatch<R>>,
    pump: Option<&crate::structures::ControlledPumpBatch<R>>,
    trajectory: Option<&crate::structures::GateTrajectoryBatch<R>>,
    time_s: f64,
) -> Result<(), SurfaceChannelError> {
    let mut children: Vec<(&str, usize, f64)> = Vec::new();
    if let Some(gate) = gate {
        let steps = gate.accepted_steps().max(gate.actuators().accepted_steps());
        children.push(("controlled-gate", steps, gate.time()));
        children.push(("controlled-gate actuators", steps, gate.actuators().time()));
    }
    if let Some(pump) = pump {
        children.push(("controlled-pump", pump.accepted_steps(), pump.time()));
    }
    for (child, steps, time) in children {
        if steps != 0 {
            return Err(rejected(format!(
                "child={child} accepted_steps={steps} is not fresh: supply the child before its first interval"
            )));
        }
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
    use super::super::SurfaceChannelOperating;
    use super::super::scenario::{
        Cpu, build, context, controlled_gate, operating, signal, trajectory,
    };
    use crate::coupling::surface_channels::pools::SurfaceChannelPools;
    use crate::coupling::surface_channels::pools::scenario::{
        build as manual_build, channel, declaration, link, long_weir, pond, pool, surface,
    };
    use crate::coupling::surface_channels::structures::scenario::{Tanks, TestResult, gate_law};
    use crate::coupling::surface_channels::structures::{
        SurfaceChannelOperationOwner, SurfaceChannelStructure,
    };
    use crate::coupling::surface_channels::{SurfaceChannelDefinition, SurfaceChannelError};
    use crate::structures::{
        ControlledGateBatch, DirectionPolicy, StructureBatch, StructureOperatingPoint,
    };

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

    fn gate(context: &StructureBatch<Cpu>, start: f64) -> TestResult<ControlledGateBatch<Cpu>> {
        controlled_gate(
            context,
            "outlet",
            [0.4, 0.6],
            (0.3, 0.1),
            (start, &TIMES, 4.0),
        )
    }

    /// The input message of a coordinator with `structures` and `operating`.
    fn rejection(
        structures: Vec<SurfaceChannelStructure>,
        operating: SurfaceChannelOperating<Cpu>,
    ) -> TestResult<String> {
        match build(Tanks::at(0.0, 0.2), pools()?, &[2.0], structures, operating) {
            Err(SurfaceChannelError::Input(message)) => Ok(message),
            Err(other) => Err(format!("expected an input rejection, got {other}").into()),
            Ok(_) => Err("the coordinator admitted invalid operating children".into()),
        }
    }

    /// Operating children holding `gate` and the pond signal.
    fn with_gate(gate: ControlledGateBatch<Cpu>) -> SurfaceChannelOperating<Cpu> {
        SurfaceChannelOperating {
            gate: Some(gate),
            ..operating(vec![signal("outlet", pool("pond"))])
        }
    }

    /// Frozen criteria, fixed before the first run:
    /// - The valid coordinator builds.
    /// - Each `operation-ownership` row of the plan rejects, and its message names the entry
    ///   first: child context, zero owners, two owners, a dynamic owner on another law, an
    ///   unknown signal node, an unbound or unknown controller signal, a signal bound twice, a
    ///   child that is not fresh, and a child off the common time.
    #[test]
    fn each_operation_ownership_row_rejects() -> TestResult {
        let actuated = || structures(SurfaceChannelOperationOwner::Actuated);
        let shared = context(&actuated()?)?;
        build(
            Tanks::at(0.0, 0.2),
            pools()?,
            &[2.0],
            actuated()?,
            with_gate(gate(&shared, 0.0)?),
        )?;
        let mut cases: Vec<(&str, String)> = Vec::new();
        let mut wider = actuated()?;
        wider.push(link(
            "extra",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::GateOpening { opening_m: 0.0 },
            (pool("pond"), 0.0),
            (channel(), 0.0),
        ));
        cases.push((
            "structure context differs",
            rejection(actuated()?, with_gate(gate(&context(&wider)?, 0.0)?))?,
        ));
        cases.push((
            "owner=Actuated claimed by children=[]",
            rejection(actuated()?, operating(Vec::new()))?,
        ));
        let manual = structures(SurfaceChannelOperationOwner::Manual(
            StructureOperatingPoint::GateOpening { opening_m: 0.1 },
        ))?;
        cases.push((
            "claimed by children=[\"actuated\"]",
            rejection(manual, with_gate(gate(&shared, 0.0)?))?,
        ));
        let prescribed = structures(SurfaceChannelOperationOwner::Prescribed)?;
        cases.push((
            "children=[\"actuated\", \"prescribed\"]",
            rejection(
                prescribed,
                SurfaceChannelOperating {
                    trajectory: Some(trajectory(&shared, "outlet", &[(0.0, 0.0), (4.0, 0.2)])?),
                    ..with_gate(gate(&shared, 0.0)?)
                },
            )?,
        ));
        let mut pumped = actuated()?;
        pumped[0].operation = SurfaceChannelOperationOwner::ControlledPump;
        let (setup, _) = Tanks::at(0.0, 0.2).parts()?;
        let law = match SurfaceChannelDefinition::with_nodes(setup, pools()?, pumped) {
            Err(SurfaceChannelError::Input(message)) => message,
            other => return Err(format!("a pump owner on a gate rejects: {other:?}").into()),
        };
        cases.push(("does not drive its law", law));
        let unknown = SurfaceChannelOperating {
            signals: vec![signal("outlet", channel())],
            ..with_gate(gate(&shared, 0.0)?)
        };
        cases.push((
            "node=\"reach:r:cell:0\" is no coordinator node",
            rejection(actuated()?, unknown)?,
        ));
        let unbound = SurfaceChannelOperating {
            signals: Vec::new(),
            ..with_gate(gate(&shared, 0.0)?)
        };
        cases.push(("has no binding", rejection(actuated()?, unbound)?));
        let stranger = SurfaceChannelOperating {
            signals: vec![signal("spill", pool("pond"))],
            ..with_gate(gate(&shared, 0.0)?)
        };
        cases.push((
            "names no controller signal",
            rejection(actuated()?, stranger)?,
        ));
        let twice = SurfaceChannelOperating {
            signals: vec![signal("outlet", pool("pond")), signal("outlet", surface())],
            ..with_gate(gate(&shared, 0.0)?)
        };
        cases.push(("is bound twice", rejection(actuated()?, twice)?));
        let advanced = gate(&shared, 0.0)?.trial_to(0.5, &[])?.into_next();
        cases.push((
            "accepted_steps=1 is not fresh",
            rejection(actuated()?, with_gate(advanced))?,
        ));
        cases.push((
            "time=0.5, common time_s=0",
            rejection(actuated()?, with_gate(gate(&shared, 0.5)?))?,
        ));
        for (reason, message) in &cases {
            assert!(message.starts_with("operation-ownership"), "{message}");
            assert!(message.contains(reason), "{reason}: {message}");
        }
        Ok(())
    }

    /// Frozen criteria, fixed before the first run:
    /// - A dynamic owner without operating children rejects with `operation-ownership` through
    ///   the manual pool constructor.
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
