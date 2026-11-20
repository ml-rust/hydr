//! Admission of pools, structure links, pool ends, and operating children: one test per
//! rejection that the capability declarations list, each matching its capability entry.

use crate::pools_fixture::{
    Batch, Case, PoolForcing, Result, channel, controlled_gate, gate_law, manual, operating,
    orifice_law, pond, pool, pool_end, pump_law, signal, step, structure, surface, surveyed_reach,
    weir_law,
};
use hydr::channel::ChannelEnd;
use hydr::coupling::{
    SurfaceChannelDefinition, SurfaceChannelError, SurfaceChannelNode, SurfaceChannelOperating,
    SurfaceChannelOperationOwner,
};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint};

/// A wet pond above a dry surface cell, joined by one passive spill weir.
fn ponded() -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.5);
    case.ponds = vec![(pond("pond", 0.0, 8.0, 2.0)?, 0.6 * 8.0)];
    case.structures = vec![structure(
        "spill",
        weir_law(0.5, 1.0)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (pool("pond"), 0.0),
        (surface(), 0.0),
    )];
    Ok(case)
}

/// The input message of building `case`, failing when it admits.
fn rejection(case: &Case) -> Result<String> {
    let (setup, initial) = case.parts()?;
    match Batch::new(crate::runtime::client(), setup, initial) {
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
        Ok(_) => Err("the coordinator admitted an invalid case".into()),
    }
}

/// Fails unless every edit of the ponded case rejects with a message that starts with `entry`.
fn rejects_each(entry: &str, edits: Vec<Edit>) -> Result {
    ponded()?.build()?;
    for (index, edit) in edits.iter().enumerate() {
        let mut case = ponded()?;
        edit(&mut case)?;
        let message = rejection(&case)?;
        assert!(message.starts_with(entry), "case {index}: {message}");
    }
    Ok(())
}

type Edit = Box<dyn Fn(&mut Case) -> Result>;

/// Frozen criterion: an element endpoint string other than its node identity, an unknown pool,
/// a missing channel cell, a surface cell outside the grid, and both endpoints on one node each
/// reject with `structure-endpoint-identity`.
#[test]
fn structure_endpoints_must_name_their_nodes() -> Result {
    let edits: Vec<Edit> = vec![
        Box::new(|c| {
            c.structures[0].element.endpoint_b = "surface-cell:1".into();
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].endpoint_a.node = pool("ghost");
            c.structures[0].element.endpoint_a = pool("ghost").identity();
            Ok(())
        }),
        Box::new(|c| {
            let node = SurfaceChannelNode::ChannelCell {
                reach_id: "r".into(),
                cell: 3,
            };
            c.structures[0].element.endpoint_b = node.identity();
            c.structures[0].endpoint_b.node = node;
            Ok(())
        }),
        Box::new(|c| {
            let node = SurfaceChannelNode::SurfaceCell(5);
            c.structures[0].element.endpoint_b = node.identity();
            c.structures[0].endpoint_b.node = node;
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].element.endpoint_b = pool("pond").identity();
            c.structures[0].endpoint_b.node = pool("pond");
            Ok(())
        }),
    ];
    rejects_each("structure-endpoint-identity", edits)
}

/// Frozen criterion: an element datum other than the surface datum rejects with
/// `datum-or-gravity-mismatch`.
#[test]
fn structures_share_the_surface_datum() -> Result {
    let edits: Vec<Edit> = vec![Box::new(|c| {
        c.structures[0].element.datum = "Other datum".into();
        Ok(())
    })];
    rejects_each("datum-or-gravity-mismatch", edits)
}

/// Frozen criterion: an attachment bed below its pool origin or node bed, and a crest, gate
/// sill, or orifice sill below an attachment bed, each reject with
/// `unobstructed-pool-connection`.
#[test]
fn attachments_stay_obstructed() -> Result {
    let edits: Vec<Edit> = vec![
        Box::new(|c| {
            c.structures[0].endpoint_a.bed_elevation_m = -0.1;
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].endpoint_b.bed_elevation_m = -0.1;
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].element.law = weir_law(0.5, 1.0)?;
            c.structures[0].endpoint_b.bed_elevation_m = 0.6;
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].element.law = gate_law(0.1)?;
            c.structures[0].operation =
                manual(StructureOperatingPoint::GateOpening { opening_m: 0.2 });
            c.structures[0].endpoint_a.bed_elevation_m = 0.2;
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].element.law = orifice_law(0.1, 0.3, true)?;
            c.structures[0].endpoint_b.bed_elevation_m = 0.2;
            Ok(())
        }),
    ];
    rejects_each("unobstructed-pool-connection", edits)
}

/// Frozen criterion: a bidirectional pump and a bidirectional orifice without symmetric
/// reversal evidence each reject with `non-symmetric-structure`.
#[test]
fn bidirectional_laws_need_symmetric_evidence() -> Result {
    let edits: Vec<Edit> = vec![
        Box::new(|c| {
            c.structures[0].element.law = pump_law(0.1, None)?;
            c.structures[0].operation =
                manual(StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 });
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].element.law = orifice_law(0.0, 0.2, false)?;
            Ok(())
        }),
    ];
    rejects_each("non-symmetric-structure", edits)
}

/// Frozen criterion: two pool ends on one reach end reject with `reach-end-ownership`, and so
/// does a caller policy at a pool end in a trial.
#[test]
fn a_reach_end_has_one_owner() -> Result {
    let edits: Vec<Edit> = vec![Box::new(|c| {
        c.pool_ends = vec![
            pool_end("one", ChannelEnd::Right, "pond"),
            pool_end("two", ChannelEnd::Right, "pond"),
        ];
        Ok(())
    })];
    rejects_each("reach-end-ownership", edits)?;
    let mut case = ponded()?;
    case.pool_ends = vec![pool_end("outfall", ChannelEnd::Right, "pond")];
    let batch = case.build()?;
    let mut held = crate::pools_fixture::forcing(&batch, 0.5, &PoolForcing::default());
    held.reaches[0].right = Some(hydr::channel::ChannelBoundary::Wall);
    match batch.trial_to(0.5, &held) {
        Err(SurfaceChannelError::Input(message)) => {
            assert!(message.starts_with("reach-end-ownership"), "{message}");
        }
        other => return Err(format!("a caller policy at a pool end rejects: {other:?}").into()),
    }
    Ok(())
}

/// Frozen criterion: a pool end on a surveyed reach admits and holds that one pool end.
#[test]
fn surveyed_reaches_take_a_pool_end() -> Result {
    let mut case = ponded()?;
    case.pool_ends = vec![pool_end("outfall", ChannelEnd::Right, "pond")];
    let (mut setup, initial) = case.parts()?;
    setup.reaches[0].definition = surveyed_reach(0.0);
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    assert_eq!(batch.pool_end_ledgers()?.len(), 1);
    Ok(())
}

/// Frozen criterion: support for another pool, a zero-volume stage off the curve origin, a
/// pool that joins nothing, and support without pools each reject with `pool-support`.
#[test]
fn pool_support_matches_the_pools() -> Result {
    let support = |edit: fn(&mut hydr::coupling::SurfaceChannelSetup)| -> Result<String> {
        let (mut setup, initial) = ponded()?.parts()?;
        edit(&mut setup);
        match Batch::new(crate::runtime::client(), setup, initial) {
            Err(SurfaceChannelError::Input(message)) => Ok(message),
            other => Err(format!("expected an input rejection, got {other:?}").into()),
        }
    };
    let messages = [
        support(|s| s.pool_support[0].pool_id = "other".into())?,
        support(|s| s.pool_support[0].zero_volume_stage_m = 0.1)?,
        support(|s| {
            s.structures[0].endpoint_a.node = surface();
            s.structures[0].element.endpoint_a = surface().identity();
            s.structures[0].endpoint_b.node = channel();
            s.structures[0].element.endpoint_b = channel().identity();
        })?,
    ];
    for message in messages {
        assert!(message.starts_with("pool-support"), "{message}");
    }
    let mut case = Case::tanks(0.0, 0.5);
    case.structures = ponded()?.structures;
    case.structures[0].endpoint_a.node = surface();
    case.structures[0].element.endpoint_a = surface().identity();
    case.structures[0].endpoint_b.node = channel();
    case.structures[0].element.endpoint_b = channel().identity();
    let (mut setup, initial) = case.parts()?;
    setup.pool_support = ponded()?.parts()?.0.pool_support;
    match Batch::new(crate::runtime::client(), setup, initial) {
        Err(SurfaceChannelError::Input(message)) => {
            assert!(message.starts_with("pool-support"), "{message}");
            Ok(())
        }
        other => Err(format!("support without pools rejects: {other:?}").into()),
    }
}

/// Frozen criterion: a repeated pool, structure, pool-end, or scenario identity, and a
/// structure named like a pool, each reject with `duplicate-links`.
#[test]
fn identities_stay_unique() -> Result {
    let edits: Vec<Edit> = vec![
        Box::new(|c| {
            let again = c.ponds[0].clone();
            c.ponds.push(again);
            Ok(())
        }),
        Box::new(|c| {
            let again = c.structures[0].clone();
            c.structures.push(again);
            Ok(())
        }),
        Box::new(|c| {
            c.pool_ends = vec![
                pool_end("outfall", ChannelEnd::Right, "pond"),
                pool_end("outfall", ChannelEnd::Left, "pond"),
            ];
            Ok(())
        }),
        Box::new(|c| {
            let mut other = c.structures[0].clone();
            other.element.id = "other".into();
            c.structures.push(other);
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].element.id = "pond".into();
            Ok(())
        }),
    ];
    rejects_each("duplicate-links", edits)
}

/// Frozen criterion: empty scenario evidence, an empty scenario identity, and empty
/// connection evidence each reject with `missing-link-evidence`.
#[test]
fn evidence_stays_nonempty() -> Result {
    let edits: Vec<Edit> = vec![
        Box::new(|c| {
            c.structures[0].scenario.evidence = String::new();
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].scenario.id = " ".into();
            Ok(())
        }),
        Box::new(|c| {
            c.structures[0].endpoint_a.physical_connection_assertion = String::new();
            Ok(())
        }),
    ];
    rejects_each("missing-link-evidence", edits)
}

/// Frozen criterion: a caller binding under `structure-exchange:` rejects with
/// `reserved-binding-prefix`.
#[test]
fn callers_never_bind_structure_exchange() -> Result {
    let (mut setup, initial) = ponded()?.parts()?;
    setup.reaches[0].definition.lateral_bindings[0].boundary_id = "structure-exchange:x".into();
    match Batch::new(crate::runtime::client(), setup, initial) {
        Err(SurfaceChannelError::Input(message)) => {
            assert!(message.starts_with("reserved-binding-prefix"), "{message}");
            Ok(())
        }
        other => Err(format!("a reserved binding rejects: {other:?}").into()),
    }
}

/// An actuated outlet gate from the pond onto the surface with its relay on the pond stage.
fn actuated() -> Result<Case> {
    let mut case = ponded()?;
    case.structures.push(structure(
        "outlet",
        gate_law(0.0)?,
        DirectionPolicy::Bidirectional,
        SurfaceChannelOperationOwner::Actuated,
        (pool("pond"), 0.0),
        (surface(), 0.0),
    ));
    case.operating = Some(SurfaceChannelOperating {
        gates: Some(controlled_gate(
            "outlet",
            [0.4, 0.7],
            ([0.0, 0.3], 0.1),
            (0.0, &[1.0, 2.0], 4.0),
        )),
        ..operating(vec![signal("outlet", pool("pond"))])
    });
    Ok(case)
}

/// Frozen criterion: each operation-ownership row rejects with `operation-ownership`: a
/// dynamic owner without operating children, a declaration without a child, an actuated owner
/// on a weir, a structure with a second owner, a dynamic owner its child does not claim, a
/// signal on an unknown node, a controller signal without a binding, and a child off the
/// common time.
#[test]
fn each_structure_has_one_operation_owner() -> Result {
    actuated()?.build()?;
    let mut cases = Vec::new();
    let mut case = actuated()?;
    case.operating = None;
    cases.push(case);
    let mut case = actuated()?;
    case.operating = Some(operating(Vec::new()));
    cases.push(case);
    let mut case = actuated()?;
    case.structures[0].operation = SurfaceChannelOperationOwner::Actuated;
    cases.push(case);
    let mut case = actuated()?;
    case.structures[1].operation = manual(StructureOperatingPoint::GateOpening { opening_m: 0.1 });
    cases.push(case);
    let mut case = actuated()?;
    case.structures[1].operation = SurfaceChannelOperationOwner::Prescribed;
    cases.push(case);
    let mut case = actuated()?;
    if let Some(declared) = case.operating.as_mut() {
        declared.signals = vec![signal("outlet", channel())];
    }
    cases.push(case);
    let mut case = actuated()?;
    if let Some(declared) = case.operating.as_mut() {
        declared.signals = Vec::new();
    }
    cases.push(case);
    let mut case = actuated()?;
    case.operating = Some(SurfaceChannelOperating {
        gates: Some(controlled_gate(
            "outlet",
            [0.4, 0.7],
            ([0.0, 0.3], 0.1),
            (0.5, &[1.0, 2.0], 4.0),
        )),
        ..operating(vec![signal("outlet", pool("pond"))])
    });
    cases.push(case);
    for (index, case) in cases.iter().enumerate() {
        let message = rejection(case)?;
        assert!(
            message.starts_with("operation-ownership"),
            "case {index}: {message}"
        );
    }
    Ok(())
}

/// The first-trial input message of `case`, failing on any other outcome.
fn trial_rejection(case: &Case) -> Result<String> {
    let batch = case.build()?;
    let before = crate::pools_fixture::state(&batch)?;
    let outcome = step(&batch, 0.5, &PoolForcing::default());
    assert_eq!(crate::pools_fixture::state(&batch)?, before);
    match outcome {
        Err(error) => match error.downcast::<SurfaceChannelError>() {
            Ok(error) => match *error {
                SurfaceChannelError::Input(message) => Ok(message),
                other => Err(format!("expected an input rejection, got {other}").into()),
            },
            Err(other) => Err(other),
        },
        Ok(_) => Err("the trial admitted an invalid state".into()),
    }
}

/// Frozen criterion: a running pump whose donor stands at its suction bound rejects with
/// `pump-without-suction`, and a running pump into a dry surface cell without an outlet
/// rejects with `pump-dry-delivery`. Neither retries, and the batch stays unchanged.
#[test]
fn running_pumps_need_suction_and_a_delivery() -> Result {
    let running = manual(StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 });
    let mut case = ponded()?;
    case.structures[0] = structure(
        "lift",
        pump_law(0.6, None)?,
        DirectionPolicy::AtoBOnly,
        running.clone(),
        (pool("pond"), 0.0),
        (channel(), 0.0),
    );
    let message = trial_rejection(&case)?;
    assert!(message.starts_with("pump-without-suction"), "{message}");
    let mut case = ponded()?;
    case.structures[0] = structure(
        "lift",
        pump_law(0.1, None)?,
        DirectionPolicy::AtoBOnly,
        running,
        (pool("pond"), 0.0),
        (surface(), 0.0),
    );
    let message = trial_rejection(&case)?;
    assert!(message.starts_with("pump-dry-delivery"), "{message}");
    Ok(())
}

/// Frozen criterion: definition checks need no runtime: a pump owner on a gate rejects at
/// `SurfaceChannelDefinition::new` with `operation-ownership`.
#[test]
fn definitions_check_operation_laws_without_a_runtime() -> Result {
    let mut case = actuated()?;
    case.structures[1].operation = SurfaceChannelOperationOwner::ControlledPump;
    let (setup, _) = case.parts()?;
    match SurfaceChannelDefinition::new(setup) {
        Err(SurfaceChannelError::Input(message)) => {
            assert!(message.starts_with("operation-ownership"), "{message}");
            Ok(())
        }
        other => Err(format!("a pump owner on a gate rejects: {other:?}").into()),
    }
}
