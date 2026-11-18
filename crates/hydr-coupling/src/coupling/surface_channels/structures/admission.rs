//! Admission of structure links: identities, endpoints, datum, beds, direction, and operation.
//!
//! Each rejection names its capability entry first.

use super::{
    CheckedNode, CheckedStructure, SurfaceChannelAttachment, SurfaceChannelNode,
    SurfaceChannelOperationOwner, SurfaceChannelStructure,
};
use crate::coupling::links::cutoffs::pool_attachment_admitted;
use crate::coupling::surface_channels::bindings::{channel_cells, structure_binding_ids};
use crate::coupling::surface_channels::pools::CheckedPools;
use crate::coupling::surface_channels::validation::text;
use crate::coupling::surface_channels::{SurfaceChannelError, SurfaceChannelSetup};
use crate::structures::{DirectionPolicy, StructureLaw, StructureOperatingPoint};
use std::collections::{BTreeMap, BTreeSet};

/// Rejects with `entry` for structure `id`.
fn rejected(entry: &str, id: &str, detail: String) -> SurfaceChannelError {
    SurfaceChannelError::input(format!("{entry}: structure={id:?} {detail}"))
}

/// Checks every structure of `structures`. Returns them in ascending identity order.
///
/// `reach_index` maps each checked reach identity to its setup index. `pools` holds the checked
/// pools. `taken` holds every link, port, and pool identity.
pub(in super::super) fn check_structures(
    setup: &SurfaceChannelSetup,
    reach_index: &BTreeMap<&str, usize>,
    pools: Option<&CheckedPools>,
    taken: &BTreeSet<&str>,
    structures: Vec<SurfaceChannelStructure>,
) -> Result<Vec<CheckedStructure>, SurfaceChannelError> {
    let mut checked = BTreeMap::new();
    let mut scenarios = BTreeSet::new();
    for structure in structures {
        let id = structure.element.id.clone();
        let evidence = |name: &str, value: &str| {
            text(
                &format!("missing-link-evidence: structure={id:?} {name}"),
                value,
            )
        };
        evidence("id", &id)?;
        if taken.contains(id.as_str()) || checked.contains_key(&id) {
            return Err(rejected(
                "duplicate-links",
                &id,
                "repeats a structure, link, port, or pool identity: give each one a unique identity"
                    .into(),
            ));
        }
        evidence("scenario.id", &structure.scenario.id)?;
        evidence("scenario.evidence", &structure.scenario.evidence)?;
        if !scenarios.insert(structure.scenario.id.clone()) {
            return Err(rejected(
                "duplicate-links",
                &id,
                format!(
                    "scenario={:?} repeats: give each structure scenario a unique identity",
                    structure.scenario.id
                ),
            ));
        }
        for (name, attachment) in [
            ("endpoint_a", &structure.endpoint_a),
            ("endpoint_b", &structure.endpoint_b),
        ] {
            evidence(
                &format!("{name}.physical_connection_assertion"),
                &attachment.physical_connection_assertion,
            )?;
        }
        let nodes = [
            resolve(setup, reach_index, pools, "endpoint_a", &structure)?,
            resolve(setup, reach_index, pools, "endpoint_b", &structure)?,
        ];
        check_pair(&structure, &nodes)?;
        let beds = [
            structure.endpoint_a.bed_elevation_m,
            structure.endpoint_b.bed_elevation_m,
        ];
        check_law(&structure, beds, setup.surface.config.gravity)?;
        let operating_point = operation(&structure)?;
        let (inflow_binding, withdrawal_binding) = structure_binding_ids(&id);
        checked.insert(
            id,
            CheckedStructure {
                structure,
                nodes,
                beds,
                operating_point,
                inflow_binding,
                withdrawal_binding,
            },
        );
    }
    Ok(checked.into_values().collect())
}

/// Resolves endpoint `name` of `structure` and checks its identity, datum, and bed.
fn resolve(
    setup: &SurfaceChannelSetup,
    reach_index: &BTreeMap<&str, usize>,
    pools: Option<&CheckedPools>,
    name: &str,
    structure: &SurfaceChannelStructure,
) -> Result<CheckedNode, SurfaceChannelError> {
    let element = &structure.element;
    let id = element.id.as_str();
    let identity = |detail: String| rejected("structure-endpoint-identity", id, detail);
    let (attachment, endpoint): (&SurfaceChannelAttachment, &str) = if name == "endpoint_a" {
        (&structure.endpoint_a, &element.endpoint_a)
    } else {
        (&structure.endpoint_b, &element.endpoint_b)
    };
    let expected = attachment.node.identity();
    if endpoint != expected {
        return Err(identity(format!(
            "element {name}={endpoint:?} differs from node identity {expected:?}: name the node identity in the element"
        )));
    }
    let domain = &setup.surface.domain;
    let datum = domain.reference().vertical_datum();
    if element.datum != datum {
        return Err(rejected(
            "datum-or-gravity-mismatch",
            id,
            format!(
                "datum={:?}, surface vertical_datum={datum:?}: express the structure in the surface datum",
                element.datum
            ),
        ));
    }
    let bed = attachment.bed_elevation_m;
    let below = |what: &str, floor: f64| {
        rejected(
            "unobstructed-pool-connection",
            id,
            format!(
                "{name} bed_elevation_m={bed} lies below its {what}={floor} or is nonfinite: raise the attachment bed to at least the {what}"
            ),
        )
    };
    match &attachment.node {
        SurfaceChannelNode::SurfaceCell(cell) => {
            let cell_count = domain.grid().cell_count();
            if *cell >= cell_count || !domain.active()[*cell] {
                return Err(identity(format!(
                    "{name} surface_cell={cell}, cells={cell_count}: attach an active cell inside the grid"
                )));
            }
            let node_bed = domain.bed()[*cell];
            if !bed.is_finite() || bed < node_bed {
                return Err(below("node bed", node_bed));
            }
            Ok(CheckedNode::Surface(*cell))
        }
        SurfaceChannelNode::ChannelCell { reach_id, cell } => {
            let Some(&reach) = reach_index.get(reach_id.as_str()) else {
                return Err(identity(format!(
                    "{name} reach_id={reach_id:?} names no setup reach: use a declared reach identity"
                )));
            };
            let cells = channel_cells(&setup.reaches[reach].definition);
            let Some(&(cell_id, channel_bed)) = cells.get(*cell) else {
                return Err(identity(format!(
                    "{name} channel cell={cell}, reach={reach_id:?} cells={}: attach an existing reach cell",
                    cells.len()
                )));
            };
            if !bed.is_finite() || bed < channel_bed {
                return Err(below("node bed", channel_bed));
            }
            Ok(CheckedNode::Channel {
                reach_index: reach,
                cell: *cell,
                cell_id: cell_id.to_string(),
            })
        }
        SurfaceChannelNode::Pool(pool_id) => {
            let Some((position, pool)) = pools.and_then(|pools| pools.find(pool_id)) else {
                return Err(identity(format!(
                    "{name} pool={pool_id:?} names no declared pool: use a declared pool identity"
                )));
            };
            let origin = pool.curve.origin();
            if !pool_attachment_admitted(
                &element.datum,
                pool.curve.vertical_reference(),
                bed,
                origin,
            ) {
                return Err(below("pool origin", origin));
            }
            Ok(CheckedNode::Pool(position))
        }
    }
}

/// Checks that the two endpoints join distinct nodes of an admitted pair.
///
/// Admitted pairs join a surface cell and a channel cell, a surface cell and a pool, a channel cell
/// and a pool, or two pools.
fn check_pair(
    structure: &SurfaceChannelStructure,
    nodes: &[CheckedNode; 2],
) -> Result<(), SurfaceChannelError> {
    let id = structure.element.id.as_str();
    if structure.endpoint_a.node == structure.endpoint_b.node {
        return Err(rejected(
            "structure-endpoint-identity",
            id,
            "joins one node at both endpoints: attach the endpoints to two distinct nodes".into(),
        ));
    }
    let same = match nodes {
        [CheckedNode::Surface(_), CheckedNode::Surface(_)] => Some("surface cells"),
        [CheckedNode::Channel { .. }, CheckedNode::Channel { .. }] => Some("channel cells"),
        _ => None,
    };
    if let Some(kind) = same {
        return Err(rejected(
            "structure-endpoint-identity",
            id,
            format!(
                "joins two {kind}: join a surface cell and a channel cell, a surface cell or a channel cell and a pool, or two pools"
            ),
        ));
    }
    Ok(())
}

/// Checks the law's gravity, its control elevation against both attachment beds, and its
/// direction policy.
fn check_law(
    structure: &SurfaceChannelStructure,
    beds: [f64; 2],
    gravity: f64,
) -> Result<(), SurfaceChannelError> {
    let element = &structure.element;
    let id = element.id.as_str();
    let (control, symmetric, law_gravity) = match &element.law {
        StructureLaw::SuppressedRectangularWeir(p) => (
            Some(("crest", p.crest_m())),
            p.applicability().symmetric_reversal,
            None,
        ),
        StructureLaw::HydrostaticRectangularGate(p) => (
            Some(("sill", p.sill_m())),
            p.applicability().symmetric_reversal,
            Some(p.gravity_m_per_s2()),
        ),
        StructureLaw::SubmergedOrifice(p) => (
            Some(("sill", p.sill_m())),
            p.applicability().symmetric_reversal,
            Some(p.gravity_m_per_s2()),
        ),
        StructureLaw::Pump(_) => (None, false, None),
    };
    if let Some(law_gravity) = law_gravity
        && law_gravity.to_bits() != gravity.to_bits()
    {
        return Err(rejected(
            "datum-or-gravity-mismatch",
            id,
            format!(
                "gravity={law_gravity}, surface gravity={gravity}: supply bit-identical gravity"
            ),
        ));
    }
    if let Some((name, elevation)) = control
        && (elevation < beds[0] || elevation < beds[1])
    {
        return Err(rejected(
            "unobstructed-pool-connection",
            id,
            format!(
                "{name}={elevation} lies below attachment bed A={} or B={}: an unobstructed connection is unsupported; raise the {name} to at least both beds",
                beds[0], beds[1]
            ),
        ));
    }
    if element.direction == DirectionPolicy::Bidirectional && !symmetric {
        return Err(rejected(
            "non-symmetric-structure",
            id,
            if control.is_none() {
                "is a bidirectional pump: a pump moves water one way; set AtoBOnly or BtoAOnly"
                    .into()
            } else {
                "is bidirectional without symmetric reversal evidence: supply a coefficient that holds under reversed flow or set a one-way direction policy".into()
            },
        ));
    }
    Ok(())
}

/// The operation of `structure`, checked against its law kind.
///
/// A prescribed or actuated owner drives a gate, and a controlled-pump owner drives a pump. A
/// dynamic owner returns the closed state its operating child replaces on every attempt.
fn operation(
    structure: &SurfaceChannelStructure,
) -> Result<StructureOperatingPoint, SurfaceChannelError> {
    let law = &structure.element.law;
    let point = match (&structure.operation, law) {
        (SurfaceChannelOperationOwner::Manual(point), _) => point,
        (
            SurfaceChannelOperationOwner::Prescribed | SurfaceChannelOperationOwner::Actuated,
            StructureLaw::HydrostaticRectangularGate(_),
        ) => return Ok(StructureOperatingPoint::GateOpening { opening_m: 0.0 }),
        (SurfaceChannelOperationOwner::ControlledPump, StructureLaw::Pump(_)) => {
            return Ok(StructureOperatingPoint::PumpStoppedIsolated);
        }
        (owner, _) => {
            return Err(rejected(
                "operation-ownership",
                &structure.element.id,
                format!(
                    "owner={owner:?} does not drive its law: give a prescribed or actuated owner to a gate and a controlled-pump owner to a pump"
                ),
            ));
        }
    };
    let admitted = match law {
        StructureLaw::SuppressedRectangularWeir(_) | StructureLaw::SubmergedOrifice(_) => {
            matches!(point, StructureOperatingPoint::Passive)
        }
        StructureLaw::HydrostaticRectangularGate(_) => {
            matches!(point, StructureOperatingPoint::GateOpening { .. })
        }
        StructureLaw::Pump(_) => matches!(
            point,
            StructureOperatingPoint::PumpRunning { .. }
                | StructureOperatingPoint::PumpStoppedIsolated
        ),
    };
    if !admitted {
        return Err(rejected(
            "operation-ownership",
            &structure.element.id,
            format!(
                "operation={point:?} does not drive its law: give a weir or orifice Passive, a gate GateOpening, and a pump PumpRunning or PumpStoppedIsolated"
            ),
        ));
    }
    Ok(*point)
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{
        DT, INFLOW, REACH, Tanks, TestResult, WITHDRAWAL, gate_law, orifice_law, pump_law, step,
        structure, withdrawal,
    };
    use super::*;
    use crate::channel::ChannelLateralInput;
    use crate::coupling::surface_channels::SurfaceChannelDefinition;

    fn culvert() -> TestResult<SurfaceChannelStructure> {
        Ok(structure(
            "culvert",
            orifice_law(0.1, 0.3)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            [0.0, 0.0],
        ))
    }

    /// Input message of a definition of resting tanks with `structures`.
    fn rejection(structures: Vec<SurfaceChannelStructure>) -> TestResult<String> {
        let (mut setup, _) = Tanks::at(0.6, 0.6).parts()?;
        setup.structures = structures;
        match SurfaceChannelDefinition::new(setup) {
            Err(SurfaceChannelError::Input(message)) => Ok(message),
            Err(other) => Err(format!("expected an input rejection, got {other}").into()),
            Ok(_) => Err("the definition admitted an invalid structure".into()),
        }
    }

    #[test]
    fn structure_bindings_follow_the_caller_bindings() -> TestResult {
        let (mut setup, _) = Tanks::at(0.6, 0.6).parts()?;
        setup.structures = vec![culvert()?];
        let definition = SurfaceChannelDefinition::new(setup)?;
        let reach = definition.coordinated_reach(REACH).ok_or("the reach")?;
        let ids: Vec<&str> = reach
            .lateral_bindings
            .iter()
            .map(|b| b.boundary_id.as_str())
            .collect();
        let synthesized = [
            "structure-exchange:culvert:inflow",
            "structure-exchange:culvert:withdrawal",
        ];
        assert_eq!(ids, [INFLOW, WITHDRAWAL, synthesized[0], synthesized[1]]);
        assert_eq!(definition.surface_cells(), [0]);
        Ok(())
    }

    #[test]
    fn each_structure_rejection_names_its_entry() -> TestResult {
        let mut cases: Vec<(SurfaceChannelStructure, &str)> = Vec::new();
        let mut case = culvert()?;
        case.element.endpoint_b = "reach:r:cell:1".into();
        cases.push((case, "structure-endpoint-identity"));
        let mut case = culvert()?;
        case.endpoint_a.node = case.endpoint_b.node.clone();
        case.element.endpoint_a = case.element.endpoint_b.clone();
        cases.push((case, "structure-endpoint-identity"));
        let mut case = culvert()?;
        case.endpoint_b.node = SurfaceChannelNode::ChannelCell {
            reach_id: REACH.into(),
            cell: 3,
        };
        case.element.endpoint_b = case.endpoint_b.node.identity();
        cases.push((case, "structure-endpoint-identity"));
        let mut case = culvert()?;
        case.element.datum = "Another datum".into();
        cases.push((case, "datum-or-gravity-mismatch"));
        let mut case = culvert()?;
        case.endpoint_a.bed_elevation_m = -0.1;
        cases.push((case, "unobstructed-pool-connection"));
        let mut case = culvert()?;
        case.endpoint_b.bed_elevation_m = 0.2;
        cases.push((case, "unobstructed-pool-connection"));
        let mut case = culvert()?;
        case.element.law = pump_law(0.3, None)?;
        case.operation =
            SurfaceChannelOperationOwner::Manual(StructureOperatingPoint::PumpRunning {
                speed_ratio: 1.0,
            });
        cases.push((case, "non-symmetric-structure"));
        let mut case = culvert()?;
        case.element.law = gate_law(0.2)?;
        cases.push((case, "operation-ownership"));
        let mut case = culvert()?;
        case.operation = SurfaceChannelOperationOwner::Actuated;
        cases.push((case, "operation-ownership"));
        let mut case = culvert()?;
        case.element.law = gate_law(0.2)?;
        case.operation = SurfaceChannelOperationOwner::ControlledPump;
        cases.push((case, "operation-ownership"));
        let mut case = culvert()?;
        case.element.law = pump_law(0.3, None)?;
        case.element.direction = DirectionPolicy::AtoBOnly;
        case.operation = SurfaceChannelOperationOwner::Prescribed;
        cases.push((case, "operation-ownership"));
        let mut case = culvert()?;
        case.scenario.evidence = " ".into();
        cases.push((case, "missing-link-evidence"));
        for (case, entry) in cases {
            let message = rejection(vec![case])?;
            assert!(message.starts_with(entry), "{entry}: {message}");
        }
        let message = rejection(vec![culvert()?, culvert()?])?;
        assert!(message.starts_with("duplicate-links"), "{message}");
        Ok(())
    }

    #[test]
    fn callers_never_use_the_structure_exchange_prefix() -> TestResult {
        let (mut setup, _) = Tanks::at(0.6, 0.6).parts()?;
        setup.reaches[0].definition.lateral_bindings[0].boundary_id = "structure-exchange:x".into();
        let message = match SurfaceChannelDefinition::new(setup) {
            Err(SurfaceChannelError::Input(message)) => message,
            other => return Err(format!("expected an input rejection, got {other:?}").into()),
        };
        assert!(message.starts_with("reserved-binding-prefix"), "{message}");
        let batch = Tanks::at(0.6, 0.6).build(vec![culvert()?])?;
        let ChannelLateralInput::Withdrawal { amount, .. } = withdrawal(0.01) else {
            return Err("a withdrawal".into());
        };
        let forged = ChannelLateralInput::Withdrawal {
            boundary_id: "structure-exchange:culvert:withdrawal".into(),
            amount,
        };
        match step(&batch, DT, &[forged]) {
            Err(SurfaceChannelError::Input(message)) => {
                assert!(message.starts_with("reserved-binding-prefix"), "{message}");
            }
            other => return Err(format!("expected an input rejection, got {other:?}").into()),
        }
        Ok(())
    }
}
