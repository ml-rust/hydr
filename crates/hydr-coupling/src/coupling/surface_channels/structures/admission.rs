//! Admission of structure links: identities, endpoints, datum, beds, direction, and operation.
//!
//! Each rejection names its capability entry first.

use super::{
    CheckedStructure, SurfaceChannelNode, SurfaceChannelOperationOwner, SurfaceChannelStructure,
};
use crate::coupling::surface_channels::bindings::{channel_cells, structure_binding_ids};
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
/// `reach_index` maps each checked reach identity to its setup index. `taken` holds every link
/// and port identity.
pub(in super::super) fn check_structures(
    setup: &SurfaceChannelSetup,
    reach_index: &BTreeMap<&str, usize>,
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
                "repeats a structure, link, or port identity: give each one a unique identity"
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
        let resolved = resolve(setup, reach_index, &structure)?;
        check_law(&structure, resolved.beds, setup.surface.config.gravity)?;
        let operating_point = operation(&structure)?;
        let (inflow_binding, withdrawal_binding) = structure_binding_ids(&id);
        checked.insert(
            id,
            CheckedStructure {
                structure,
                surface_cell: resolved.surface_cell,
                reach_index: resolved.reach_index,
                channel_cell: resolved.channel_cell,
                cell_id: resolved.cell_id,
                beds: resolved.beds,
                operating_point,
                inflow_binding,
                withdrawal_binding,
            },
        );
    }
    Ok(checked.into_values().collect())
}

/// Node indices and attachment beds of one structure.
struct Resolved {
    surface_cell: usize,
    reach_index: usize,
    channel_cell: usize,
    cell_id: String,
    beds: [f64; 2],
}

/// Resolves both endpoints and checks their identities, datum, and beds.
fn resolve(
    setup: &SurfaceChannelSetup,
    reach_index: &BTreeMap<&str, usize>,
    structure: &SurfaceChannelStructure,
) -> Result<Resolved, SurfaceChannelError> {
    let element = &structure.element;
    let id = element.id.as_str();
    let identity = |detail: String| rejected("structure-endpoint-identity", id, detail);
    let (
        SurfaceChannelNode::SurfaceCell(surface_cell),
        SurfaceChannelNode::ChannelCell { reach_id, cell },
    ) = (&structure.endpoint_a.node, &structure.endpoint_b.node)
    else {
        return Err(identity(
            "joins no surface cell at endpoint A and channel cell at endpoint B: attach A to a surface cell and B to a channel cell; set the direction policy for one-way flow".into(),
        ));
    };
    for (name, node, endpoint) in [
        (
            "endpoint_a",
            &structure.endpoint_a.node,
            &element.endpoint_a,
        ),
        (
            "endpoint_b",
            &structure.endpoint_b.node,
            &element.endpoint_b,
        ),
    ] {
        let expected = node.identity();
        if *endpoint != expected {
            return Err(identity(format!(
                "element {name}={endpoint:?} differs from node identity {expected:?}: name the node identity in the element"
            )));
        }
    }
    let domain = &setup.surface.domain;
    let cell_count = domain.grid().cell_count();
    if *surface_cell >= cell_count || !domain.active()[*surface_cell] {
        return Err(identity(format!(
            "surface_cell={surface_cell}, cells={cell_count}: attach an active cell inside the grid"
        )));
    }
    let Some(&reach) = reach_index.get(reach_id.as_str()) else {
        return Err(identity(format!(
            "reach_id={reach_id:?} names no setup reach: use a declared reach identity"
        )));
    };
    let cells = channel_cells(&setup.reaches[reach].definition);
    let Some(&(cell_id, channel_bed)) = cells.get(*cell) else {
        return Err(identity(format!(
            "channel cell={cell}, reach={reach_id:?} cells={}: attach an existing reach cell",
            cells.len()
        )));
    };
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
    let beds = [
        structure.endpoint_a.bed_elevation_m,
        structure.endpoint_b.bed_elevation_m,
    ];
    let node_beds = [domain.bed()[*surface_cell], channel_bed];
    for (name, (bed, node)) in ["endpoint_a", "endpoint_b"]
        .iter()
        .zip(beds.iter().zip(node_beds))
    {
        if !bed.is_finite() || *bed < node {
            return Err(rejected(
                "unobstructed-pool-connection",
                id,
                format!(
                    "{name} bed_elevation_m={bed} lies below its node bed={node} or is nonfinite: raise the attachment bed to at least the node bed"
                ),
            ));
        }
    }
    Ok(Resolved {
        surface_cell: *surface_cell,
        reach_index: reach,
        channel_cell: *cell,
        cell_id: cell_id.to_string(),
        beds,
    })
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
fn operation(
    structure: &SurfaceChannelStructure,
) -> Result<StructureOperatingPoint, SurfaceChannelError> {
    let SurfaceChannelOperationOwner::Manual(point) = &structure.operation;
    let admitted = match &structure.element.law {
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
        let (setup, _) = Tanks::at(0.6, 0.6).parts()?;
        match SurfaceChannelDefinition::with_structures(setup, structures) {
            Err(SurfaceChannelError::Input(message)) => Ok(message),
            Err(other) => Err(format!("expected an input rejection, got {other}").into()),
            Ok(_) => Err("the definition admitted an invalid structure".into()),
        }
    }

    #[test]
    fn structure_bindings_follow_the_caller_bindings() -> TestResult {
        let (setup, _) = Tanks::at(0.6, 0.6).parts()?;
        let definition = SurfaceChannelDefinition::with_structures(setup, vec![culvert()?])?;
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
        let message = match SurfaceChannelDefinition::with_structures(setup, Vec::new()) {
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
