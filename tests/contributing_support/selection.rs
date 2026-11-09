use super::fixture::{Fixture, Result, face, measured, standalone_measured, suppressed};
use hydr::runoff::{
    BoundaryBinding, Contribution, ContributionKind, SupportAtom, WholeReplacement,
};
use hydr::solver::{Axis, BoundaryConditions, BoundaryFace};

fn leaf(id: &str, kind: ContributionKind) -> Contribution {
    Contribution {
        id: id.into(),
        input_identity: format!("{id}-input"),
        provenance: "Explicit component already included in the composed hydrograph".into(),
        atoms: vec![],
        includes: vec![],
        kind,
    }
}

fn right() -> BoundaryFace {
    BoundaryFace {
        axis: Axis::X,
        column: 2,
        row: 0,
    }
}

fn two_imported_roots() -> Result<Fixture> {
    let mut fixture = standalone_measured()?;
    fixture.boundaries =
        BoundaryConditions::with_open_faces(&fixture.domain, vec![face(), right()])?;
    fixture.input.contributions = vec![
        Contribution {
            id: "root-a".into(),
            input_identity: "input-a".into(),
            provenance: "Composed upstream hydrograph A".into(),
            atoms: vec!["a".into()],
            includes: vec![],
            kind: ContributionKind::ImportedRunoff {
                boundary_id: "boundary-a".into(),
            },
        },
        Contribution {
            id: "root-b".into(),
            input_identity: "input-b".into(),
            provenance: "Composed upstream hydrograph B".into(),
            atoms: vec!["b".into()],
            includes: vec![],
            kind: ContributionKind::ImportedRunoff {
                boundary_id: "boundary-b".into(),
            },
        },
    ];
    fixture.input.surface_cells = vec![suppressed(0, "a", "root-a"), suppressed(1, "b", "root-b")];
    fixture.input.boundaries = vec![
        BoundaryBinding {
            id: "boundary-a".into(),
            face: face(),
            input_identity: "input-a".into(),
            provenance: "Explicit upstream boundary A".into(),
            contribution: Some("root-a".into()),
        },
        BoundaryBinding {
            id: "boundary-b".into(),
            face: right(),
            input_identity: "input-b".into(),
            provenance: "Explicit upstream boundary B".into(),
            contribution: Some("root-b".into()),
        },
    ];
    Ok(fixture)
}

#[test]
fn partial_excess_duplicate_and_unknown_whole_replacements_reject() -> Result {
    let baseline = measured()?;
    baseline.plan()?;
    for mode in 0..6 {
        let mut fixture = measured()?;
        match mode {
            0 => {
                fixture.input.contributions[2]
                    .atoms
                    .pop()
                    .ok_or("missing measured atom")?;
            }
            1 => {
                fixture.input.atoms.push(SupportAtom {
                    id: "extra".into(),
                    area_m2: 1.0,
                });
                fixture.input.contributions[2].atoms.push("extra".into());
            }
            2 => {
                fixture.input.replacements[0]
                    .replaced
                    .pop()
                    .ok_or("missing replaced source")?;
            }
            3 => fixture.input.replacements[0]
                .replaced
                .push("source-a".into()),
            4 => fixture.input.replacements[0].replaced[0] = "unknown".into(),
            5 => fixture.input.replacements[0].selected = "source-a".into(),
            _ => return Err("unknown replacement fixture".into()),
        }
        assert!(fixture.plan().is_err(), "replacement mode={mode}");
    }
    Ok(())
}

#[test]
fn conflicting_and_cyclic_replacement_selection_rejects() -> Result {
    let mut conflict = measured()?;
    conflict.boundaries =
        BoundaryConditions::with_open_faces(&conflict.domain, vec![face(), right()])?;
    conflict.input.contributions.push(Contribution {
        id: "other-measurement".into(),
        input_identity: "other-input".into(),
        provenance: "Second whole measured replacement".into(),
        atoms: vec!["a".into()],
        includes: vec![],
        kind: ContributionKind::MeasuredFlow {
            boundary_id: "other-boundary".into(),
        },
    });
    conflict.input.boundaries.push(BoundaryBinding {
        id: "other-boundary".into(),
        face: right(),
        input_identity: "other-input".into(),
        provenance: "Second synthetic boundary".into(),
        contribution: Some("other-measurement".into()),
    });
    conflict.input.replacements.push(WholeReplacement {
        selected: "other-measurement".into(),
        replaced: vec!["source-a".into()],
    });
    assert!(conflict.plan().is_err());
    let mut cycle = measured()?;
    cycle.input.replacements[0]
        .replaced
        .push("measurement".into());
    assert!(cycle.plan().is_err());
    Ok(())
}

#[test]
fn baseflow_and_release_are_included_once_in_one_unchanged_physical_input() -> Result {
    let mut fixture = standalone_measured()?;
    fixture.input.contributions[0].includes = vec!["base".into(), "release".into()];
    fixture
        .input
        .contributions
        .push(leaf("base", ContributionKind::Baseflow));
    fixture
        .input
        .contributions
        .push(leaf("release", ContributionKind::Release));
    let plan = fixture.plan()?;
    let expected = fixture.interval.boundaries[0].value.clone();
    let packet = plan.admit_interval(fixture.interval)?;
    assert_eq!(packet.forcing().boundaries, vec![expected]);
    assert!(plan.trial_subcatchments(&packet, None)?.is_none());
    Ok(())
}

#[test]
fn orphan_duplicate_shared_and_nonleaf_inclusions_reject() -> Result {
    let mut orphan = standalone_measured()?;
    orphan
        .input
        .contributions
        .push(leaf("base", ContributionKind::Baseflow));
    assert!(orphan.plan().is_err());
    let mut duplicate = standalone_measured()?;
    duplicate.input.contributions[0].includes = vec!["base".into(), "base".into()];
    duplicate
        .input
        .contributions
        .push(leaf("base", ContributionKind::Baseflow));
    assert!(duplicate.plan().is_err());
    let mut shared = two_imported_roots()?;
    shared.plan()?;
    shared.input.contributions[0].includes = vec!["base".into()];
    shared.input.contributions[1].includes = vec!["base".into()];
    shared
        .input
        .contributions
        .push(leaf("base", ContributionKind::Baseflow));
    assert!(shared.plan().is_err());
    let mut nonleaf = standalone_measured()?;
    nonleaf.input.contributions[0].includes = vec!["measurement".into()];
    assert!(nonleaf.plan().is_err());
    let mut modelled = super::fixture::primary()?;
    modelled.input.contributions[0].includes = vec!["base".into()];
    modelled
        .input
        .contributions
        .push(leaf("base", ContributionKind::Baseflow));
    assert!(modelled.plan().is_err());
    Ok(())
}

#[test]
fn replaced_imported_parent_cannot_leave_an_unowned_additive_leaf() -> Result {
    let mut fixture = standalone_measured()?;
    fixture.input.contributions.push(Contribution {
        id: "imported".into(),
        input_identity: "gauge-input".into(),
        provenance: "Former already composed imported source".into(),
        atoms: vec!["a".into(), "b".into()],
        includes: vec!["base".into()],
        kind: ContributionKind::ImportedRunoff {
            boundary_id: "gauge".into(),
        },
    });
    fixture
        .input
        .contributions
        .push(leaf("base", ContributionKind::Baseflow));
    fixture.input.replacements = vec![WholeReplacement {
        selected: "measurement".into(),
        replaced: vec!["imported".into()],
    }];
    assert!(fixture.plan().is_err());
    fixture.input.contributions[0].includes = vec!["base".into()];
    let plan = fixture.plan()?;
    let expected = fixture.interval.boundaries[0].value.clone();
    let packet = plan.admit_interval(fixture.interval)?;
    assert_eq!(packet.forcing().boundaries, vec![expected]);
    Ok(())
}

#[test]
fn included_leaf_cannot_inject_independently_or_carry_primary_atoms() -> Result {
    let mut fixture = standalone_measured()?;
    fixture.input.contributions[0].includes = vec!["base".into()];
    fixture
        .input
        .contributions
        .push(leaf("base", ContributionKind::Baseflow));
    fixture.plan()?;
    let mut injected = fixture.input.clone();
    injected.boundaries.push(BoundaryBinding {
        id: "base-boundary".into(),
        face: right(),
        input_identity: "base-input".into(),
        provenance: "Attempted independent leaf injection".into(),
        contribution: Some("base".into()),
    });
    let boundaries = BoundaryConditions::with_open_faces(&fixture.domain, vec![face(), right()])?;
    assert!(
        hydr::runoff::SupportPlan::new(
            injected,
            &fixture.domain,
            &boundaries,
            &fixture.definitions
        )
        .is_err()
    );
    fixture.input.contributions[1].atoms = vec!["a".into()];
    assert!(fixture.plan().is_err());
    Ok(())
}

#[test]
fn every_selected_imported_root_requires_its_exact_uncomposed_boundary() -> Result {
    let baseline = two_imported_roots()?;
    baseline.plan()?;
    for mode in 0..4 {
        let mut fixture = two_imported_roots()?;
        match mode {
            0 => {
                fixture.input.boundaries.pop();
            }
            1 => fixture.input.boundaries[0].input_identity = "other-input".into(),
            2 => {
                fixture.input.contributions[0].kind = ContributionKind::ImportedRunoff {
                    boundary_id: "boundary-b".into(),
                }
            }
            3 => fixture.input.boundaries[1].face = face(),
            _ => return Err("unknown imported-boundary fixture".into()),
        }
        assert!(fixture.plan().is_err(), "boundary mode={mode}");
    }
    Ok(())
}
