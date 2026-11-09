use super::fixture::{Fixture, Result, definition, domain, model, ordered, primary, suppressed};
use hydr::runoff::{
    CellRainAssignment, ContributionKind, SupportAtom, SupportErrorKind, SurfaceCellSupport,
};

fn rejects(fixture: &Fixture) -> Result {
    assert!(fixture.plan().is_err());
    Ok(())
}

#[test]
fn duplicate_ids_definition_aliases_and_outlet_aliases_reject() -> Result {
    let mut duplicate_source = primary()?;
    duplicate_source.plan()?;
    duplicate_source
        .input
        .contributions
        .push(duplicate_source.input.contributions[0].clone());
    rejects(&duplicate_source)?;
    let mut duplicate_atom = primary()?;
    duplicate_atom
        .input
        .atoms
        .push(duplicate_atom.input.atoms[0].clone());
    rejects(&duplicate_atom)?;
    let mut duplicate_definition = primary()?;
    duplicate_definition
        .definitions
        .push(duplicate_definition.definitions[0].clone());
    rejects(&duplicate_definition)?;
    let mut alias = primary()?;
    alias
        .input
        .contributions
        .push(model("alias", "upstream-model", "local"));
    alias.input.surface_cells[1] = suppressed(1, "local", "alias");
    rejects(&alias)?;
    let mut duplicate_outlet = ordered()?;
    if let ContributionKind::ModelledRunoff { outlet_id, .. } =
        &mut duplicate_outlet.input.contributions[1].kind
    {
        *outlet_id = "source-a-outlet".into();
    }
    rejects(&duplicate_outlet)
}

#[test]
fn surface_footprints_cannot_repeat_atoms_even_when_both_cells_are_suppressed() -> Result {
    let mut fixture = primary()?;
    fixture.input.atoms.retain(|atom| atom.id == "upstream");
    fixture.input.surface_cells[1] = suppressed(1, "upstream", "upstream-source");
    rejects(&fixture)?;
    let mut duplicate_cell = primary()?;
    duplicate_cell
        .input
        .surface_cells
        .push(duplicate_cell.input.surface_cells[0].clone());
    rejects(&duplicate_cell)
}

#[test]
fn every_atom_and_active_cell_requires_complete_known_ownership() -> Result {
    let mut unknown = primary()?;
    unknown.input.surface_cells[0].atoms[0] = "unknown".into();
    rejects(&unknown)?;
    let mut orphan = primary()?;
    orphan.input.atoms.push(SupportAtom {
        id: "orphan".into(),
        area_m2: 1.0,
    });
    rejects(&orphan)?;
    let mut missing_cell = primary()?;
    missing_cell
        .input
        .surface_cells
        .pop()
        .ok_or("missing active cell")?;
    rejects(&missing_cell)?;
    let mut unknown_owner = primary()?;
    unknown_owner.input.surface_cells[0].rain = CellRainAssignment::Suppressed {
        selected_contribution_id: "unknown".into(),
    };
    rejects(&unknown_owner)?;
    let mut inactive = primary()?;
    inactive.domain = domain(2, 1.0, vec![true, false])?;
    rejects(&inactive)
}

#[test]
fn partial_cell_ownership_never_scales_or_mixes_nonlinear_model_support() -> Result {
    for rain in [
        CellRainAssignment::Direct,
        CellRainAssignment::Suppressed {
            selected_contribution_id: "upstream-source".into(),
        },
    ] {
        let mut fixture = primary()?;
        fixture.domain = domain(1, 1.0, vec![true])?;
        fixture.definitions[0] = definition("upstream-model", 0.5)?;
        fixture.input.atoms = vec![
            SupportAtom {
                id: "upstream".into(),
                area_m2: 0.5,
            },
            SupportAtom {
                id: "local".into(),
                area_m2: 0.5,
            },
        ];
        fixture.input.surface_cells = vec![SurfaceCellSupport {
            cell: 0,
            atoms: vec!["upstream".into(), "local".into()],
            rain,
        }];
        rejects(&fixture)?;
    }
    Ok(())
}

#[test]
fn actual_target_areas_and_positive_additions_remain_representable() -> Result {
    let mut wrong_atom = primary()?;
    wrong_atom.input.atoms[0].area_m2 = 1.01;
    rejects(&wrong_atom)?;
    let mut wrong_model = primary()?;
    wrong_model.definitions[0] = definition("upstream-model", 0.99)?;
    rejects(&wrong_model)?;
    for area in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut fixture = primary()?;
        fixture.input.atoms[0].area_m2 = area;
        rejects(&fixture)?;
    }
    let mut absorbed = primary()?;
    absorbed.domain = domain(1, 1e8, vec![true])?;
    absorbed.definitions.clear();
    absorbed.input.contributions.clear();
    absorbed.input.atoms = vec![
        SupportAtom {
            id: "large".into(),
            area_m2: 1e16,
        },
        SupportAtom {
            id: "small".into(),
            area_m2: 1.0,
        },
    ];
    absorbed.input.surface_cells = vec![SurfaceCellSupport {
        cell: 0,
        atoms: vec!["large".into(), "small".into()],
        rain: CellRainAssignment::Direct,
    }];
    let error = match absorbed.plan() {
        Ok(_) => return Err("absorbed positive area requires rejection".into()),
        Err(error) => error,
    };
    let support = error
        .downcast_ref::<hydr::runoff::SupportError>()
        .ok_or("missing typed support error")?;
    assert_eq!(support.kind(), Some(SupportErrorKind::Area));
    Ok(())
}

#[test]
fn partition_completeness_and_provenance_require_explicit_assertions() -> Result {
    let mut incomplete = primary()?;
    incomplete.input.partition.complete_disjoint = false;
    rejects(&incomplete)?;
    let mut unnamed = primary()?;
    unnamed.input.partition.identity.clear();
    rejects(&unnamed)?;
    let mut no_provenance = primary()?;
    no_provenance.input.partition.provenance.clear();
    rejects(&no_provenance)
}
