//! Complete declared ownership and represented static area checks.

use super::{
    CellRainAssignment, ContributionKind, SupportError, SupportErrorKind, SupportPlanInput,
};
use crate::model::Domain;
use crate::numerics::outward::{rounding_bound, upward_add};
use crate::runoff::SubcatchmentDefinition;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn check(
    input: &SupportPlanInput,
    domain: &Domain,
    definitions: &[SubcatchmentDefinition],
    selected: &[bool],
    sources: &BTreeMap<String, usize>,
) -> Result<(), SupportError> {
    let mut atoms = BTreeMap::new();
    for atom in &input.atoms {
        super::definition::text(&atom.id, "atom.id")?;
        if !atom.area_m2.is_finite()
            || atom.area_m2 <= 0.0
            || atoms.insert(atom.id.clone(), atom.area_m2).is_some()
        {
            return Err(err(
                SupportErrorKind::Partition,
                format!(
                    "atom={:?},area={}: require unique identity and finite positive area",
                    atom.id, atom.area_m2
                ),
            ));
        }
    }
    if atoms.is_empty() {
        return Err(err(
            SupportErrorKind::Partition,
            "empty partition: supply complete support atoms".into(),
        ));
    }
    let definitions: BTreeMap<_, _> = definitions.iter().map(|d| (&d.id, d)).collect();
    let mut owners = BTreeMap::new();
    for (index, source) in input.contributions.iter().enumerate() {
        let leaf = matches!(
            source.kind,
            ContributionKind::Baseflow | ContributionKind::Release
        );
        check_atoms(&source.atoms, &atoms, &source.id, !leaf)?;
        if let ContributionKind::ModelledRunoff { definition_id, .. } = &source.kind {
            let definition = definitions.get(definition_id).ok_or_else(|| {
                err(
                    SupportErrorKind::Identity,
                    format!("unknown numerical definition={definition_id:?}"),
                )
            })?;
            area(
                &source.atoms,
                &atoms,
                definition.parameters.geometry().total_area_m2,
                &source.id,
            )?;
        }
        if selected[index] {
            for atom in &source.atoms {
                if let Some(previous) = owners.insert(atom.clone(), source.id.clone()) {
                    return Err(err(
                        SupportErrorKind::Ownership,
                        format!(
                            "atom={atom:?},owners={previous:?}/{:?}: assign exactly one selected primary source",
                            source.id
                        ),
                    ));
                }
            }
        }
    }
    let mut cells = BTreeSet::new();
    let mut footprint = BTreeMap::new();
    for support in &input.surface_cells {
        if support.cell >= domain.active().len()
            || !domain.active()[support.cell]
            || !cells.insert(support.cell)
        {
            return Err(err(
                SupportErrorKind::Coverage,
                format!(
                    "cell={}: require each active cell exactly once and no inactive footprints",
                    support.cell
                ),
            ));
        }
        let label = format!("cell={}", support.cell);
        check_atoms(&support.atoms, &atoms, &label, true)?;
        area(&support.atoms, &atoms, domain.grid().cell_area(), &label)?;
        for atom in &support.atoms {
            if let Some(previous) = footprint.insert(atom, support.cell) {
                return Err(err(
                    SupportErrorKind::Ownership,
                    format!(
                        "atom={atom:?},cells={previous}/{}: cell footprints cannot overlap, including suppressed cells",
                        support.cell
                    ),
                ));
            }
        }
        match &support.rain {
            CellRainAssignment::Direct => {
                for atom in &support.atoms {
                    if let Some(previous) = owners.insert(atom.clone(), label.clone()) {
                        return Err(err(
                            SupportErrorKind::Ownership,
                            format!(
                                "atom={atom:?},direct={label},existing={previous:?}: declare one primary owner"
                            ),
                        ));
                    }
                }
            }
            CellRainAssignment::Suppressed {
                selected_contribution_id,
            } => {
                let index = *sources.get(selected_contribution_id).ok_or_else(|| {
                    err(
                        SupportErrorKind::Ownership,
                        format!("{label}: unknown suppression owner={selected_contribution_id:?}"),
                    )
                })?;
                if !selected[index]
                    || support
                        .atoms
                        .iter()
                        .any(|atom| owners.get(atom) != Some(selected_contribution_id))
                {
                    return Err(err(
                        SupportErrorKind::Ownership,
                        format!(
                            "{label}: complete footprint must belong to selected owner={selected_contribution_id:?}"
                        ),
                    ));
                }
            }
        }
    }
    if cells.len() != domain.active().iter().filter(|v| **v).count() {
        return Err(err(
            SupportErrorKind::Coverage,
            "supply complete whole footprints for every active cell".into(),
        ));
    }
    if owners.len() != atoms.len() {
        let missing: Vec<_> = atoms
            .keys()
            .filter(|id| !owners.contains_key(*id))
            .collect();
        return Err(err(
            SupportErrorKind::Ownership,
            format!("uncovered atoms={missing:?}: assign exactly one selected primary owner"),
        ));
    }
    Ok(())
}

fn check_atoms(
    ids: &[String],
    atoms: &BTreeMap<String, f64>,
    label: &str,
    required: bool,
) -> Result<(), SupportError> {
    let mut seen = BTreeSet::new();
    if required && ids.is_empty() {
        return Err(err(
            SupportErrorKind::Partition,
            format!("{label}: primary support cannot be empty"),
        ));
    }
    for id in ids {
        if !atoms.contains_key(id) || !seen.insert(id) {
            return Err(err(
                SupportErrorKind::Partition,
                format!("{label},atom={id:?}: require known unique footprint atoms"),
            ));
        }
    }
    Ok(())
}

fn area(
    ids: &[String],
    atoms: &BTreeMap<String, f64>,
    target: f64,
    label: &str,
) -> Result<(), SupportError> {
    let mut sum = 0.0;
    let mut bound = 0.0;
    let arithmetic = |error: crate::numerics::outward::OutwardError| {
        err(
            SupportErrorKind::Area,
            format!("{label}: static area arithmetic: {error}"),
        )
    };
    for id in ids {
        let value = *atoms.get(id).ok_or_else(|| {
            err(
                SupportErrorKind::Partition,
                format!("{label}: unknown area atom={id:?}"),
            )
        })?;
        let next = sum + value;
        if !next.is_finite() || next <= sum {
            return Err(err(
                SupportErrorKind::Area,
                format!(
                    "{label},atom={id:?},area={value},sum={sum}: positive area is absorbed or overflows"
                ),
            ));
        }
        bound = upward_add(bound, rounding_bound(next).map_err(arithmetic)?).map_err(arithmetic)?;
        sum = next;
    }
    let allowance =
        upward_add(bound, rounding_bound(target).map_err(arithmetic)?).map_err(arithmetic)?;
    if !target.is_finite() || target <= 0.0 || (sum - target).abs() > allowance {
        return Err(err(
            SupportErrorKind::Area,
            format!(
                "{label},atom_area={sum},target_area={target},bound={allowance}: supply complete actual target support"
            ),
        ));
    }
    Ok(())
}
fn err(kind: SupportErrorKind, context: String) -> SupportError {
    SupportError::new(kind, context)
}
