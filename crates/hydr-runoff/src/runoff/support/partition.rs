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
        assign_rain(
            &label,
            &support.atoms,
            &support.rain,
            (selected, sources),
            &mut owners,
        )?;
    }
    if cells.len() != domain.active().iter().filter(|v| **v).count() {
        return Err(err(
            SupportErrorKind::Coverage,
            "supply complete whole footprints for every active cell".into(),
        ));
    }
    let cell_atoms: BTreeSet<&String> = footprint.keys().copied().collect();
    check_pool_rain(
        input,
        (&atoms, &cell_atoms),
        (selected, sources),
        &mut owners,
    )?;
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

/// Selection flags and contribution indices by identity.
type Selection<'a> = (&'a [bool], &'a BTreeMap<String, usize>);

/// Known atoms with their areas, and the atoms of every cell footprint.
type Footprints<'a> = (&'a BTreeMap<String, f64>, &'a BTreeSet<&'a String>);

/// Records one whole footprint's rain owner.
///
/// Direct rain makes the footprint its own primary owner. Suppressed rain requires every atom
/// to belong to the named selected source already.
fn assign_rain(
    label: &str,
    ids: &[String],
    rain: &CellRainAssignment,
    (selected, sources): Selection<'_>,
    owners: &mut BTreeMap<String, String>,
) -> Result<(), SupportError> {
    match rain {
        CellRainAssignment::Direct => {
            for atom in ids {
                if let Some(previous) = owners.insert(atom.clone(), label.to_owned()) {
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
            if !selected.get(index).copied().unwrap_or(false)
                || ids
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
    Ok(())
}

/// Checks pool rain support: one entry per pool, known disjoint atoms outside every cell
/// footprint, and one rain owner per atom.
///
/// A pool's direct rain and a selected upstream source never share an atom.
fn check_pool_rain(
    input: &SupportPlanInput,
    (atoms, cell_atoms): Footprints<'_>,
    selection: Selection<'_>,
    owners: &mut BTreeMap<String, String>,
) -> Result<(), SupportError> {
    let mut pools = BTreeSet::new();
    let mut pool_atoms = BTreeMap::new();
    for support in &input.pool_rain {
        super::definition::text(&support.pool_id, "pool_rain.pool_id")?;
        if !pools.insert(support.pool_id.as_str()) {
            return Err(err(
                SupportErrorKind::Coverage,
                format!(
                    "pool={:?}: declare rain support at most once per pool",
                    support.pool_id
                ),
            ));
        }
        let label = format!("pool={:?}", support.pool_id);
        check_atoms(&support.atoms, atoms, &label, true)?;
        for atom in &support.atoms {
            if cell_atoms.contains(atom) {
                return Err(err(
                    SupportErrorKind::Ownership,
                    format!(
                        "atom={atom:?},{label}: pool rain support cannot overlap a cell footprint"
                    ),
                ));
            }
            if let Some(previous) = pool_atoms.insert(atom, support.pool_id.as_str()) {
                return Err(err(
                    SupportErrorKind::Ownership,
                    format!(
                        "atom={atom:?},pools={previous:?}/{:?}: pool rain supports cannot overlap",
                        support.pool_id
                    ),
                ));
            }
        }
        assign_rain(&label, &support.atoms, &support.rain, selection, owners)?;
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

#[cfg(test)]
mod tests {
    use super::super::SupportAtom;
    use super::super::fixture::{Result, definition, fixture, kind, pool, suppressed_by};
    use super::*;

    #[test]
    fn direct_pool_rain_owns_its_atoms() -> Result {
        let plan = fixture()?.plan()?;
        assert!(!plan.pool_rain_suppressed("basin"));
        assert!(!plan.pool_rain_suppressed("elsewhere"));
        Ok(())
    }

    #[test]
    fn suppressed_pool_rain_requires_the_selected_owner_of_every_atom() -> Result {
        let mut owned = fixture()?;
        owned.definitions = vec![definition("upstream-model", 1.5)?];
        owned.input.contributions[0].atoms.push("pool".into());
        owned.input.pool_rain = vec![pool("basin", &["pool"], suppressed_by("upstream-source"))];
        assert!(owned.plan()?.pool_rain_suppressed("basin"));

        let mut unowned = fixture()?;
        unowned.input.pool_rain = vec![pool("basin", &["pool"], suppressed_by("upstream-source"))];
        assert_eq!(kind(unowned.plan()), Some(SupportErrorKind::Ownership));

        let mut unknown = fixture()?;
        unknown.input.pool_rain = vec![pool("basin", &["pool"], suppressed_by("nobody"))];
        assert_eq!(kind(unknown.plan()), Some(SupportErrorKind::Ownership));
        Ok(())
    }

    #[test]
    fn direct_pool_rain_rejects_an_atom_of_a_selected_source() -> Result {
        let mut overlap = fixture()?;
        overlap.definitions = vec![definition("upstream-model", 1.5)?];
        overlap.input.contributions[0].atoms.push("pool".into());
        assert_eq!(kind(overlap.plan()), Some(SupportErrorKind::Ownership));
        Ok(())
    }

    #[test]
    fn pool_rain_rejects_cell_overlap_pool_overlap_and_repeats() -> Result {
        let mut cell = fixture()?;
        cell.input.pool_rain = vec![pool("basin", &["local"], CellRainAssignment::Direct)];
        assert_eq!(kind(cell.plan()), Some(SupportErrorKind::Ownership));

        let mut shared = fixture()?;
        shared.input.pool_rain = vec![
            pool("basin", &["pool"], CellRainAssignment::Direct),
            pool("pond", &["pool"], CellRainAssignment::Direct),
        ];
        assert_eq!(kind(shared.plan()), Some(SupportErrorKind::Ownership));

        let mut repeated = fixture()?;
        repeated.input.atoms.push(SupportAtom {
            id: "second".into(),
            area_m2: 0.5,
        });
        repeated.input.pool_rain = vec![
            pool("basin", &["pool"], CellRainAssignment::Direct),
            pool("basin", &["second"], CellRainAssignment::Direct),
        ];
        assert_eq!(kind(repeated.plan()), Some(SupportErrorKind::Coverage));
        Ok(())
    }

    #[test]
    fn pool_rain_rejects_empty_unknown_and_uncovered_atoms() -> Result {
        let mut empty = fixture()?;
        empty.input.pool_rain = vec![pool("basin", &[], CellRainAssignment::Direct)];
        assert_eq!(kind(empty.plan()), Some(SupportErrorKind::Partition));

        let mut unknown = fixture()?;
        unknown.input.pool_rain = vec![pool("basin", &["missing"], CellRainAssignment::Direct)];
        assert_eq!(kind(unknown.plan()), Some(SupportErrorKind::Partition));

        let mut uncovered = fixture()?;
        uncovered.input.pool_rain.clear();
        assert_eq!(kind(uncovered.plan()), Some(SupportErrorKind::Ownership));
        Ok(())
    }
}
