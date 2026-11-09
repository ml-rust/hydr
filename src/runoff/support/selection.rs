//! Whole measured replacements and declaration-only leaf ownership.

use super::{ContributionKind, SupportError, SupportErrorKind, SupportPlanInput};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn select(
    input: &SupportPlanInput,
    indices: &BTreeMap<String, usize>,
) -> Result<Vec<bool>, SupportError> {
    let mut selected = vec![true; input.contributions.len()];
    let mut roots = BTreeSet::new();
    let mut replaced = BTreeSet::new();
    for replacement in &input.replacements {
        let root = *indices.get(&replacement.selected).ok_or_else(|| {
            error(
                SupportErrorKind::Replacement,
                format!("unknown selected root={:?}", replacement.selected),
            )
        })?;
        if !matches!(
            input.contributions[root].kind,
            ContributionKind::MeasuredFlow { .. }
        ) || !roots.insert(root)
            || replacement.replaced.is_empty()
        {
            return Err(error(
                SupportErrorKind::Replacement,
                format!(
                    "selected={:?}: require one measured replacement with nonempty complete replaced roots",
                    replacement.selected
                ),
            ));
        }
        let mut union = BTreeSet::new();
        for id in &replacement.replaced {
            let index = *indices.get(id).ok_or_else(|| {
                error(
                    SupportErrorKind::Replacement,
                    format!("unknown replaced source={id:?}"),
                )
            })?;
            let source = &input.contributions[index];
            if !matches!(
                source.kind,
                ContributionKind::ModelledRunoff { .. } | ContributionKind::ImportedRunoff { .. }
            ) || !replaced.insert(index)
            {
                return Err(error(
                    SupportErrorKind::Replacement,
                    format!("replaced={id:?}: require a unique modelled/imported source"),
                ));
            }
            if let ContributionKind::ImportedRunoff { boundary_id } = &source.kind
                && !matches!(&input.contributions[root].kind, ContributionKind::MeasuredFlow { boundary_id: target } if target == boundary_id)
            {
                return Err(error(
                    SupportErrorKind::Destination,
                    format!(
                        "replaced={id:?},selected={:?}: preserve imported boundary identity={boundary_id:?}",
                        replacement.selected
                    ),
                ));
            }
            for atom in &source.atoms {
                if !union.insert(atom) {
                    return Err(error(
                        SupportErrorKind::Replacement,
                        format!(
                            "selected={:?},atom={atom:?}: replaced footprints overlap",
                            replacement.selected
                        ),
                    ));
                }
            }
            selected[index] = false;
        }
        let target: BTreeSet<_> = input.contributions[root].atoms.iter().collect();
        if union != target {
            return Err(error(
                SupportErrorKind::Replacement,
                format!(
                    "selected={:?}: require exact union of all replaced support atoms",
                    replacement.selected
                ),
            ));
        }
    }
    let mut leaves = BTreeMap::<usize, usize>::new();
    for (index, source) in input.contributions.iter().enumerate() {
        let leaf = matches!(
            source.kind,
            ContributionKind::Baseflow | ContributionKind::Release
        );
        if leaf && (!source.atoms.is_empty() || !source.includes.is_empty()) {
            return Err(error(
                SupportErrorKind::Inclusion,
                format!("leaf={:?}: require empty atoms and includes", source.id),
            ));
        }
        if matches!(source.kind, ContributionKind::ModelledRunoff { .. })
            && !source.includes.is_empty()
        {
            return Err(error(
                SupportErrorKind::Inclusion,
                format!(
                    "modelled={:?}: rainfall runoff cannot include additive leaves",
                    source.id
                ),
            ));
        }
        let mut local = BTreeSet::new();
        for id in &source.includes {
            let child = *indices.get(id).ok_or_else(|| {
                error(
                    SupportErrorKind::Inclusion,
                    format!("source={:?},unknown leaf={id:?}", source.id),
                )
            })?;
            if !matches!(
                input.contributions[child].kind,
                ContributionKind::Baseflow | ContributionKind::Release
            ) || !local.insert(child)
            {
                return Err(error(
                    SupportErrorKind::Inclusion,
                    format!(
                        "source={:?},included={id:?}: require unique Baseflow/Release leaves",
                        source.id
                    ),
                ));
            }
            if selected[index]
                && (!matches!(
                    source.kind,
                    ContributionKind::ImportedRunoff { .. } | ContributionKind::MeasuredFlow { .. }
                ) || leaves.insert(child, index).is_some())
            {
                return Err(error(
                    SupportErrorKind::Inclusion,
                    format!("leaf={id:?}: require exactly one selected imported/measured parent"),
                ));
            }
        }
    }
    for (index, source) in input.contributions.iter().enumerate() {
        if matches!(
            source.kind,
            ContributionKind::Baseflow | ContributionKind::Release
        ) {
            selected[index] = false;
            if !leaves.contains_key(&index) {
                return Err(error(
                    SupportErrorKind::Inclusion,
                    format!(
                        "leaf={:?}: bind to one selected imported/measured source, including after replacement",
                        source.id
                    ),
                ));
            }
        }
    }
    Ok(selected)
}

fn error(kind: SupportErrorKind, context: String) -> SupportError {
    SupportError::new(kind, context)
}
