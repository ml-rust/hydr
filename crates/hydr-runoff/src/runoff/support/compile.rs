//! Static compilation into resolved immutable real-consumer targets.

use super::definition::{CompiledPlan, text};
use super::{ContributionKind, SupportError, SupportErrorKind, SupportPlan, SupportPlanInput};
use crate::contract::BoundaryConditions;
use crate::model::Domain;
use crate::runoff::SubcatchmentDefinition;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

impl SupportPlan {
    /// Check declared partition ownership against actual numerical targets.
    ///
    /// ARUS supplies geometric evidence. This constructor checks metadata and complete source assignments.
    pub fn new(
        input: SupportPlanInput,
        domain: &Domain,
        boundaries: &BoundaryConditions,
        definitions: &[SubcatchmentDefinition],
    ) -> Result<Self, SupportError> {
        text(&input.partition.identity, "partition.identity")?;
        text(
            &input.partition.covered_support,
            "partition.covered_support",
        )?;
        text(&input.partition.provenance, "partition.provenance")?;
        text(&input.surface_input_identity, "surface_input_identity")?;
        if !input.partition.complete_disjoint {
            return Err(err(
                SupportErrorKind::Partition,
                "require explicit complete_disjoint assertion with application GIS evidence",
            ));
        }
        let checked_boundaries =
            BoundaryConditions::with_open_faces(domain, boundaries.open_faces().to_vec())?;
        let mut definition_indices = BTreeMap::new();
        for (index, definition) in definitions.iter().enumerate() {
            text(&definition.id, "definition.id")?;
            if definition_indices
                .insert(definition.id.clone(), index)
                .is_some()
            {
                return Err(err(
                    SupportErrorKind::Identity,
                    format!(
                        "duplicate definition={:?}: supply unique numerical definitions",
                        definition.id
                    ),
                ));
            }
        }
        let mut sources = BTreeMap::new();
        let mut model_indices = BTreeMap::new();
        let mut outlets = BTreeSet::new();
        for (index, source) in input.contributions.iter().enumerate() {
            for (value, field) in [
                (&source.id, "contribution.id"),
                (&source.input_identity, "contribution.input_identity"),
                (&source.provenance, "contribution.provenance"),
            ] {
                text(value, field)?;
            }
            if sources.insert(source.id.clone(), index).is_some() {
                return Err(err(
                    SupportErrorKind::Identity,
                    format!("duplicate contribution={:?}", source.id),
                ));
            }
            match &source.kind {
                ContributionKind::ModelledRunoff {
                    definition_id,
                    outlet_id,
                } => {
                    text(definition_id, "modelled.definition_id")?;
                    text(outlet_id, "modelled.outlet_id")?;
                    let definition = *definition_indices.get(definition_id).ok_or_else(|| {
                        err(
                            SupportErrorKind::Identity,
                            format!(
                                "source={:?}: unknown definition={definition_id:?}",
                                source.id
                            ),
                        )
                    })?;
                    if model_indices.insert(definition, index).is_some()
                        || !outlets.insert(outlet_id)
                    {
                        return Err(err(
                            SupportErrorKind::Ownership,
                            format!(
                                "source={:?},definition={definition_id:?},outlet={outlet_id:?}: require one contribution per definition and unique outlets",
                                source.id
                            ),
                        ));
                    }
                }
                ContributionKind::ImportedRunoff { boundary_id }
                | ContributionKind::MeasuredFlow { boundary_id } => {
                    text(boundary_id, "contribution.boundary_id")?
                }
                ContributionKind::Baseflow | ContributionKind::Release => {}
            }
        }
        if model_indices.len() != definitions.len() {
            return Err(err(
                SupportErrorKind::Coverage,
                "require exactly one modelled contribution per supplied definition, including replaced definitions",
            ));
        }
        let selected = super::selection::select(&input, &sources)?;
        super::partition::check(&input, domain, definitions, &selected, &sources)?;
        let mut boundary_indices = BTreeMap::new();
        let mut faces = BTreeSet::new();
        let mut bound_roots = BTreeSet::new();
        let declared: BTreeSet<_> = checked_boundaries.open_faces().iter().copied().collect();
        for (index, binding) in input.boundaries.iter().enumerate() {
            for (value, field) in [
                (&binding.id, "boundary.id"),
                (&binding.input_identity, "boundary.input_identity"),
                (&binding.provenance, "boundary.provenance"),
            ] {
                text(value, field)?;
            }
            if boundary_indices.insert(binding.id.clone(), index).is_some()
                || !declared.contains(&binding.face)
                || !faces.insert(binding.face)
            {
                return Err(err(
                    SupportErrorKind::Destination,
                    format!(
                        "boundary={:?},face={:?}: require unique binding per declared open face",
                        binding.id, binding.face
                    ),
                ));
            }
            if let Some(id) = &binding.contribution {
                let source_index = *sources.get(id).ok_or_else(|| {
                    err(
                        SupportErrorKind::Identity,
                        format!("boundary={:?}: unknown source={id:?}", binding.id),
                    )
                })?;
                let source = &input.contributions[source_index];
                let matching = matches!(&source.kind,ContributionKind::ImportedRunoff{boundary_id}|ContributionKind::MeasuredFlow{boundary_id} if boundary_id==&binding.id);
                if !selected[source_index]
                    || !matching
                    || source.input_identity != binding.input_identity
                    || !bound_roots.insert(source_index)
                {
                    return Err(err(
                        SupportErrorKind::Composition,
                        format!(
                            "boundary={:?},source={id:?}: bind one selected imported/measured root with matching boundary and input identity",
                            binding.id
                        ),
                    ));
                }
            }
        }
        if faces != declared {
            return Err(err(
                SupportErrorKind::Coverage,
                "supply exactly one boundary identity/provenance binding for every open face",
            ));
        }
        for (index, source) in input.contributions.iter().enumerate() {
            if let ContributionKind::ImportedRunoff { boundary_id }
            | ContributionKind::MeasuredFlow { boundary_id } = &source.kind
                && (!boundary_indices.contains_key(boundary_id)
                    || (selected[index] && !bound_roots.contains(&index)))
            {
                return Err(err(
                    SupportErrorKind::Destination,
                    format!(
                        "source={:?},boundary={boundary_id:?}: selected roots require exactly one matching bound face",
                        source.id
                    ),
                ));
            }
        }
        let mut active_indices = Vec::new();
        let mut active_contributions = Vec::new();
        let mut active_definitions = Vec::new();
        for (index, definition) in definitions.iter().enumerate() {
            let source = *model_indices.get(&index).ok_or_else(|| {
                err(
                    SupportErrorKind::Coverage,
                    "missing resolved model definition",
                )
            })?;
            if selected[source] {
                active_indices.push(index);
                active_contributions.push(source);
                active_definitions.push(definition.clone());
            }
        }
        Ok(Self {
            inner: Arc::new(CompiledPlan {
                input,
                domain: domain.clone(),
                boundaries: checked_boundaries,
                definitions: active_definitions,
                active_indices,
                active_contributions,
                boundary_indices,
            }),
        })
    }
}
fn err(kind: SupportErrorKind, context: impl Into<String>) -> SupportError {
    SupportError::new(kind, context)
}
