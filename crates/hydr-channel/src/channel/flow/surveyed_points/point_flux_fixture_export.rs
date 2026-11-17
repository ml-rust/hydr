use super::{fixtures, flux_fixture_model::Bounds, flux_fixtures};
use std::collections::BTreeMap;

fn pair(value: Option<Bounds>) -> String {
    value.map_or_else(
        || "-".into(),
        |[lower, upper]| format!("{lower:016x},{upper:016x}"),
    )
}

pub(super) fn emit() -> Result<(), Box<dyn std::error::Error>> {
    let executions = fixtures::executions();
    let mut by_id = BTreeMap::new();
    for execution in &executions {
        if by_id.insert(execution.id, execution).is_some() {
            return Err(format!(
                "Repeated execution identity {}. Check candidate fixtures",
                execution.id
            )
            .into());
        }
    }
    let mut emitted = BTreeMap::new();
    for reference in flux_fixtures::REFERENCES {
        if emitted.insert(reference.id, ()).is_some() {
            return Err(format!(
                "Repeated flux identity {}. Check flux fixtures",
                reference.id
            )
            .into());
        }
        let execution = by_id.get(reference.id).ok_or_else(|| {
            format!(
                "Missing execution identity {}. Check candidate fixtures",
                reference.id
            )
        })?;
        let config = fixtures::CONFIGS.get(execution.config).ok_or_else(|| {
            format!(
                "Missing config {} for {}. Check candidate fixtures",
                execution.config, reference.id
            )
        })?;
        eprintln!(
            "POINT_FLUX_REFERENCE\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            reference.id,
            reference.source_group,
            reference.physical_group,
            reference.outcome,
            reference.config_identity,
            reference.expected_identity,
            reference.truth_identity,
            reference.flux_gate_identity,
            reference.water_identity,
            reference.momentum_identity,
            pair(reference.water),
            pair(reference.momentum),
            reference.positive_water_required,
        );
        eprintln!(
            "POINT_FLUX_JOIN\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            execution.id,
            execution.config,
            execution.source_group,
            execution.group,
            execution.declared,
            execution.outcome,
            execution.physical,
            execution.config_identity,
            execution.expected_identity,
        );
        eprintln!(
            "POINT_FLUX_CONFIG\t{}\t{:?}\t{:?}",
            execution.id, config.floats, config.counts
        );
        for (index, declaration) in reference.quantization.iter().enumerate() {
            eprintln!(
                "POINT_FLUX_QUANTIZATION\t{}\t{index}\t{}\t{}\t{}\t{}\t{}\t{}",
                reference.id,
                declaration.quantity,
                declaration.exact_rational,
                pair(Some(declaration.projected)),
                declaration.maximum_projected_width_rational,
                declaration.positive_water_not_dry,
                declaration.declaration_identity,
            );
        }
    }
    if emitted.len() != by_id.len() {
        return Err(
            "Flux identities omit candidate executions. Check both fixture identity sets".into(),
        );
    }
    Ok(())
}
