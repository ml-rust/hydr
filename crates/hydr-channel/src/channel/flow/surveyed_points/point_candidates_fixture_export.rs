use super::{
    Result,
    fixture_model::{Execution, State},
    fixtures, frozen,
};
use std::collections::BTreeSet;

fn optional(value: Option<[u64; 2]>) -> String {
    value.map_or_else(|| "-".into(), |v| format!("{v:?}"))
}
fn state(id: &str, role: &str, index: usize, value: &State) {
    eprintln!(
        "POINT_STATE\t{id}\t{role}\t{index}\t{}\t{:?}\t{}\t{}\t{:?}\t{}",
        value.label,
        value.area,
        optional(value.depth),
        optional(value.velocity),
        value.discharge,
        value.velocity_provenance
    );
}
pub(super) fn emit(rows: &[Execution]) -> Result {
    let mut ids = BTreeSet::new();
    for (owner, (id, cap, coordinates)) in frozen::profiles().enumerate() {
        eprintln!("POINT_PROFILE\t{owner}\t{id}\t{cap}\t{coordinates:?}");
    }
    for (i, cfg) in fixtures::CONFIGS.iter().enumerate() {
        eprintln!("POINT_CONFIG\t{i}\t{:?}\t{:?}", cfg.floats, cfg.counts);
    }
    for row in rows {
        assert!(ids.insert(row.id), "Repeated execution identity {}", row.id);
        let source = frozen::cases()
            .iter()
            .find(|c| c.id == row.id)
            .ok_or("Missing captured source fixture identity")?;
        eprintln!(
            "POINT_ROW\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            row.id,
            row.config,
            row.source_group,
            row.group,
            row.declared,
            row.outcome,
            row.physical,
            row.config_identity,
            row.expected_identity
        );
        eprintln!(
            "POINT_INPUT\t{}\t{}\t{:?}",
            row.id, source.owner, source.inputs
        );
        for (i, trace) in row.traces.iter().enumerate() {
            state(row.id, "trace", i, trace);
        }
        if let Some(star) = row.star {
            state(row.id, "star", 0, &star);
        }
        for (i, edge) in row.edges.iter().enumerate() {
            eprintln!(
                "POINT_EDGE\t{}\t{i}\t{}\t{:?}",
                row.id, edge.label, edge.bounds
            );
        }
    }
    assert_eq!(ids.len(), 70);
    Ok(())
}
