use super::fixture_model::Execution;

/// Frozen physical edge bounds of one row, in their unlabelled authority order.
/// Matching uses these bounds without inventing branch labels.
pub(in crate::channel::flow::surveyed_points) fn expected(
    row: &Execution,
) -> Vec<([f64; 2], &'static str)> {
    row.edges
        .iter()
        .enumerate()
        .map(|(i, edge)| {
            assert_eq!(edge.label, format!("edge_{i}"));
            (edge.bounds.map(f64::from_bits), edge.label)
        })
        .collect()
}
