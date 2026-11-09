use super::case::{Case, LENGTH, Result, centre};
use hydr::model::CellState;

pub(super) struct Metrics {
    pub depth_l1: f64,
    pub qx_l1: f64,
}

pub(super) fn compare(
    case: Case,
    hydr: &[CellState],
    reference: &[CellState],
    residual: f64,
) -> Result<Metrics> {
    let count = hydr.len();
    if count == 0 || reference.len() != count {
        return Err("field shapes differ or are empty: supply matched nonempty fields".into());
    }
    let dx = LENGTH / count as f64;
    let mut h_l1 = 0.0;
    let mut qx_l1 = 0.0;
    let mut min_h = f64::INFINITY;
    let mut max_h = 0.0_f64;
    for (cell, expected) in hydr.iter().zip(reference) {
        h_l1 += (cell.h - expected.h).abs();
        qx_l1 += (cell.qx - expected.qx).abs();
        min_h = min_h.min(cell.h).min(expected.h);
        max_h = max_h.max(cell.h).max(expected.h);
    }
    h_l1 /= count as f64;
    qx_l1 /= count as f64;
    let threshold = case.front_threshold();
    let front = |field: &[CellState]| -> Result<f64> {
        field
            .iter()
            .rposition(|cell| cell.h > threshold)
            .map(|index| centre(index, count))
            .ok_or_else(|| {
                format!("front threshold={threshold}: require a resolved upstream column").into()
            })
    };
    let hydr_front = front(hydr)?;
    let reference_front = front(reference)?;
    let front_distance = (hydr_front - reference_front).abs();
    let reference_storage = reference.iter().map(|cell| cell.h * dx).sum::<f64>();
    let reference_residual = reference_storage - case.initial_storage();
    println!(
        "case={} cells={count} depth_l1={h_l1:.17e} qx_l1={qx_l1:.17e} hydr_front={hydr_front:.17e} reference_front={reference_front:.17e} front_distance={front_distance:.17e} hydr_volume_residual={residual:.17e} reference_volume_residual={reference_residual:.17e} min_h={min_h:.17e} max_h={max_h:.17e}",
        case.name
    );
    if h_l1 > 0.03 || qx_l1 > 0.1 || front_distance > 4.0 * dx {
        return Err(format!("comparison case={} cells={count}: require depth_l1<=0.03, qx_l1<=0.1, front_distance<={}", case.name, 4.0*dx).into());
    }
    if reference_residual.abs() > 1e-8 + 1e-12 * case.initial_storage() {
        return Err(format!(
            "reference storage residual={reference_residual}: require conserved unit-width water"
        )
        .into());
    }
    Ok(Metrics {
        depth_l1: h_l1,
        qx_l1,
    })
}
