//! Exact Ritter depth integrals for finite-volume front diagnostics.

use super::case::{END, LENGTH, Result, centre};

const GRAVITY: f64 = 9.81;

fn depth_integral(left: f64, right: f64) -> f64 {
    let c = GRAVITY.sqrt();
    let head = -c * END;
    let front = 2.0 * c * END;
    let constant = (right.min(head) - left).max(0.0);
    let fan_left = left.max(head);
    let fan_right = right.min(front);
    let fan = if fan_right > fan_left {
        ((front - fan_left).powi(3) - (front - fan_right).powi(3)) / (27.0 * GRAVITY * END.powi(2))
    } else {
        0.0
    };
    constant + fan
}

pub(super) fn mean_front(cells: usize, threshold: f64) -> Result<f64> {
    if cells == 0 || !threshold.is_finite() || threshold <= 0.0 || threshold >= 1.0 {
        return Err(format!(
            "Ritter front cells={cells} threshold={threshold}: require positive cells and 0<threshold<1"
        ).into());
    }
    let dx = LENGTH / cells as f64;
    // Threshold exact cell means, matching the finite-volume fields being compared.
    (0..cells)
        .rev()
        .find(|&index| {
            let left = -0.5 * LENGTH + index as f64 * dx;
            depth_integral(left, left + dx) / dx > threshold
        })
        .map(|index| centre(index, cells))
        .ok_or_else(|| {
            format!("Ritter front threshold={threshold}: require a resolved wet cell").into()
        })
}

#[cfg(test)]
mod tests {
    use super::mean_front;
    use crate::comparison::case::Case;

    #[test]
    fn exact_cell_mean_threshold_front_uses_matched_centres()
    -> Result<(), Box<dyn std::error::Error>> {
        let threshold = Case::parse("dry-ritter")?.front_threshold();
        for (cells, expected) in [(100, 23.0), (200, 23.5), (400, 23.75)] {
            assert_eq!(mean_front(cells, threshold)?, expected);
        }
        Ok(())
    }
}
