use super::check::rejected;
use crate::solver::{Axis, EngineError, SnapshotData, StepDiagnostics};

pub(super) fn check(data: &SnapshotData, step: &StepDiagnostics) -> Result<(), EngineError> {
    let mut speeds = [0.0_f64; 2];
    for face in &step.faces {
        if !face.signal_speed.is_finite() || face.signal_speed < 0.0 {
            return Err(rejected(
                "face signal speed",
                format!(
                    "face={:?}, speed={}: require a finite nonnegative numerical bound",
                    face.face, face.signal_speed
                ),
            ));
        }
        let axis = if face.face.axis == Axis::X { 0 } else { 1 };
        speeds[axis] = speeds[axis].max(face.signal_speed);
    }
    let (dx, dy) = data.setup.domain.grid().spacing();
    let frequency = speeds[0] / dx + speeds[1] / dy;
    if !frequency.is_finite() {
        return Err(rejected(
            "face CFL",
            format!("frequency={frequency}: require representable directional signal speeds"),
        ));
    }
    let limit = if frequency > 0.0 {
        data.setup.config.cfl / frequency
    } else {
        f64::INFINITY
    };
    if limit.is_nan()
        || limit <= 0.0
        || !same_bound(step.cfl_limit, limit)
        || !bounded(step.dt, limit)
    {
        return Err(rejected(
            "face CFL",
            format!(
                "stored_limit={}, dt={}, face_limit={limit}: require the native combined face-speed bound",
                step.cfl_limit, step.dt
            ),
        ));
    }
    Ok(())
}

/// Native face sums allow 64 epsilon relative rounding, without an absolute unit floor.
fn same_bound(left: f64, right: f64) -> bool {
    left == right
        || (left.is_finite()
            && right.is_finite()
            && (left - right).abs() <= 64.0 * f64::EPSILON * left.abs().max(right.abs()))
}

pub(super) fn bounded(value: f64, upper: f64) -> bool {
    value <= upper
        || (value.is_finite()
            && upper.is_finite()
            && value - upper <= 64.0 * f64::EPSILON * value.abs().max(upper.abs()))
}
