use super::check::rejected;
use crate::contract::{Axis, EngineError, SolverConfig};
use crate::solver::{FaceTransfer, TransmissiveDiagnostics};

pub(super) fn flux(
    state: &TransmissiveDiagnostics,
    face: &FaceTransfer,
    width: f64,
    config: &SolverConfig,
) -> Result<[f64; 3], EngineError> {
    let h = state.depth;
    let q = state.outward_discharge_per_width;
    let tangent = state.tangential_discharge_per_width;
    if ![h, q, tangent].iter().all(|value| value.is_finite()) || h < 0.0 {
        return Err(rejected(
            "transmissive state",
            format!("state={state:?}: require finite nonnegative depth and discharge"),
        ));
    }
    if h == 0.0 {
        if q != 0.0 || tangent != 0.0 || face.signal_speed != 0.0 {
            return Err(rejected(
                "transmissive dry state",
                format!(
                    "state={state:?}, speed={}: dry copying requires zero discharges and zero signal speed",
                    face.signal_speed
                ),
            ));
        }
        return Ok([0.0; 3]);
    }
    let u = q / h;
    let v = tangent / h;
    let c = config.gravity.sqrt() * h.sqrt();
    let margin = config.boundary.critical_relative * u.abs().max(c);
    if ![u, v, c].iter().all(|value| value.is_finite())
        || u - c <= margin
        || face.signal_speed < u + c
    {
        return Err(rejected(
            "transmissive regime",
            format!(
                "state={state:?}: require resolved supercritical outflow and a conservative signal bound"
            ),
        ));
    }
    let sign = if face.left.is_some() { 1.0 } else { -1.0 };
    let normal = q * u + 0.5 * config.gravity * h * h;
    let transverse = sign * q * v;
    let result = if face.face.axis == Axis::X {
        [sign * q, normal, transverse]
    } else {
        [sign * q, transverse, normal]
    };
    if !result.iter().all(|value| value.is_finite()) || !(q * width).is_finite() {
        return Err(rejected(
            "transmissive flux",
            format!("state={state:?}, width={width}: require representable physical flux"),
        ));
    }
    Ok(result)
}
