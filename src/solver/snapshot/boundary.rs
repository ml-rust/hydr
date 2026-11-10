use super::SnapshotData;
use super::cfl::bounded;
use super::check::rejected;
use crate::solver::{Axis, BoundaryForcing, EngineError, StepDiagnostics};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn check(data: &SnapshotData, step: &StepDiagnostics) -> Result<(), EngineError> {
    let interval = data
        .setup
        .forcing
        .at(step.start)
        .ok_or_else(|| rejected("boundary interval", "missing forcing coverage"))?;
    let faces: BTreeMap<_, _> = step.faces.iter().map(|face| (face.face, face)).collect();
    let traces: BTreeMap<_, _> = step
        .boundary_traces
        .iter()
        .map(|trace| (trace.face, trace))
        .collect();
    let expected: BTreeSet<_> = interval
        .boundaries
        .iter()
        .filter(|value| {
            matches!(
                value,
                BoundaryForcing::Stage(_) | BoundaryForcing::Discharge(_)
            )
        })
        .map(BoundaryForcing::face)
        .collect();
    if traces.len() != step.boundary_traces.len() || !traces.keys().eq(expected.iter()) {
        return Err(rejected(
            "boundary traces",
            "require one diagnostic for each physical boundary and none for full-flux ports",
        ));
    }
    let transmissive: BTreeMap<_, _> = step
        .transmissive_states
        .iter()
        .map(|state| (state.face, state))
        .collect();
    let expected_transmissive: BTreeSet<_> = interval
        .boundaries
        .iter()
        .filter(|value| matches!(value, BoundaryForcing::Transmissive(_)))
        .map(BoundaryForcing::face)
        .collect();
    if transmissive.len() != step.transmissive_states.len()
        || !transmissive.keys().eq(expected_transmissive.iter())
    {
        return Err(rejected(
            "transmissive states",
            "require exactly one witness for each declared transmissive face",
        ));
    }
    let (dx, dy) = data.setup.domain.grid().spacing();
    let relative = data
        .setup
        .config
        .boundary
        .characteristic_relative
        .max(64.0 * f64::EPSILON);
    let mut boundary_speeds = [0.0_f64; 2];
    for value in &interval.boundaries {
        let face = faces.get(&value.face()).ok_or_else(|| {
            rejected(
                "boundary face",
                format!("face={:?}: missing numerical flux", value.face()),
            )
        })?;
        let sign = if face.left.is_some() { 1.0 } else { -1.0 };
        let width = if face.face.axis == Axis::X { dy } else { dx };
        if face.pressure_corrections != [0.0; 2] {
            return Err(rejected(
                "boundary pressure",
                format!(
                    "face={:?}: physical flux already includes pressure",
                    face.face
                ),
            ));
        }
        let axis = if face.face.axis == Axis::X { 0 } else { 1 };
        let expected_flux = match value {
            BoundaryForcing::Flux(port) => {
                if face.signal_speed != port.signal_speed {
                    return Err(rejected(
                        "boundary CFL",
                        format!(
                            "face={:?}: signal speed differs from the declared full-flux port",
                            face.face
                        ),
                    ));
                }
                boundary_speeds[axis] = boundary_speeds[axis].max(port.signal_speed);
                [
                    sign * port.water_rate / width,
                    sign * port.momentum_rate[0] / width,
                    sign * port.momentum_rate[1] / width,
                ]
            }
            BoundaryForcing::Stage(_) | BoundaryForcing::Discharge(_) => {
                let trace = traces
                    .get(&value.face())
                    .ok_or_else(|| rejected("boundary trace", "missing physical diagnostic"))?;
                let numbers = [
                    trace.depth,
                    trace.normal_velocity,
                    trace.tangential_velocity,
                    trace.outward_water_rate,
                    trace.discharge_residual_per_width,
                    trace.characteristic_velocity_residual,
                    trace.signal_speed,
                ];
                if numbers.iter().any(|value| !value.is_finite())
                    || trace.depth <= 0.0
                    || trace.signal_speed <= 0.0
                {
                    return Err(rejected(
                        "boundary trace",
                        format!(
                            "trace={trace:?}: require finite wet trace and positive signal bound"
                        ),
                    ));
                }
                if face.signal_speed != trace.signal_speed {
                    return Err(rejected(
                        "boundary CFL",
                        format!("face={:?}: trace and face signal speeds differ", face.face),
                    ));
                }
                boundary_speeds[axis] = boundary_speeds[axis].max(trace.signal_speed);
                let c = data.setup.config.gravity.sqrt() * trace.depth.sqrt();
                let margin = data.setup.config.boundary.critical_relative
                    * trace.normal_velocity.abs().max(c);
                if trace.normal_velocity - c >= -margin
                    || trace.normal_velocity + c <= margin
                    || trace.signal_speed < trace.normal_velocity.abs() + c
                {
                    return Err(rejected(
                        "boundary regime",
                        format!(
                            "trace={trace:?}: require a resolved subcritical trace and conservative speed"
                        ),
                    ));
                }
                if trace.characteristic_velocity_residual.abs()
                    > data.setup.config.boundary.characteristic_relative * 3.0 * c
                {
                    return Err(rejected(
                        "boundary compatibility",
                        format!(
                            "trace={trace:?}: characteristic residual exceeds the admitted local scale"
                        ),
                    ));
                }
                let incoming = match value {
                    BoundaryForcing::Stage(stage) => {
                        let cell =
                            crate::solver::boundary::exterior_cell(&data.setup.domain, stage.face)?;
                        if trace.depth != stage.elevation - data.setup.domain.bed()[cell]
                            || trace.discharge_residual_per_width != 0.0
                        {
                            return Err(rejected(
                                "boundary stage",
                                format!(
                                    "trace={trace:?}: depth or residual contradicts the prescribed stage"
                                ),
                            ));
                        }
                        stage.inflow_tangential_velocity
                    }
                    BoundaryForcing::Discharge(discharge) => {
                        let target = discharge.outward_rate / width;
                        let achieved = trace.outward_water_rate / width;
                        let tolerance = data.setup.config.boundary.discharge_absolute
                            + data.setup.config.boundary.discharge_relative
                                * target.abs().max(achieved.abs());
                        if (target - achieved).abs() > tolerance
                            || trace.discharge_residual_per_width.abs() > tolerance
                        {
                            return Err(rejected(
                                "boundary discharge",
                                format!(
                                    "trace={trace:?}: achieved rate or compatibility residual contradicts the prescribed discharge"
                                ),
                            ));
                        }
                        discharge.inflow_tangential_velocity
                    }
                    _ => None,
                };
                if trace.outward_water_rate < 0.0 && incoming != Some(trace.tangential_velocity) {
                    // Multiplication and division through conserved tangent momentum can round once.
                    if !incoming.is_some_and(|velocity| {
                        close(velocity, trace.tangential_velocity, relative)
                    }) {
                        return Err(rejected(
                            "boundary tangent",
                            format!(
                                "trace={trace:?}: supply the declared incoming tangent velocity"
                            ),
                        ));
                    }
                }
                let q = trace.outward_water_rate / width;
                if !close(q, trace.depth * trace.normal_velocity, relative) {
                    return Err(rejected(
                        "boundary water rate",
                        format!(
                            "trace={trace:?}: rate contradicts trace depth and normal velocity"
                        ),
                    ));
                }
                let normal = q * trace.normal_velocity
                    + 0.5 * data.setup.config.gravity * trace.depth * trace.depth;
                let tangent = sign * q * trace.tangential_velocity;
                if face.face.axis == Axis::X {
                    [sign * q, normal, tangent]
                } else {
                    [sign * q, tangent, normal]
                }
            }
            BoundaryForcing::Exterior(_) => continue,
            BoundaryForcing::Transmissive(_) => {
                let state = transmissive.get(&value.face()).ok_or_else(|| {
                    rejected("transmissive state", "missing accepted trial witness")
                })?;
                boundary_speeds[axis] = boundary_speeds[axis].max(face.signal_speed);
                super::transmissive::flux(state, face, width, &data.setup.config)?
            }
        };
        if face
            .flux
            .iter()
            .zip(expected_flux)
            .any(|(actual, expected)| !close(*actual, expected, relative))
        {
            return Err(rejected(
                "boundary flux",
                format!("face={face:?}: flux contradicts the declared forcing or physical trace"),
            ));
        }
    }
    // Boundary speeds provide a necessary CFL bound. Interior waves can make the actual bound stricter.
    let frequency = boundary_speeds[0] / dx + boundary_speeds[1] / dy;
    if !frequency.is_finite() {
        return Err(rejected(
            "boundary CFL",
            format!("frequency={frequency}: require representable directional boundary speeds"),
        ));
    }
    let necessary_limit = if frequency > 0.0 {
        data.setup.config.cfl / frequency
    } else {
        f64::INFINITY
    };
    if !bounded(step.cfl_limit, necessary_limit) || !bounded(step.dt, necessary_limit) {
        return Err(rejected(
            "boundary CFL",
            format!(
                "stored_limit={}, dt={}, boundary_limit={necessary_limit}: boundary signal speeds contradict the accepted interval",
                step.cfl_limit, step.dt
            ),
        ));
    }
    Ok(())
}

fn close(left: f64, right: f64, relative: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= relative * left.abs().max(right.abs())
}
