use super::engine::Accepted;
use super::{EngineError, Solver, StepDiagnostics};
use crate::cpu::{CpuState, CpuTensor, transport};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, ReduceOps, ScalarOps, TypeConversionOps,
    UnaryOps,
};

impl Solver {
    pub(super) fn next_accepted(&self, target: f64) -> Result<Accepted, EngineError> {
        let config = self.config();
        let client = &self.cpu.client;
        let time = self.time();
        let index = self
            .definition
            .forcing
            .intervals()
            .partition_point(|interval| interval.end <= time);
        let forcing = self
            .definition
            .forcing
            .intervals()
            .get(index)
            .ok_or_else(|| {
                EngineError::rejected(
                    "forcing",
                    format!("time={time}: supply the next explicit interval"),
                )
            })?;
        let source = &self.cpu.sources[index];
        let endpoint = target.min(forcing.end).min(time + config.max_step);
        let mut dt = endpoint - time;
        let mut last_reason = String::new();
        let zero = &self.cpu.zero;
        let one = client.add_scalar(zero, 1.0)?;
        let area = self.domain().grid().cell_area();
        let original = &self.accepted.cpu;
        for retry in 0..=config.max_retries {
            let end = if retry == 0 { endpoint } else { time + dt };
            dt = end - time;
            if !dt.is_finite() || dt <= 0.0 || (dt < config.min_step && end != endpoint) {
                return Err(EngineError::rejected(
                    "timestep",
                    format!(
                        "time={time}, dt={dt}, retry={retry}: cannot advance within min_step={}; inspect inputs and prior trial: {last_reason}",
                        config.min_step
                    ),
                ));
            }
            // Rain and donor losses execute on persistent numr CPU tensors.
            let rain = client.mul_scalar(&source.rain, dt)?;
            let loss = client.mul_scalar(&source.withdrawal, dt)?;
            let available = client.add(&original.h, &rain)?;
            self.finite(&rain, "rain depth")?;
            self.finite(&loss, "withdrawal depth")?;
            self.finite(&available, "post-rain depth")?;
            let underflow = client.maximum(
                &client.mul(&client.eq(&rain, zero)?, &client.gt(&source.rain, zero)?)?,
                &client.mul(
                    &client.eq(&loss, zero)?,
                    &client.gt(&source.withdrawal, zero)?,
                )?,
            )?;
            if self.any(&client.maximum(&client.gt(&loss, &available)?, &underflow)?)? {
                last_reason = format!(
                    "time={time}, dt={dt}: prescribed land losses exceed available water or source increments underflow"
                );
                dt *= 0.5;
                continue;
            }
            let h = client.sub(&available, &loss)?;
            let donor_depth = client.where_cond(&client.gt(&available, zero)?, &available, &one)?;
            let ratio = client.div(&h, &donor_depth)?;
            let trial = CpuState {
                h,
                qx: client.mul(&original.qx, &ratio)?,
                qy: client.mul(&original.qy, &ratio)?,
            };
            self.finite(&trial.qx, "post-source qx")?;
            self.finite(&trial.qy, "post-source qy")?;
            let flux = match transport::calculate(&self.cpu, &trial, source, config, area) {
                Ok(flux) => flux,
                Err(error @ EngineError::Boundary { .. }) => {
                    last_reason = error.to_string();
                    dt *= 0.5;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let (dx, dy) = self.domain().grid().spacing();
            let frequency = flux.speeds[0] / dx + flux.speeds[1] / dy;
            if !frequency.is_finite() {
                return Err(EngineError::rejected(
                    "CFL",
                    format!("frequency={frequency}: rescale geometry and velocity"),
                ));
            }
            let cfl_limit = if frequency > 0.0 {
                config.cfl / frequency
            } else {
                f64::INFINITY
            };
            self.finite(&flux.outgoing, "outgoing rate")?;
            let exporting = client.gt(&flux.outgoing, zero)?;
            if self.any(&client.mul(&client.eq(&trial.h, zero)?, &exporting)?)? {
                return Err(EngineError::rejected(
                    "depletion",
                    format!(
                        "time={time}: a dry donor exports water, correct the source-complete port or numerical state"
                    ),
                ));
            }
            let denominator = client.where_cond(&exporting, &flux.outgoing, &one)?;
            let bounds = client.div(&client.mul_scalar(&trial.h, area)?, &denominator)?;
            let unlimited = client.add_scalar(zero, f64::INFINITY)?;
            let bounds = client.where_cond(&exporting, &bounds, &unlimited)?;
            let depletion_limit = client.min(&bounds, &[0], false)?.item::<f64>()?;
            let limit = cfl_limit.min(depletion_limit);
            if dt > limit {
                last_reason = format!("CFL limit={cfl_limit}, depletion limit={depletion_limit}");
                // Room for time subtraction avoids repeating an identical rounded trial.
                dt = 0.95 * limit;
                continue;
            }
            let next_h = client.add(&trial.h, &client.mul_scalar(&flux.derivative.h, dt)?)?;
            let next_qx = client.add(&trial.qx, &client.mul_scalar(&flux.derivative.qx, dt)?)?;
            let next_qy = client.add(&trial.qy, &client.mul_scalar(&flux.derivative.qy, dt)?)?;
            self.finite(&next_h, "transport depth")?;
            let dry_momentum = client.mul(
                &client.eq(&next_h, zero)?,
                &client.gt(
                    &client.add(&client.abs(&next_qx)?, &client.abs(&next_qy)?)?,
                    zero,
                )?,
            )?;
            if self.any(&client.maximum(&client.lt(&next_h, zero)?, &dry_momentum)?)? {
                last_reason = format!(
                    "time={time}, dt={dt}: transport creates negative depth or momentum without water"
                );
                dt *= 0.5;
                continue;
            }
            let local = client.sub(&next_h, &original.h)?;
            let local = client.sub(&local, &rain)?;
            let local = client.add(&local, &loss)?;
            let local = client.sub(&local, &client.mul_scalar(&flux.derivative.h, dt)?)?;
            let local = client.abs(&client.mul_scalar(&local, area)?)?;
            let max_local_residual = client.max(&local, &[0], false)?.item::<f64>()?;
            let mut next = self.friction(
                CpuState {
                    h: next_h,
                    qx: next_qx,
                    qy: next_qy,
                },
                dt,
            )?;
            let reset = client.mul(
                &client.gt(&next.h, zero)?,
                &client.lt(
                    &next.h,
                    &client.add_scalar(zero, config.momentum_film_depth)?,
                )?,
            )?;
            let removed_x = client.where_cond(&reset, &next.qx, zero)?;
            let removed_y = client.where_cond(&reset, &next.qy, zero)?;
            let momentum_removed = [
                self.total(&client.mul_scalar(&removed_x, area)?)?,
                self.total(&client.mul_scalar(&removed_y, area)?)?,
            ];
            next.qx = client.where_cond(&reset, zero, &next.qx)?;
            next.qy = client.where_cond(&reset, zero, &next.qy)?;
            self.finite(&next.qx, "accepted qx")?;
            self.finite(&next.qy, "accepted qy")?;
            let storage_before = self.ledger().storage;
            let storage_after = self.total(&client.mul_scalar(&next.h, area)?)?;
            let rain_volume = self.total(&client.mul_scalar(&rain, area)?)?;
            let withdrawal_volume = self.total(&client.mul_scalar(&loss, area)?)?;
            let boundary_in = flux.boundary_in_rate * dt;
            let boundary_out = flux.boundary_out_rate * dt;
            let residual = (storage_after - storage_before)
                - (rain_volume - withdrawal_volume + boundary_in - boundary_out);
            let scale = storage_before
                .max(storage_after)
                .max(rain_volume)
                .max(withdrawal_volume)
                .max(boundary_in)
                .max(boundary_out)
                .max(1.0);
            let tolerance = config.volume_absolute + config.volume_relative * scale;
            if ![
                storage_after,
                rain_volume,
                withdrawal_volume,
                boundary_in,
                boundary_out,
                residual,
                max_local_residual,
                momentum_removed[0],
                momentum_removed[1],
                tolerance,
            ]
            .iter()
            .all(|value| value.is_finite())
                || residual.abs() > tolerance
                || max_local_residual > tolerance
            {
                return Err(EngineError::rejected(
                    "water accounting",
                    format!(
                        "time={time}, dt={dt}, residual={residual}, max_local={max_local_residual}, tolerance={tolerance}: inspect sources and flux accounting"
                    ),
                ));
            }
            let diagnostics = StepDiagnostics {
                start: time,
                end,
                dt,
                retries: retry,
                cfl_limit,
                depletion_limit,
                storage_before,
                storage_after,
                rain_volume,
                withdrawal_volume,
                boundary_in,
                boundary_out,
                residual,
                normalized_residual: residual / scale,
                max_local_residual,
                momentum_removed,
                faces: flux.results(&self.cpu, dt)?,
                transmissive_states: source.exterior.results(flux.transmissive_state.as_ref())?,
                boundary_traces: match &flux.boundary_trace {
                    Some(trace) => source.characteristic.results(&self.cpu, trace)?,
                    None => Vec::new(),
                },
            };
            let ledger = self.ledger().accepted(&diagnostics)?;
            let cumulative_scale = ledger
                .initial_storage
                .max(ledger.storage)
                .max(ledger.rain)
                .max(ledger.withdrawal)
                .max(ledger.boundary_in)
                .max(ledger.boundary_out)
                .max(1.0);
            let cumulative_tolerance =
                config.volume_absolute + config.volume_relative * cumulative_scale;
            if !cumulative_tolerance.is_finite() || ledger.residual.abs() > cumulative_tolerance {
                return Err(EngineError::rejected(
                    "cumulative ledger",
                    format!(
                        "residual={}, tolerance={cumulative_tolerance}: inspect accumulated drift",
                        ledger.residual
                    ),
                ));
            }
            // This readback serves outputs and input checks, never subsequent numerical execution.
            let state = next.to_flow(self.domain())?;
            let steps = self.accepted.steps.checked_add(1).ok_or_else(|| {
                EngineError::rejected(
                    "step counter",
                    "accepted step count overflows: start another run",
                )
            })?;
            return Ok(Accepted {
                time: end,
                state,
                ledger,
                steps,
                last: Some(diagnostics),
                cpu: next,
            });
        }
        Err(EngineError::rejected(
            "trial retries",
            format!("time={time}, limit={}: {last_reason}", config.max_retries),
        ))
    }

    fn friction(&self, state: CpuState, dt: f64) -> Result<CpuState, EngineError> {
        let client = &self.cpu.client;
        let zero = &self.cpu.zero;
        let wet = client.gt(&state.h, zero)?;
        let depth = client.where_cond(&wet, &state.h, &client.add_scalar(zero, 1.0)?)?;
        let speed = client.hypot(&state.qx, &state.qy)?;
        self.finite(&speed, "Manning discharge magnitude")?;
        let active_drag = client.mul(
            &client.gt(&speed, zero)?,
            &client.gt(&self.cpu.manning, zero)?,
        )?;
        let power = client.pow_scalar(&depth, 7.0 / 3.0)?;
        let safe_power = client.where_cond(&active_drag, &power, &client.add_scalar(zero, 1.0)?)?;
        let drag = client.mul(&self.cpu.manning, &self.cpu.manning)?;
        let drag = client.mul(&drag, &speed)?;
        let drag = client.mul_scalar(&drag, self.config().gravity * dt)?;
        let drag = client.div(&drag, &safe_power)?;
        let denominator = client.add_scalar(&drag, 1.0)?;
        self.finite(&denominator, "Manning denominator")?;
        Ok(CpuState {
            h: state.h,
            qx: client.div(&state.qx, &denominator)?,
            qy: client.div(&state.qy, &denominator)?,
        })
    }

    fn total(&self, values: &CpuTensor) -> Result<f64, EngineError> {
        Ok(self.cpu.client.sum(values, &[0], false)?.item::<f64>()?)
    }

    fn any(&self, mask: &CpuTensor) -> Result<bool, EngineError> {
        Ok(self
            .cpu
            .client
            .any(&self.cpu.client.cast(mask, DType::F64)?, &[0], false)?
            .item::<f64>()?
            != 0.0)
    }

    fn finite(&self, values: &CpuTensor, name: &'static str) -> Result<(), EngineError> {
        let client = &self.cpu.client;
        let invalid = client.maximum(
            &client.cast(&client.isnan(values)?, DType::F64)?,
            &client.cast(&client.isinf(values)?, DType::F64)?,
        )?;
        if self.any(&invalid)? {
            let values = values.try_to_vec::<f64>()?;
            let first = values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite());
            return Err(EngineError::rejected(
                name,
                format!(
                    "time={}, first nonfinite input={first:?}: inspect physical scales and tensor operations",
                    self.time()
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::CartesianGrid;
    use crate::model::{CellState, Domain, FlowState, SpatialReference};
    use crate::solver::{BoundaryConditions, ForcingInterval, ForcingSchedule, SolverConfig};

    #[test]
    fn numr_friction_matches_exact_fixed_depth_vector_decay()
    -> Result<(), Box<dyn std::error::Error>> {
        let domain = Domain::new(
            CartesianGrid::new(1, 1, 1.0, 1.0)?,
            vec![0.0],
            vec![0.03],
            vec![true],
            SpatialReference::new(
                "synthetic metric plane".into(),
                "synthetic datum".into(),
                [0.0; 2],
            )?,
        )?;
        let state = FlowState::new(
            &domain,
            vec![CellState {
                h: 1.0,
                qx: 2.0,
                qy: 1.0,
            }],
        )?;
        let walls = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &walls,
            vec![ForcingInterval {
                start: 0.0,
                end: 1.0,
                rain: vec![0.0],
                withdrawal: vec![0.0],
                boundaries: vec![],
            }],
        )?;
        let solver = Solver::new(domain, state, walls, forcing, SolverConfig::default(), 0.0)?;
        let result = solver
            .friction(solver.accepted.cpu.clone(), 0.1)?
            .to_flow(solver.domain())?;
        let denominator = 1.0 + 9.81 * 0.03_f64.powi(2) * 5.0_f64.sqrt() * 0.1;
        assert!((result.cells()[0].qx - 2.0 / denominator).abs() <= 1e-12);
        assert!((result.cells()[0].qy - 1.0 / denominator).abs() <= 1e-12);
        assert_eq!(result.cells()[0].h, 1.0);
        Ok(())
    }
}
