use super::engine::Accepted;
use super::{EngineError, Solver, StepDiagnostics};
use crate::tensor::land_sources::{SourceInputs, calculate as calculate_sources, checked_volume};
use crate::tensor::{TensorState, transport};
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, ReduceOps, ScalarOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>
        + HypotOps<R>
        + numr::ops::CumulativeOps<R>
        + numr::ops::ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    pub(super) fn next_accepted(&self, target: f64) -> Result<Accepted<R>, EngineError> {
        let config = self.config();
        let client = &self.context.client;
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
        let source = &self.context.sources[index];
        let endpoint = target.min(forcing.end).min(time + config.max_step);
        let mut dt = endpoint - time;
        let mut last_reason = String::new();
        let mut last_source_error = None;
        let zero = &self.context.zero;
        let one = client.add_scalar(zero, 1.0)?;
        let area = self.domain().grid().cell_area();
        let original = &self.accepted.flow;
        for retry in 0..=config.max_retries {
            let end = if retry == 0 { endpoint } else { time + dt };
            dt = end - time;
            if !dt.is_finite() || dt <= 0.0 || (dt < config.min_step && end != endpoint) {
                if let Some(error) = last_source_error {
                    return Err(error);
                }
                return Err(EngineError::rejected(
                    "timestep",
                    format!(
                        "time={time}, dt={dt}, retry={retry}: cannot advance within min_step={}; inspect inputs and prior trial: {last_reason}",
                        config.min_step
                    ),
                ));
            }
            let sources = match calculate_sources(
                &self.context,
                SourceInputs {
                    original,
                    land: &self.accepted.land,
                    forcing: source,
                },
                area,
                dt,
                self.ledger().water_adjustment_abs_bound,
                config,
            ) {
                Ok(sources) => {
                    last_source_error = None;
                    sources
                }
                Err(error @ EngineError::LandSource { .. })
                | Err(
                    error @ EngineError::Soil {
                        retryable: true, ..
                    },
                ) => {
                    last_reason = error.to_string();
                    last_source_error = Some(error);
                    dt *= 0.5;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let trial = &sources.state;
            let rain = &sources.rain;
            let loss = &sources.withdrawal;
            self.finite(&trial.qx, "post-source qx")?;
            self.finite(&trial.qy, "post-source qy")?;
            let flux = match transport::calculate(&self.context, trial, source, config, area) {
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
            let local = client.sub(&local, rain)?;
            let local = client.add(&local, loss)?;
            let local = client.add(&local, &sources.infiltration)?;
            let local = client.add(&local, &sources.evaporation)?;
            let local = client.sub(&local, &sources.adjustment)?;
            let local = client.sub(&local, &client.mul_scalar(&flux.derivative.h, dt)?)?;
            let local = client.abs(&client.mul_scalar(&local, area)?)?;
            let max_local_residual = client.max(&local, &[0], false)?.item::<f64>()?;
            let mut next = self.friction(
                TensorState {
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
            let storage_after = checked_volume(client, &next.h, area)?;
            let rain_volume = sources.rain_volume;
            let withdrawal_volume = sources.withdrawal_volume;
            let infiltration_volume = sources.infiltration_volume;
            let evaporation_volume = sources.evaporation_volume;
            let water_adjustment_volume = sources.adjustment_volume;
            let boundary_in = flux.boundary_in_rate * dt;
            let boundary_out = flux.boundary_out_rate * dt;
            let residual = (storage_after - storage_before)
                - (rain_volume - infiltration_volume - evaporation_volume - withdrawal_volume
                    + boundary_in
                    - boundary_out
                    + water_adjustment_volume);
            let scale = storage_before
                .max(storage_after)
                .max(rain_volume)
                .max(withdrawal_volume)
                .max(infiltration_volume)
                .max(evaporation_volume)
                .max(water_adjustment_volume.abs())
                .max(boundary_in)
                .max(boundary_out)
                .max(1.0);
            let tolerance = config.volume_absolute + config.volume_relative * scale;
            if ![
                storage_after,
                rain_volume,
                withdrawal_volume,
                infiltration_volume,
                evaporation_volume,
                water_adjustment_volume,
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
                infiltration_volume,
                evaporation_volume,
                soil_water_adjustment_volume: water_adjustment_volume,
                water_adjustment_volume,
                water_adjustment_abs_charge: sources.adjustment_charge,
                sources: Vec::new(),
                boundary_in,
                boundary_out,
                residual,
                normalized_residual: residual / scale,
                max_local_residual,
                momentum_removed,
                faces: Vec::new(),
                transmissive_states: Vec::new(),
                boundary_traces: Vec::new(),
            };
            let ledger = self.ledger().accepted(&diagnostics)?;
            let cumulative_scale = ledger
                .initial_storage
                .max(ledger.storage)
                .max(ledger.rain)
                .max(ledger.withdrawal)
                .max(ledger.infiltration)
                .max(ledger.evaporation)
                .max(ledger.water_adjustment.abs())
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
            self.check_admission(&next)?;
            let land = sources.soil.next.clone();
            let receipt = super::receipts::StepReceipt::Device(Box::new(
                super::receipts::DeviceStepReceipt {
                    scalar: diagnostics,
                    forcing_index: index,
                    flux,
                    sources,
                    start_depth: original.h.clone(),
                },
            ));
            let steps = self.accepted.steps.checked_add(1).ok_or_else(|| {
                EngineError::rejected(
                    "step counter",
                    "accepted step count overflows: start another run",
                )
            })?;
            return Ok(Accepted {
                time: end,
                land,
                ledger,
                steps,
                last: Some(receipt),
                flow: next,
            });
        }
        if let Some(error) = last_source_error {
            return Err(error);
        }
        Err(EngineError::rejected(
            "trial retries",
            format!("time={time}, limit={}: {last_reason}", config.max_retries),
        ))
    }

    fn friction(&self, state: TensorState<R>, dt: f64) -> Result<TensorState<R>, EngineError> {
        let client = &self.context.client;
        let zero = &self.context.zero;
        let wet = client.gt(&state.h, zero)?;
        let depth = client.where_cond(&wet, &state.h, &client.add_scalar(zero, 1.0)?)?;
        let speed = client.hypot(&state.qx, &state.qy)?;
        self.finite(&speed, "Manning discharge magnitude")?;
        let active_drag = client.mul(
            &client.gt(&speed, zero)?,
            &client.gt(&self.context.manning, zero)?,
        )?;
        let power = client.pow_scalar(&depth, 7.0 / 3.0)?;
        let safe_power = client.where_cond(&active_drag, &power, &client.add_scalar(zero, 1.0)?)?;
        let drag = client.mul(&self.context.manning, &self.context.manning)?;
        let drag = client.mul(&drag, &speed)?;
        let drag = client.mul_scalar(&drag, self.config().gravity * dt)?;
        let drag = client.div(&drag, &safe_power)?;
        let denominator = client.add_scalar(&drag, 1.0)?;
        self.finite(&denominator, "Manning denominator")?;
        Ok(TensorState {
            h: state.h,
            qx: client.div(&state.qx, &denominator)?,
            qy: client.div(&state.qy, &denominator)?,
        })
    }

    fn total(&self, values: &Tensor<R>) -> Result<f64, EngineError> {
        Ok(self
            .context
            .client
            .sum(values, &[0], false)?
            .item::<f64>()?)
    }

    fn any(&self, mask: &Tensor<R>) -> Result<bool, EngineError> {
        Ok(self
            .context
            .client
            .any(&self.context.client.cast(mask, DType::F64)?, &[0], false)?
            .item::<f64>()?
            != 0.0)
    }

    fn finite(&self, values: &Tensor<R>, name: &'static str) -> Result<(), EngineError> {
        let client = &self.context.client;
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
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

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
                potential_evaporation: vec![0.0],
                withdrawal: vec![0.0],
                boundaries: vec![],
            }],
        )?;
        let land = crate::soil::LandModel::impervious(&domain);
        let soil = crate::soil::SoilState::initialized(&land)?;
        let solver = Solver::<CpuRuntime>::new(
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
            crate::solver::SolverSetup {
                domain,
                land,
                boundaries: walls,
                forcing,
                config: SolverConfig::default(),
            },
            crate::solver::InitialState {
                time: 0.0,
                flow: state,
                soil,
            },
        )?;
        let result = solver
            .friction(solver.accepted.flow.clone(), 0.1)?
            .to_flow(solver.domain())?;
        let denominator = 1.0 + 9.81 * 0.03_f64.powi(2) * 5.0_f64.sqrt() * 0.1;
        assert!((result.cells()[0].qx - 2.0 / denominator).abs() <= 1e-12);
        assert!((result.cells()[0].qy - 1.0 / denominator).abs() <= 1e-12);
        assert_eq!(result.cells()[0].h, 1.0);
        Ok(())
    }
}
