//! Completion phase of one accepted-interval trial: transport update through the receipt.

use super::super::engine::Accepted;
use super::super::receipts::{DeviceStepReceipt, StepReceipt};
use super::super::tensor::TensorState;
use super::super::tensor::land_sources::checked_volume;
use super::super::{Solver, StepDiagnostics};
use super::prepare::{IntervalTrial, PreparedInterval};
use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

/// Exchange source applied between transport and friction.
///
/// Uninhabited until a coupled coordinator supplies exchange rates, so `None` is the only value.
pub(in crate::solver) enum ExchangeSource {}

/// Completion-phase outcome.
pub(in crate::solver) enum Completion<R: Runtime<DType = DType>> {
    /// The trial passes every check and becomes the next accepted state.
    Accepted(Box<Accepted<R>>),
    /// Transport creates negative depth or dry momentum. The caller halves the step.
    Retry(String),
}

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
    /// Runs transport, the optional exchange source, friction, film reset, accounting, and admission.
    ///
    /// The caller checks `prepared.limit()` against the step first.
    ///
    /// # Errors
    /// Returns nonfinite-state, water-accounting, cumulative-ledger, admission, and step-counter errors.
    pub(in crate::solver) fn complete_interval(
        &self,
        prepared: PreparedInterval<R>,
        exchange: Option<&ExchangeSource>,
    ) -> Result<Completion<R>, EngineError> {
        let PreparedInterval {
            trial:
                IntervalTrial {
                    index,
                    end,
                    dt,
                    retry,
                },
            sources,
            flux,
            cfl_limit,
            depletion_limit,
        } = prepared;
        let config = self.config();
        let client = &self.context.client;
        let time = self.time();
        let zero = &self.context.zero;
        let area = self.domain().grid().cell_area();
        let original = &self.accepted.flow;
        let trial = &sources.state;
        let rain = &sources.rain;
        let loss = &sources.withdrawal;
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
            return Ok(Completion::Retry(format!(
                "time={time}, dt={dt}: transport creates negative depth or momentum without water"
            )));
        }
        if let Some(exchange) = exchange {
            // The exchange source adds its depth change here and to both water balances below.
            match *exchange {}
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
        let receipt = StepReceipt::Device(Box::new(DeviceStepReceipt {
            scalar: diagnostics,
            forcing_index: index,
            flux,
            sources,
            start_depth: original.h.clone(),
        }));
        let steps = self.accepted.steps.checked_add(1).ok_or_else(|| {
            EngineError::rejected(
                "step counter",
                "accepted step count overflows: start another run",
            )
        })?;
        Ok(Completion::Accepted(Box::new(Accepted {
            time: end,
            land,
            ledger,
            steps,
            last: Some(receipt),
            flow: next,
        })))
    }
}
