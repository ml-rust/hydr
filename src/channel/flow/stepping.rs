use super::{
    ChannelBoundaryInterval, ChannelError, ChannelReach, ChannelStep, ChannelStepDiagnostics,
    accounting::Impulses, engine::Accepted, receipt::LastInterval,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, MidpointOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Calculate one accepted interval without modifying the input reach.
    ///
    /// The returned endpoint can precede the request because of numerical interval limits.
    pub fn step(
        &self,
        requested_end_s: f64,
        forcing: &ChannelBoundaryInterval,
    ) -> Result<ChannelStep<R>, ChannelError> {
        forcing.check(self.time(), requested_end_s, self.definition())?;
        let context = &self.context;
        let c = &context.client;
        let config = self.config();
        let count = context.count();
        let endpoint = requested_end_s.min(self.time() + config.max_step);
        let mut dt = endpoint - self.time();
        let mut reason = String::new();
        for retries in 0..=config.max_retries {
            let end = if retries == 0 {
                endpoint
            } else {
                self.time() + dt
            };
            dt = end - self.time();
            if !dt.is_finite() || dt <= 0.0 || (dt < config.min_step && end != endpoint) {
                return Err(ChannelError::numerical(format!(
                    "channel time={}, dt={dt}, retry={retries}, min_step={}: cannot advance; inspect preceding trial: {reason}",
                    self.time(),
                    config.min_step
                )));
            }
            let fields = context.fields(&self.accepted.volume, &self.accepted.momentum)?;
            let flux = context.flux(&fields, forcing)?;
            let weighted_speed = c.mul(&context.face_width, &flux.speed)?;
            let moving_signal = c.gt(&flux.speed, &context.constant(&flux.speed, 0.0)?)?;
            context.represented(&moving_signal, &weighted_speed, "width-weighted wave speed")?;
            let frequency = c.div(
                &c.add(
                    &weighted_speed.narrow(0, 0, count)?,
                    &weighted_speed.narrow(0, 1, count)?,
                )?,
                &c.mul(&context.width, &context.length)?,
            )?;
            context.finite(&frequency, "width-aware CFL frequency")?;
            let frequency = c.max(&frequency, &[0], false)?.item::<f64>()?;
            let cfl_limit = if frequency > 0.0 {
                config.cfl / frequency
            } else {
                f64::INFINITY
            };
            let zero = context.constant(&self.accepted.volume, 0.0)?;
            let one = context.constant(&self.accepted.volume, 1.0)?;
            let outgoing = c.add(
                &c.maximum(&flux.water.narrow(0, 1, count)?, &zero)?,
                &c.maximum(&c.neg(&flux.water.narrow(0, 0, count)?)?, &zero)?,
            )?;
            context.finite(&outgoing, "all-outgoing water rate")?;
            let exporting = c.gt(&outgoing, &zero)?;
            let safe_outgoing = c.where_cond(&exporting, &outgoing, &one)?;
            let bounds = c.div(&self.accepted.volume, &safe_outgoing)?;
            let bounds = c.where_cond(
                &exporting,
                &bounds,
                &context.constant(&bounds, f64::INFINITY)?,
            )?;
            let depletion_limit = c.min(&bounds, &[0], false)?.item::<f64>()?;
            let limit = cfl_limit.min(depletion_limit);
            if dt > limit {
                reason = format!(
                    "width-aware CFL={cfl_limit}, all-outgoing depletion={depletion_limit}"
                );
                dt = 0.95 * limit;
                continue;
            }
            let face_volume = c.mul_scalar(&flux.water, dt)?;
            let transfer_underflow_bound = context.product_bound(
                &flux.water,
                &context.constant(&flux.water, 0.0)?,
                &context.constant(&flux.water, dt)?,
                &face_volume,
                "shared numerical face volume",
            )?;
            let (face_momentum, face_momentum_bound) = context.integrate_momentum(
                &flux.momentum,
                &flux.momentum_bound,
                dt,
                "shared face momentum",
            )?;
            let (left_pressure, left_pressure_bound) = context.integrate_momentum(
                &flux.left_pressure,
                &flux.left_pressure_bound,
                dt,
                "left pressure impulse",
            )?;
            let (right_pressure, right_pressure_bound) = context.integrate_momentum(
                &flux.right_pressure,
                &flux.right_pressure_bound,
                dt,
                "right pressure impulse",
            )?;
            let (left_reconstructed_pressure, left_reconstructed_bound) = context
                .integrate_momentum(
                    &flux.left_reconstructed_pressure,
                    &flux.left_reconstructed_pressure_bound,
                    dt,
                    "left reconstructed pressure impulse",
                )?;
            let (right_reconstructed_pressure, right_reconstructed_bound) = context
                .integrate_momentum(
                    &flux.right_reconstructed_pressure,
                    &flux.right_reconstructed_pressure_bound,
                    dt,
                    "right reconstructed pressure impulse",
                )?;
            let left_bound = context.bound_add(
                &face_momentum_bound.narrow(0, 0, count)?,
                &right_pressure_bound.narrow(0, 0, count)?,
            )?;
            let left_bound =
                context.bound_add(&left_bound, &right_reconstructed_bound.narrow(0, 0, count)?)?;
            let right_bound = context.bound_add(
                &face_momentum_bound.narrow(0, 1, count)?,
                &left_pressure_bound.narrow(0, 1, count)?,
            )?;
            let right_bound =
                context.bound_add(&right_bound, &left_reconstructed_bound.narrow(0, 1, count)?)?;
            let momentum_underflow_bound = context.bound_add(&left_bound, &right_bound)?;
            let max_momentum_underflow_bound = c
                .max(&momentum_underflow_bound, &[0], false)?
                .item::<f64>()?;
            let volume = c.add(
                &self.accepted.volume,
                &c.sub(
                    &face_volume.narrow(0, 0, count)?,
                    &face_volume.narrow(0, 1, count)?,
                )?,
            )?;
            // Original cell pressure cancels between its two faces before rounded subtraction.
            let left_momentum = c.sub(
                &face_momentum.narrow(0, 0, count)?,
                &right_reconstructed_pressure.narrow(0, 0, count)?,
            )?;
            let right_momentum = c.sub(
                &face_momentum.narrow(0, 1, count)?,
                &left_reconstructed_pressure.narrow(0, 1, count)?,
            )?;
            let transported = c.add(
                &self.accepted.momentum,
                &c.sub(&left_momentum, &right_momentum)?,
            )?;
            match context.fields(&volume, &transported) {
                Ok(_) => (),
                Err(error @ ChannelError::Numerical { .. }) => {
                    reason = error.to_string();
                    dt *= 0.5;
                    continue;
                }
                Err(error) => return Err(error),
            }
            let momentum = match context.friction(&volume, &transported, dt) {
                Ok(momentum) => momentum,
                Err(error @ ChannelError::Numerical { .. }) => {
                    reason = error.to_string();
                    dt *= 0.5;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let friction = c.sub(&momentum, &transported)?;
            context.fields(&volume, &momentum)?;
            let balance = context.balance(
                &self.accepted,
                &volume,
                &momentum,
                Impulses {
                    water: &face_volume,
                    momentum: &face_momentum,
                    left_pressure: &left_pressure,
                    right_pressure: &right_pressure,
                    friction: &friction,
                    momentum_underflow_bound: &momentum_underflow_bound,
                    transfer_underflow_bound: &transfer_underflow_bound,
                },
            )?;
            let steps = self.accepted.steps.checked_add(1).ok_or_else(|| {
                ChannelError::numerical("channel accepted count overflows: start another run")
            })?;
            let scalar = ChannelStepDiagnostics {
                start_s: self.time(),
                end_s: end,
                retries,
                cfl_limit_s: cfl_limit,
                depletion_limit_s: depletion_limit,
                storage_before_m3: self.accepted.ledger.storage_m3,
                storage_after_m3: balance.ledger.storage_m3,
                boundary_in_m3: balance.incoming,
                boundary_out_m3: balance.outgoing,
                residual_m3: balance.residual,
                max_local_residual_m3: balance.max_local,
                max_local_momentum_residual_m4_per_s: balance.max_momentum,
                max_momentum_underflow_bound_m4_per_s: max_momentum_underflow_bound,
                face_volume_m3: Vec::new(),
                face_momentum_m4_per_s: Vec::new(),
                left_pressure_impulse_m4_per_s: Vec::new(),
                right_pressure_impulse_m4_per_s: Vec::new(),
                friction_impulse_m4_per_s: Vec::new(),
                momentum_underflow_bound_m4_per_s: Vec::new(),
                face_transfer_underflow_bound_m3: Vec::new(),
                transfer_underflow_bound_m3: balance.transfer_charge,
            };
            let last = LastInterval {
                previous_volume: self.accepted.volume.clone(),
                previous_momentum: self.accepted.momentum.clone(),
                previous_ledger: self.accepted.ledger.clone(),
                previous_time: self.time(),
                previous_steps: self.accepted.steps,
                forcing: forcing.clone(),
                requested_end: requested_end_s,
                scalar,
                face_volume,
                face_momentum,
                left_pressure,
                right_pressure,
                friction,
                momentum_underflow_bound,
                transfer_underflow_bound,
            };
            return Ok(ChannelStep {
                next: Self {
                    context: Arc::clone(context),
                    accepted: Accepted {
                        volume,
                        momentum,
                        time: end,
                        ledger: balance.ledger,
                        steps,
                        last: Some(Arc::new(last)),
                    },
                },
            });
        }
        Err(ChannelError::numerical(format!(
            "channel time={}, retries={}: no accepted interval; inspect final trial: {reason}",
            self.time(),
            config.max_retries
        )))
    }
}

#[cfg(test)]
#[path = "stepping_tests.rs"]
mod tests;
