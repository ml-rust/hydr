mod entry;

use super::{
    ChannelError, ChannelReach, ChannelStep, ChannelStepDiagnostics, accounting::Impulses,
    boundary::assembly::is_scalar, conserved_point::ConservedPoint, engine::Accepted,
    forcing::BoundChannelForcing, interval::IntervalControl, interval_start::IntervalStart,
    interval_start::ReservedStart, receipt::LastInterval, surveyed_residuals::ExpandedImpulses,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
    },
};
use std::sync::Arc;

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + TensorOps<R>
        + NextRepresentableOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Steps with `reserved` accepted-start operands when they belong to this reach.
    pub(super) fn step_bound_policy(
        &self,
        requested_end_s: f64,
        forcing: &BoundChannelForcing<R>,
        exact_endpoint: bool,
        reserved: Option<&ReservedStart<R>>,
    ) -> Result<ChannelStep<R>, ChannelError> {
        if !Arc::ptr_eq(&self.context.definition, &forcing.definition)
            && !self.definition().same_bits(&forcing.definition)
        {
            return Err(ChannelError::input(
                "bound forcing belongs to different geometry or source bindings: bind inputs with the destination reach",
            ));
        }
        super::definition::check_forcing_interval(
            self.time(),
            requested_end_s,
            forcing.start_s,
            forcing.end_s,
        )?;
        let context = &self.context;
        let c = &context.client;
        let config = self.config();
        let exact = forcing.lateral.has_interval_volumes;
        if exact && forcing.end_s.to_bits() != requested_end_s.to_bits() {
            return Err(ChannelError::input(format!(
                "integrated forcing end={}, requested_end={requested_end_s}: supply original volumes for the exact requested span",
                forcing.end_s
            )));
        }
        let control = IntervalControl {
            start: self.time(),
            requested_end: requested_end_s,
            minimum: config.min_step,
            exact: exact || exact_endpoint,
        };
        let endpoint = requested_end_s.min(self.time() + config.max_step);
        if (exact || exact_endpoint) && endpoint < requested_end_s {
            control.shorten(
                requested_end_s - self.time(),
                endpoint - self.time(),
                0,
                format!(
                    "maximum step={} requires a shorter integrated span",
                    config.max_step
                ),
            )?;
        }
        let mut dt = endpoint - self.time();
        let mut reason = String::new();
        // These depend only on the accepted state and the start forcing.
        let start =
            context.select_start(reserved, &self.accepted, &forcing.left, &forcing.right)?;
        let IntervalStart {
            donor,
            fields,
            flux,
            outgoing,
            start_cfl,
        } = &*start;
        let start_cfl = *start_cfl;
        for retries in 0..=config.max_retries {
            let end = if retries == 0 {
                endpoint
            } else {
                self.time() + dt
            };
            dt = end - self.time();
            if !dt.is_finite() || dt <= 0.0 || (dt < config.min_step && end != endpoint) {
                return Err(ChannelError::Unresolved {
                    dt_s: dt,
                    retries,
                    reason,
                });
            }
            let prepared = context.lateral.prepare(
                c,
                &forcing.lateral,
                &self.accepted.lateral,
                donor,
                &fields.velocity,
                dt,
            )?;
            let reservation = if prepared.has_nonzero_values {
                Some(context.combined_depletion_limit(
                    donor,
                    outgoing,
                    &prepared.withdrawal_volume,
                    &prepared.withdrawal_volume_bound,
                    dt,
                )?)
            } else {
                None
            };
            let (depletion_limit, donor_admitted) = if let Some(reservation) = &reservation {
                (reservation.limit_s, reservation.admitted)
            } else {
                let limit = context.face_depletion_limit(donor, outgoing)?;
                (limit, dt <= limit)
            };
            let limit = start_cfl.min(depletion_limit);
            if dt > start_cfl || !donor_admitted {
                reason = format!(
                    "accepted-start CFL={start_cfl}, combined original-donor depletion={depletion_limit}"
                );
                dt = control.shorten(dt, 0.95 * limit, retries, reason.clone())?;
                continue;
            }
            let transport = context.integrate_transport(flux, dt)?;
            if let Some(reservation) = &reservation
                && !context.actual_outgoing_admitted(
                    &reservation.remaining_lower,
                    &transport.face_volume,
                    &transport.face_volume_low,
                )?
            {
                reason = "actual outgoing shared face volumes exceed reserved original donor water: recompute a shorter interval".into();
                dt = control.shorten(dt, 0.95 * dt, retries, reason.clone())?;
                continue;
            }
            let application = if context.surveyed() {
                context.surveyed_application(&self.accepted, &transport, &prepared)
            } else {
                context.lateral.apply_lateral_and_transport(
                    c,
                    &self.accepted.volume.high,
                    &self.accepted.momentum.high,
                    &transport.water_delta.high,
                    &transport.momentum_delta.high,
                    &prepared,
                )
            };
            let application = match application {
                Ok(application) => application,
                Err(error @ ChannelError::Numerical { .. }) => {
                    reason = error.to_string();
                    dt = control.shorten(dt, dt * 0.5, retries, reason.clone())?;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let volume = ConservedPoint {
                high: application.volume.clone(),
                low: application.volume_low.clone(),
            };
            let candidate_momentum = ConservedPoint {
                high: application.momentum.clone(),
                low: application.momentum_low.clone(),
            };
            let candidate_fields = match context.fields(&volume, &candidate_momentum) {
                Ok(fields) => fields,
                Err(error @ ChannelError::Numerical { .. }) => {
                    reason = error.to_string();
                    dt = control.shorten(dt, dt * 0.5, retries, reason.clone())?;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let mut candidate_cfl = if prepared.has_nonzero_values {
                let speed =
                    match context.candidate_speed(&candidate_fields, &forcing.left, &forcing.right)
                    {
                        Ok(speed) => speed,
                        Err(error @ ChannelError::Numerical { .. }) => {
                            reason = error.to_string();
                            dt = control.shorten(dt, dt * 0.5, retries, reason.clone())?;
                            continue;
                        }
                        Err(error) => return Err(error),
                    };
                let limit = context.cfl_limit(
                    &speed.outer,
                    speed.internal.as_ref(),
                    speed.fronts.as_ref(),
                )?;
                if dt > limit {
                    reason = format!("prefriction candidate CFL={limit} is below duration={dt}");
                    dt = control.shorten(dt, 0.95 * limit, retries, reason.clone())?;
                    continue;
                }
                Some(limit)
            } else {
                None
            };
            let friction_geometry;
            let friction;
            let friction_low;
            let (momentum, momentum_bound, max_momentum_bound) = if context.surveyed() {
                let trial = match context.surveyed_friction_point(
                    &application.volume,
                    &candidate_momentum,
                    dt,
                ) {
                    Ok(trial) => trial,
                    Err(error @ ChannelError::Numerical { .. }) => {
                        reason = error.to_string();
                        dt = control.shorten(dt, dt * 0.5, retries, reason.clone())?;
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                let bound = context.bound_add(
                    &transport.momentum_underflow_bound,
                    &application.momentum_selection_bound,
                )?;
                let bound = context.bound_add(&bound, &trial.radius)?;
                let max = c.max(&bound, &[0], false)?.item::<f64>()?;
                friction = trial.impulse.high;
                friction_low = trial.impulse.low;
                friction_geometry = trial.geometry;
                (trial.point, bound, max)
            } else {
                let trial =
                    match context.friction_trial(&application.volume, &application.momentum, dt) {
                        Ok(trial) => trial,
                        Err(error @ ChannelError::Numerical { .. }) => {
                            reason = error.to_string();
                            dt = control.shorten(dt, dt * 0.5, retries, reason.clone())?;
                            continue;
                        }
                        Err(error) => return Err(error),
                    };
                friction = c.sub(&trial.momentum, &application.momentum)?;
                friction_low = context.constant(&friction, 0.0)?;
                friction_geometry = trial.geometry;
                (
                    ConservedPoint {
                        high: trial.momentum,
                        low: application.momentum_low.clone(),
                    },
                    transport.momentum_underflow_bound.clone(),
                    transport.max_momentum_underflow_bound,
                )
            };
            let has_scalar_boundary = is_scalar(&forcing.left) || is_scalar(&forcing.right);
            let final_fields = match context.fields(&volume, &momentum) {
                Ok(fields) => fields,
                Err(error @ ChannelError::Numerical { .. }) if has_scalar_boundary => {
                    reason = error.to_string();
                    dt = control.shorten(dt, dt * 0.5, retries, reason.clone())?;
                    continue;
                }
                Err(error) => return Err(error),
            };
            if has_scalar_boundary {
                let speed =
                    match context.candidate_speed(&final_fields, &forcing.left, &forcing.right) {
                        Ok(speed) => speed,
                        Err(error @ ChannelError::Numerical { .. }) => {
                            reason = error.to_string();
                            dt = control.shorten(dt, dt * 0.5, retries, reason.clone())?;
                            continue;
                        }
                        Err(error) => return Err(error),
                    };
                let limit = context.cfl_limit(
                    &speed.outer,
                    speed.internal.as_ref(),
                    speed.fronts.as_ref(),
                )?;
                if dt > limit {
                    reason =
                        format!("postfriction scalar boundary CFL={limit} is below duration={dt}");
                    dt = control.shorten(dt, 0.95 * limit, retries, reason.clone())?;
                    continue;
                }
                candidate_cfl = Some(candidate_cfl.map_or(limit, |source| source.min(limit)));
            }
            let lateral = context
                .lateral
                .finalize_state(&prepared, application.water_assembly_charge_m3)?;
            let expanded = context.surveyed().then_some(ExpandedImpulses {
                face_volume_low: &transport.face_volume_low,
                face_momentum_low: &transport.face_momentum_low,
                friction_low: &friction_low,
                selection_bound: &application.water_selection_bound,
            });
            let balance = match context.balance(
                &self.accepted,
                &volume,
                &momentum,
                Impulses {
                    water: &transport.face_volume,
                    momentum: &transport.face_momentum,
                    left_pressure: &transport.left_pressure,
                    right_pressure: &transport.right_pressure,
                    friction: &friction,
                    momentum_underflow_bound: &momentum_bound,
                    transfer_underflow_bound: &transport.transfer_underflow_bound,
                    lateral: &prepared,
                    application: &application,
                    next_lateral: &lateral,
                    expanded,
                },
            ) {
                Ok(balance) => balance,
                Err(error @ ChannelError::Numerical { .. }) if prepared.has_nonzero_values => {
                    reason = error.to_string();
                    dt = control.shorten(dt, dt * 0.5, retries, reason.clone())?;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let steps = self.accepted.steps.checked_add(1).ok_or_else(|| {
                ChannelError::numerical("channel accepted count overflows: start another run")
            })?;
            let scalar = ChannelStepDiagnostics {
                boundary_traces: [None, None],
                riemann: None,
                friction_geometry: None,
                start_s: self.time(),
                end_s: end,
                retries,
                cfl_limit_s: candidate_cfl.map_or(start_cfl, |limit| start_cfl.min(limit)),
                accepted_start_cfl_limit_s: start_cfl,
                candidate_cfl_limit_s: candidate_cfl,
                depletion_limit_s: depletion_limit,
                storage_before_m3: self.accepted.ledger.storage_m3,
                storage_after_m3: balance.ledger.storage_m3,
                boundary_in_m3: balance.incoming,
                boundary_out_m3: balance.outgoing,
                lateral_in_m3: prepared.summary.incoming_m3,
                lateral_out_m3: prepared.summary.withdrawal_m3,
                lateral_assembly_abs_bound_m3: application.water_assembly_charge_m3,
                lateral_counter_addition_abs_bound_m3: prepared.summary.counter_addition_bound_m3,
                residual_m3: balance.residual,
                max_local_residual_m3: balance.max_local,
                max_local_momentum_residual_m4_per_s: balance.max_momentum,
                max_momentum_underflow_bound_m4_per_s: max_momentum_bound,
                face_volume_m3: Vec::new(),
                face_volume_low_m3: Vec::new(),
                storage_selection_bound_m3: Vec::new(),
                recovery_selection_bound_m2: Vec::new(),
                discharge_recovery_selection_bound_m3_per_s: Vec::new(),
                face_momentum_m4_per_s: Vec::new(),
                face_momentum_low_m4_per_s: Vec::new(),
                left_pressure_impulse_m4_per_s: Vec::new(),
                right_pressure_impulse_m4_per_s: Vec::new(),
                friction_impulse_m4_per_s: Vec::new(),
                friction_impulse_low_m4_per_s: Vec::new(),
                momentum_underflow_bound_m4_per_s: Vec::new(),
                face_transfer_underflow_bound_m3: Vec::new(),
                transfer_underflow_bound_m3: balance.transfer_charge,
                lateral_momentum_abs_bound_m4_per_s: Vec::new(),
                lateral_receipts: Vec::new(),
            };
            let last = LastInterval {
                previous_volume: self.accepted.volume.clone(),
                previous_momentum: self.accepted.momentum.clone(),
                previous_lateral: self.accepted.lateral.clone(),
                previous_ledger: self.accepted.ledger.clone(),
                previous_time: self.time(),
                previous_steps: self.accepted.steps,
                forcing: forcing.clone(),
                requested_end: requested_end_s,
                scalar,
                riemann: flux.riemann.clone(),
                friction_geometry,
                boundary_traces: flux.boundary_traces.clone(),
                face_volume: transport.face_volume,
                face_volume_low: transport.face_volume_low,
                face_momentum: transport.face_momentum,
                face_momentum_low: transport.face_momentum_low,
                storage_selection_bound: application.water_selection_bound,
                recovery_selection_bound: final_fields.area_selection_bound,
                discharge_recovery_selection_bound: final_fields.discharge_selection_bound,
                left_pressure: transport.left_pressure,
                right_pressure: transport.right_pressure,
                friction,
                friction_low,
                momentum_underflow_bound: momentum_bound,
                transfer_underflow_bound: transport.transfer_underflow_bound,
                lateral_receipt: prepared.receipt,
                lateral_momentum_bound: application.momentum_assembly_bound,
                lateral_water_assembly_bound: application.water_assembly_bound,
            };
            return Ok(ChannelStep {
                next: Self {
                    context: Arc::clone(context),
                    accepted: Accepted {
                        volume,
                        momentum,
                        lateral,
                        time: end,
                        ledger: balance.ledger,
                        steps,
                        last: Some(Arc::new(last)),
                    },
                },
            });
        }
        Err(ChannelError::Unresolved {
            dt_s: dt,
            retries: config.max_retries,
            reason,
        })
    }
}
