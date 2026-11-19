//! Physical fluxes, budgets, compatibility bounds and the receipt of one surveyed trace.

use super::{
    client::TraceClient,
    error::trace_error,
    interior::{Interior, bounds},
    point::TracePoint,
    request::TraceRequest,
};
use crate::channel::flow::{
    ChannelError,
    boundary::{Lane, NativeBoundaryTraces, boundary_rejection},
    context::Context,
    geometry::SurveyedGeometry,
    surveyed::{
        equation_ops::scalar,
        error::{Arithmetic, HydraulicResult, checked},
    },
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, MidpointOps, ScalarOps, UnaryOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

/// A rejection that finer quadrature can resolve: an enclosure exceeds its allowance.
pub(super) struct Unresolved(pub ChannelError);

/// Nominal centers and certified enclosures of the trace fluxes.
struct Fluxes<R: Runtime<DType = DType>> {
    area: Tensor<R>,
    velocity: Tensor<R>,
    /// Outward discharge: the enclosure center, or `σ Q` for `Discharge`.
    discharge: Tensor<R>,
    momentum: Tensor<R>,
    velocity_defect: Tensor<R>,
    water_defect: Tensor<R>,
    momentum_defect: Tensor<R>,
    water_width: Tensor<R>,
    momentum_width: Tensor<R>,
    water: Interval<R>,
    flux: Interval<R>,
}

/// Upper bound of the larger distance from `center` to the endpoints of `value`.
fn reach<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    center: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TraceClient<R>,
{
    let c = a.client();
    let center = a.point(center)?;
    let (upper, lower) = (a.point(value.upper())?, a.point(value.lower())?);
    let above = a.sub(&upper, &center)?;
    let below = a.sub(&center, &lower)?;
    Ok(c.maximum(above.upper(), below.upper())?)
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// The receipt and direct rates of the trace `point` with depth nominal `depth`.
    ///
    /// Water takes the center of `A_b u_b`, or `σ Q` with its original bits for `Discharge`.
    /// Momentum takes the center of `A_b u_b² + g I(A_b)`. A trace outside the wet subcritical
    /// branch rejects. An enclosure wider than its allowance returns `Unresolved`.
    pub(super) fn surveyed_trace_receipt(
        &self,
        geometry: &SurveyedGeometry<R>,
        interior: &Interior<R>,
        request: &TraceRequest<R>,
        point: &TracePoint<R>,
        depth: &Tensor<R>,
        iterations: usize,
    ) -> Result<Result<Lane<R>, Unresolved>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(depth, 0.0)?;
        self.boundary_require(&c.gt(point.depth.lower(), &zero)?, "trace depth is dry")?;
        if let Err(error) = self.surveyed_trace_regime(point)? {
            return Ok(Err(error));
        }
        let target = match request {
            TraceRequest::Discharge { outward, .. } => Some(outward),
            _ => None,
        };
        let fluxes = self.surveyed_trace_fluxes(point, target)?;
        if let Err(error) = self.surveyed_trace_allowances(interior, point, &fluxes)? {
            return Ok(Err(error));
        }
        let speed = self.surveyed_trace_speed(geometry, interior, point)?;
        let longitudinal = match request {
            TraceRequest::Discharge { longitudinal, .. } => longitudinal.clone(),
            _ if interior.endpoint == 0 => c.neg(&fluxes.discharge)?,
            _ => fluxes.discharge.clone(),
        };
        for value in [
            &fluxes.area,
            &fluxes.velocity,
            &fluxes.discharge,
            &fluxes.momentum,
            &longitudinal,
        ] {
            self.finite(value, "surveyed scalar boundary physical trace")?;
        }
        let mut lane = Lane {
            longitudinal,
            water_bound: zero.clone(),
            momentum_bound: zero.clone(),
            trace: NativeBoundaryTraces {
                depth_m: depth.clone(),
                area_m2: fluxes.area,
                outward_velocity_m_per_s: fluxes.velocity,
                outward_discharge_m3_per_s: fluxes.discharge,
                momentum_flux_m4_per_s2: fluxes.momentum,
                signal_speed_bound_m_per_s: speed,
                velocity_compatibility_abs_bound_m_per_s: fluxes.velocity_defect,
                water_compatibility_abs_bound_m3_per_s: fluxes.water_defect,
                momentum_compatibility_abs_bound_m4_per_s2: fluxes.momentum_defect,
                water_flux_width_m3_per_s: fluxes.water_width,
                momentum_flux_width_m4_per_s2: fluxes.momentum_width,
                cap_compatibility_approximation: zero,
                reservoir: None,
                endpoints: vec![interior.endpoint],
                root_iterations: vec![iterations],
            },
        };
        if let TraceRequest::Reservoir { reservoir, .. } = request {
            let enclosure = bounds(&point.depth);
            if let Err(error) = self.boundary_reservoir_receipt(&mut lane, reservoir, &enclosure) {
                return match error {
                    ChannelError::Numerical { .. } => Ok(Err(Unresolved(error))),
                    error => Err(error),
                };
            }
        }
        Ok(Ok(lane))
    }

    /// Rejects a certainly critical or supercritical trace. An unresolved margin returns `Unresolved`.
    fn surveyed_trace_regime(
        &self,
        point: &TracePoint<R>,
    ) -> Result<Result<(), Unresolved>, ChannelError> {
        let c = &self.client;
        let (velocity, sound) = (bounds(&point.velocity), bounds(&point.sound));
        let Err(error) = self.boundary_characteristics(&velocity, &sound) else {
            return Ok(Ok(()));
        };
        // A margin that every point of the enclosures violates rejects at any quadrature level.
        let scale = c.maximum(
            &c.maximum(&c.abs(&velocity.lo)?, &c.abs(&velocity.hi)?)?,
            &sound.hi,
        )?;
        let margin = c.mul_scalar(&scale, self.definition.config.boundary.critical_relative)?;
        let incoming = c.ge(&c.sub(&velocity.lo, &sound.hi)?, &c.neg(&margin)?)?;
        let outgoing = c.le(&c.add(&velocity.hi, &sound.hi)?, &margin)?;
        if self.all(&c.maximum(&incoming, &outgoing)?)? {
            return Err(error);
        }
        Ok(Err(Unresolved(error)))
    }

    fn surveyed_trace_fluxes(
        &self,
        point: &TracePoint<R>,
        target: Option<&Tensor<R>>,
    ) -> Result<Fluxes<R>, ChannelError> {
        let c = &self.client;
        let gravity = self.definition.config.gravity;
        checked(c, "surveyed trace physical fluxes", |a| {
            let water = a.mul(&point.area, &point.velocity)?;
            let (lo, hi) = (point.velocity.lower(), point.velocity.upper());
            let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
            let magnitude = a.bounds(
                &c.maximum(&c.maximum(lo, &c.neg(hi)?)?, &zero)?,
                &c.maximum(&c.abs(lo)?, &c.abs(hi)?)?,
            )?;
            let squared = a.mul(&magnitude, &magnitude)?;
            let advective = a.mul(&point.area, &squared)?;
            let g = scalar(a, gravity)?;
            let pressure = a.mul(&g, &point.integral)?;
            let flux = a.add(&advective, &pressure)?;
            let center = |value: &Interval<R>| c.midpoint(value.lower(), value.upper());
            let area = center(&point.area)?;
            let velocity = center(&point.velocity)?;
            let momentum = center(&flux)?;
            let discharge = match target {
                Some(target) => target.clone(),
                None => center(&water)?,
            };
            let width = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| {
                let (upper, lower) = (a.point(value.upper())?, a.point(value.lower())?);
                let span = a.sub(&upper, &lower)?;
                HydraulicResult::Ok(span.upper().clone())
            };
            let water_width = match target {
                Some(_) => zero.clone(),
                None => width(a, &water)?,
            };
            let momentum_width = width(a, &flux)?;
            Ok(Fluxes {
                velocity_defect: reach(a, &point.velocity, &velocity)?,
                water_defect: reach(a, &water, &discharge)?,
                momentum_defect: reach(a, &flux, &momentum)?,
                area,
                velocity,
                discharge,
                momentum,
                water_width,
                momentum_width,
                water,
                flux,
            })
        })
        .map_err(trace_error)
    }

    /// Budget and compatibility checks. A failure returns `Unresolved`.
    fn surveyed_trace_allowances(
        &self,
        interior: &Interior<R>,
        point: &TracePoint<R>,
        fluxes: &Fluxes<R>,
    ) -> Result<Result<(), Unresolved>, ChannelError> {
        let c = &self.client;
        let flux = &self.definition.config;
        let boundary = &flux.boundary;
        let size = |value: &Interval<R>| -> Result<Tensor<R>, ChannelError> {
            Ok(c.maximum(&c.abs(value.lower())?, &c.abs(value.upper())?)?)
        };
        let scale = c.maximum(
            &c.maximum(&size(&interior.velocity)?, &size(&point.velocity)?)?,
            &c.maximum(interior.sound.upper(), point.sound.upper())?,
        )?;
        let water_scale = c.maximum(&c.abs(&fluxes.discharge)?, &size(&fluxes.water)?)?;
        let momentum_scale = size(&fluxes.flux)?;
        let checks = [
            (
                c.le(
                    &fluxes.water_width,
                    &c.add_scalar(
                        &c.mul_scalar(&c.abs(&fluxes.discharge)?, flux.riemann_flux_relative)?,
                        flux.riemann_water_flux_absolute_m3_per_s,
                    )?,
                )?,
                "physical water flux width exceeds its controls",
            ),
            (
                c.le(
                    &fluxes.momentum_width,
                    &c.add_scalar(
                        &c.mul_scalar(&c.abs(&fluxes.momentum)?, flux.riemann_flux_relative)?,
                        flux.riemann_momentum_flux_absolute_m4_per_s2,
                    )?,
                )?,
                "physical momentum flux width exceeds its controls",
            ),
            (
                c.le(
                    &fluxes.velocity_defect,
                    &c.mul_scalar(&scale, boundary.velocity_relative)?,
                )?,
                "velocity compatibility exceeds its dedicated control",
            ),
            (
                c.le(
                    &fluxes.water_defect,
                    &c.add_scalar(
                        &c.mul_scalar(&water_scale, boundary.discharge_relative)?,
                        boundary.discharge_absolute_m3_per_s,
                    )?,
                )?,
                "water compatibility exceeds its dedicated controls",
            ),
            (
                c.le(
                    &fluxes.momentum_defect,
                    &c.add_scalar(
                        &c.mul_scalar(&momentum_scale, boundary.momentum_relative)?,
                        boundary.momentum_absolute_m4_per_s2,
                    )?,
                )?,
                "momentum compatibility exceeds its dedicated controls",
            ),
        ];
        let values = [
            &fluxes.water_width,
            &fluxes.momentum_width,
            &fluxes.velocity_defect,
            &fluxes.water_defect,
            &fluxes.momentum_defect,
        ];
        for ((mask, message), value) in checks.into_iter().zip(values) {
            if !self.all(&mask)? {
                let value = value.item::<f64>()?;
                let message = format!("{message}, bound={value:e}");
                return Ok(Err(Unresolved(boundary_rejection(&message))));
            }
        }
        Ok(Ok(()))
    }
}
