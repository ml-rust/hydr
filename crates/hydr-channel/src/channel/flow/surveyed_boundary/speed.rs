//! Unsigned signal-speed bound of a surveyed scalar endpoint face.

use super::{client::TraceClient, error::trace_error, interior::Interior, point::TracePoint};
use crate::channel::flow::{
    ChannelError,
    conserved_update::{exact, expanded},
    context::Context,
    geometry::SurveyedGeometry,
    surveyed::error::{Arithmetic, HydraulicError, checked},
    triangular::envelope::upper,
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ReduceOps, ShapeOps, UnaryOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// The upward-rounded maximum of the four certified wave terms at the end face.
    ///
    /// - `|u_i| + c_i` and `|u_b| + c_b` with `c = sqrt(g D_h)`.
    /// - The compression shock `s = u_i - sqrt(A_b σ_p / A_i)`, which must be negative.
    /// - The invariant region `u ± φ` of both states from `dry_front_bounds_tensor`.
    /// - The directed head `u_i - c_i` and tail `u_b - c_b`, which must be negative.
    pub(super) fn surveyed_trace_speed(
        &self,
        geometry: &SurveyedGeometry<R>,
        interior: &Interior<R>,
        point: &TracePoint<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let (head, tail, shock, characteristic) = checked(c, "surveyed trace wave speeds", |a| {
            let magnitude = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| {
                let size = c.maximum(&c.abs(value.lower())?, &c.abs(value.upper())?)?;
                Ok::<_, HydraulicError>(a.point(&size)?)
            };
            let head = a.sub(&interior.velocity, &interior.sound)?;
            let tail = a.sub(&point.velocity, &point.sound)?;
            let interior_size = magnitude(a, &interior.velocity)?;
            let interior_speed = a.add(&interior_size, &interior.sound)?;
            let trace_size = magnitude(a, &point.velocity)?;
            let trace_speed = a.add(&trace_size, &point.sound)?;
            let shock = match &point.shock {
                Some(shock) => Some(a.sub(&interior.velocity, shock)?),
                None => None,
            };
            let characteristic = c.maximum(interior_speed.upper(), trace_speed.upper())?;
            Ok((head, tail, shock, characteristic))
        })
        .map_err(trace_error)?;
        let zero = self.constant(&characteristic, 0.0)?;
        self.boundary_require(
            &c.lt(head.upper(), &zero)?,
            "compatible wave head lacks a negative directed speed",
        )?;
        self.boundary_require(
            &c.lt(tail.upper(), &zero)?,
            "compatible wave tail lacks a negative directed speed",
        )?;
        let mut speed = characteristic;
        if let Some(shock) = &shock {
            self.boundary_require(
                &c.lt(shock.upper(), &zero)?,
                "compatible compression wave lacks a negative directed speed",
            )?;
            speed = c.maximum(&speed, &c.abs(shock.lower())?)?;
        }
        let (area, discharge) = expanded(c, "surveyed trace invariant states", |a| {
            let interior_area = exact(a, &interior.area_point.high, &interior.area_point.low)?;
            let interior_discharge = exact(
                a,
                &interior.discharge_point.high,
                &interior.discharge_point.low,
            )?;
            let water = a.mul(&point.area, &point.velocity)?;
            let trace_area = a.expansion_from_interval(&point.area)?;
            let trace_discharge = a.expansion_from_interval(&water)?;
            let join = |a: &mut Arithmetic<'_, R>,
                        first: &CertifiedExpansion<R>,
                        second: &CertifiedExpansion<R>| {
                a.expansion_bounds(
                    &c.cat(&[first.high(), second.high()], 0)?,
                    &c.cat(&[first.low(), second.low()], 0)?,
                    &c.cat(&[first.abs_error(), second.abs_error()], 0)?,
                )
            };
            Ok((
                join(a, &interior_area, &trace_area)?,
                join(a, &interior_discharge, &trace_discharge)?,
            ))
        })?;
        let owners = Tensor::<R>::zeros(&[2], DType::I64, c.device())?;
        let (lower, upper_front) = geometry
            .section
            .dry_front_bounds_tensor(&owners, &area, &discharge, &geometry.riemann)
            .map_err(|error| {
                ChannelError::numerical(format!(
                    "scalar channel boundary: surveyed trace invariant region: {error}. Supply a resolved wet subcritical target or rescale inputs"
                ))
            })?;
        let invariant = c.maximum(&c.abs(&lower)?, &c.abs(&upper_front)?)?;
        let invariant = c.max(&invariant, &[0], false)?.reshape(&[1])?;
        speed = c.maximum(&speed, &invariant)?;
        let speed = upper::<R>(c, &speed)?;
        self.finite(&speed, "surveyed scalar boundary signal speed")?;
        Ok(speed)
    }
}
