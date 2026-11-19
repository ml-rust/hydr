//! Interior end-cell state of a surveyed scalar endpoint, in the outward frame.

use super::{client::TraceClient, error::trace_error};
use crate::channel::flow::{
    ChannelError,
    admission::Fields,
    boundary::Bounds,
    conserved_point::ConservedPoint,
    context::Context,
    geometry::SurveyedGeometry,
    surveyed::{
        equation_ops::{gravity_depth_root, scalar},
        error::checked,
    },
    surveyed_points::trace_curve::TraceCurveSources,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, ScalarOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

/// The end cell's exact state, its recovered geometry and its certified wave quantities.
///
/// Velocity and discharge point out of the reach: `σ = -1` at the left end, `+1` at the right.
pub(super) struct Interior<R: Runtime<DType = DType>> {
    /// `0` left, `1` right.
    pub endpoint: usize,
    /// Wave-curve source of the end cell on its section.
    pub sources: TraceCurveSources<R>,
    /// Two-part area, in m².
    pub area_point: ConservedPoint<R>,
    /// Two-part outward discharge, in m³/s. Negation is exact on both parts.
    pub discharge_point: ConservedPoint<R>,
    /// Encloses the recovered depth `d_i`, in metres.
    pub depth: Interval<R>,
    /// Encloses the outward velocity `u_i`, in m/s.
    pub velocity: Interval<R>,
    /// Encloses `c_i = sqrt(g A_i / T_i)`, in m/s.
    pub sound: Interval<R>,
    /// Encloses the cap depth `H`, in metres.
    pub cap: Interval<R>,
}

/// The endpoints of `value` as boundary bounds.
pub(super) fn bounds<R: Runtime<DType = DType>>(value: &Interval<R>) -> Bounds<R> {
    Bounds {
        lo: value.lower().clone(),
        hi: value.upper().clone(),
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// The end cell state at `endpoint`. A dry or non-subcritical end cell rejects.
    pub(super) fn surveyed_interior(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        endpoint: usize,
    ) -> Result<Interior<R>, ChannelError> {
        let c = &self.client;
        let cell = if endpoint == 0 { 0 } else { self.count() - 1 };
        let sign = if endpoint == 0 { -1.0 } else { 1.0 };
        let part = |point: &ConservedPoint<R>, sign: f64| {
            let high = point.high.narrow(0, cell, 1)?;
            let low = point.low.narrow(0, cell, 1)?;
            Ok::<_, ChannelError>(if sign < 0.0 {
                ConservedPoint {
                    high: c.mul_scalar(&high, sign)?,
                    low: c.mul_scalar(&low, sign)?,
                }
            } else {
                ConservedPoint { high, low }
            })
        };
        let area_point = part(&fields.area_point, 1.0)?;
        let discharge_point = part(&fields.discharge_point, sign)?;
        let (area_expansion, area, discharge) = checked(c, "surveyed trace interior point", |a| {
            let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
            let area = a.expansion_bounds(&area_point.high, &area_point.low, &zero)?;
            let discharge =
                a.expansion_bounds(&discharge_point.high, &discharge_point.low, &zero)?;
            let bounds = a.expansion_interval(&area)?;
            let discharge = a.expansion_interval(&discharge)?;
            Ok((area, bounds, discharge))
        })
        .map_err(trace_error)?;
        let zero = self.constant(area.lower(), 0.0)?;
        self.boundary_require(&c.gt(area.lower(), &zero)?, "interior depth is dry")?;
        let sources = TraceCurveSources::new(&geometry.section, &area_expansion, &geometry.riemann)
            .map_err(trace_error)?;
        let gravity = self.definition.config.gravity;
        let (velocity, sound, depth, cap) = checked(c, "surveyed trace interior waves", |a| {
            let velocity = a.div_positive_interval(&discharge, &area)?;
            let g = scalar(a, gravity)?;
            let sound = gravity_depth_root(a, &g, sources.source.hydraulic_depth_m(), None)?;
            let depth = sources.source.depth_m();
            let depth = a.bounds(depth.lower(), depth.upper())?;
            let cap = &sources.bands.cap_depth;
            let cap = a.bounds(cap.lower(), cap.upper())?;
            Ok((velocity, sound, depth, cap))
        })
        .map_err(trace_error)?;
        self.boundary_require(&c.gt(depth.lower(), &zero)?, "interior depth is dry")?;
        for value in [velocity.lower(), velocity.upper(), sound.upper()] {
            self.finite(value, "surveyed scalar boundary interior wave")?;
        }
        self.boundary_characteristics(&bounds(&velocity), &bounds(&sound))?;
        Ok(Interior {
            endpoint,
            sources,
            area_point,
            discharge_point,
            depth,
            velocity,
            sound,
            cap,
        })
    }
}
