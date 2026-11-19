//! Certified trace states on the interior wave curve at a depth enclosure.

use super::{client::TraceClient, error::trace_error, interior::Interior};
use crate::channel::flow::{
    ChannelError,
    context::Context,
    surveyed::{
        PrismaticRiemannConfig,
        equation_ops::{gravity_depth_root, scalar},
        error::checked,
        work::WorkLedger,
    },
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, ShapeOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

/// Quadrature level and work ledger of one trace solve attempt.
pub(super) struct Evaluation<'a> {
    pub config: &'a PrismaticRiemannConfig,
    pub panels: usize,
    pub work: &'a WorkLedger,
}

/// A trace depth enclosure in metres, or in normalized depth `t = d / H`.
#[derive(Clone, Copy)]
pub(super) enum TraceDepth<'a, R: Runtime<DType = DType>> {
    Depth(&'a Interval<R>),
    Normalized(&'a Interval<R>),
}

/// Enclosures of the section and wave-curve state at every depth of one depth enclosure.
pub(super) struct TracePoint<R: Runtime<DType = DType>> {
    /// Depth `d`, in metres.
    pub depth: Interval<R>,
    /// Area `A(d)`, in m².
    pub area: Interval<R>,
    /// Top width `T(d)`, in metres.
    pub top_width: Interval<R>,
    /// Hydrostatic integral `I(A)`, in m³.
    pub integral: Interval<R>,
    /// Outward velocity `u_b = u_i - f_i(A)`, in m/s.
    pub velocity: Interval<R>,
    /// `c_b = sqrt(g A / T)`, in m/s.
    pub sound: Interval<R>,
    /// 1-shock magnitude `sqrt(A σ_p / A_i)` when a compression is possible, in m/s.
    ///
    /// Its lower bound is at least `c_i`: the secant of convex pressure exceeds `c_i²`.
    pub shock: Option<Interval<R>>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// The trace state over `depth`.
    ///
    /// `f_i` increases with area, so its values at the two area bounds enclose the curve.
    pub(super) fn surveyed_trace_point(
        &self,
        interior: &Interior<R>,
        depth: TraceDepth<'_, R>,
        evaluation: &Evaluation<'_>,
    ) -> Result<TracePoint<R>, ChannelError> {
        let sources = &interior.sources;
        let c = sources.geometry.client();
        let state = checked(c, "surveyed trace section state", |a| match depth {
            TraceDepth::Depth(depth) => sources.geometry.at_depth(a, &sources.bands, depth, None),
            TraceDepth::Normalized(t) => sources.geometry.at_normalized_depth(a, &sources.bands, t),
        })
        .map_err(trace_error)?;
        let targets = c.cat(&[state.area.lower(), state.area.upper()], 0)?;
        let rows = Tensor::<R>::zeros(&[2], DType::I64, c.device())?;
        let curve = sources
            .curve(
                &rows,
                &targets,
                evaluation.config,
                evaluation.panels,
                evaluation.work,
            )
            .map_err(trace_error)?;
        let compression = curve.compression.try_to_vec::<f64>()?;
        let gravity = self.definition.config.gravity;
        checked(c, "surveyed trace wave state", |a| {
            let curve_bounds = a.bounds(
                &curve.curve.lower().narrow(0, 0, 1)?,
                &curve.curve.upper().narrow(0, 1, 1)?,
            )?;
            let velocity = a.sub(&interior.velocity, &curve_bounds)?;
            let g = scalar(a, gravity)?;
            let sound = gravity_depth_root(a, &g, &state.hydraulic_depth, None)?;
            let shock = if compression[1] != 0.0 {
                let floor = if compression[0] != 0.0 {
                    c.maximum(
                        interior.sound.lower(),
                        &curve.shock.lower().narrow(0, 0, 1)?,
                    )?
                } else {
                    interior.sound.lower().clone()
                };
                let ceiling = c.maximum(&floor, &curve.shock.upper().narrow(0, 1, 1)?)?;
                Some(a.bounds(&floor, &ceiling)?)
            } else {
                None
            };
            Ok(TracePoint {
                depth: a.bounds(state.depth.lower(), state.depth.upper())?,
                area: a.bounds(state.area.lower(), state.area.upper())?,
                top_width: a.bounds(state.top_width.lower(), state.top_width.upper())?,
                integral: a.bounds(state.integral.lower(), state.integral.upper())?,
                velocity,
                sound,
                shock,
            })
        })
        .map_err(trace_error)
    }
}
