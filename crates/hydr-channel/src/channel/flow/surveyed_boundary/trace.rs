//! Surveyed scalar endpoint traces: quadrature refinement and packed lanes.

use super::{
    client::TraceClient,
    depth::RootProgress,
    error::trace_error,
    interior::Interior,
    point::{Evaluation, TraceDepth, TracePoint},
    receipt::Unresolved,
    request::TraceRequest,
};
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    boundary::{Lane, Reservoir, ScalarFrame},
    context::Context,
    geometry::SurveyedGeometry,
    surveyed::{PrismaticRiemannWork, error::checked, work::WorkLedger},
};
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps},
    runtime::{Runtime, SquareRootContract},
};
use std::sync::Arc;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// Certified lanes of every scalar endpoint, or `None` without one.
    ///
    /// Each lane's quadrature, root and callback work joins the frame's trace work.
    pub(in crate::channel::flow) fn surveyed_scalar_frame(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Option<ScalarFrame<R>>, ChannelError> {
        let mut lanes = Vec::with_capacity(2);
        let mut work: Option<PrismaticRiemannWork> = None;
        for (endpoint, policy) in [left, right].into_iter().enumerate() {
            if !policy.is_scalar() {
                continue;
            }
            if R::square_root_contract(DType::F64) != SquareRootContract::Ieee754RoundToNearest {
                return Err(ChannelError::input(
                    "scalar channel boundaries require correctly rounded F64 square roots: select an admitted runtime",
                ));
            }
            let (lane, lane_work) = self.surveyed_trace_lane(geometry, fields, endpoint, policy)?;
            lanes.push(lane);
            work = Some(work.map_or(lane_work, |total| total.joined(lane_work)));
        }
        if lanes.is_empty() {
            return Ok(None);
        }
        Ok(Some(ScalarFrame {
            traces: Arc::new(self.pack_lanes(&lanes)?),
            lanes,
            work,
        }))
    }

    /// One certified lane. Unresolved enclosures double the Simpson panels within the
    /// configured refinement, panel and callback ceilings, then reject.
    fn surveyed_trace_lane(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        endpoint: usize,
        policy: &ChannelTensorBoundary<R>,
    ) -> Result<(Lane<R>, PrismaticRiemannWork), ChannelError> {
        let interior = self.surveyed_interior(geometry, fields, endpoint)?;
        let request = self.surveyed_trace_request(&interior, policy)?;
        let config = &geometry.riemann;
        let work = WorkLedger::new(config);
        let mut progress = RootProgress::default();
        let mut panels = config.initial_simpson_panels;
        let mut refinement = 0;
        loop {
            work.begin_refinement(panels, refinement)
                .map_err(trace_error)?;
            let evaluation = Evaluation {
                config,
                panels,
                work: &work,
            };
            match self.surveyed_trace_attempt(
                geometry,
                &interior,
                &request,
                &evaluation,
                &mut progress,
            )? {
                Ok(lane) => return Ok((lane, work.snapshot())),
                Err(Unresolved(error)) => {
                    if refinement >= config.max_refinements
                        || panels > config.max_simpson_panels / 2
                    {
                        let work = work.snapshot();
                        return Err(match error {
                            ChannelError::Numerical { cell, context } => ChannelError::Numerical {
                                cell,
                                context: format!(
                                    "{context} Refinement stops at Simpson panels={panels}, refinements={refinement}, work={work:?}"
                                ),
                            },
                            error => error,
                        });
                    }
                    panels *= 2;
                    refinement += 1;
                }
            }
        }
    }

    fn surveyed_trace_attempt(
        &self,
        geometry: &SurveyedGeometry<R>,
        interior: &Interior<R>,
        request: &TraceRequest<R>,
        evaluation: &Evaluation<'_>,
        progress: &mut RootProgress,
    ) -> Result<Result<Lane<R>, Unresolved>, ChannelError> {
        let c = interior.sources.geometry.client();
        let rest = match request {
            TraceRequest::Reservoir { reservoir, .. } => self
                .surveyed_reservoir_rest(interior, reservoir, evaluation)?
                .map(|point| (point, reservoir.head.clone())),
            _ => None,
        };
        let (point, depth) = match (rest, request) {
            (Some(rest), _) => rest,
            (None, TraceRequest::Stage { depth, bounds }) => {
                let enclosure = checked(c, "surveyed stage depth", |a| {
                    Ok(a.bounds(&bounds.lo, &bounds.hi)?)
                })
                .map_err(trace_error)?;
                let point =
                    self.surveyed_trace_point(interior, TraceDepth::Depth(&enclosure), evaluation)?;
                (point, depth.clone())
            }
            (None, TraceRequest::Discharge { .. } | TraceRequest::Reservoir { .. }) => {
                let root = self.surveyed_trace_depth(interior, request, evaluation, progress)?;
                let (lower, upper) = (self.tensor(&[root.lower])?, self.tensor(&[root.upper])?);
                let t = checked(c, "surveyed trace root enclosure", |a| {
                    Ok(a.bounds(&lower, &upper)?)
                })
                .map_err(trace_error)?;
                let point =
                    self.surveyed_trace_point(interior, TraceDepth::Normalized(&t), evaluation)?;
                let depth = self
                    .client
                    .midpoint(point.depth.lower(), point.depth.upper())?;
                (point, depth)
            }
        };
        self.surveyed_trace_receipt(
            geometry,
            interior,
            request,
            &point,
            &depth,
            progress.iterations,
        )
    }

    /// The exact reservoir trace of an end cell at rest, or `None`.
    ///
    /// The end cell holds exactly zero discharge. The exact head depth lies inside the
    /// recovered interior depth enclosure. When the trace velocity at that depth is exactly
    /// zero, `E(d) = 0` holds there, so the head depth is the unique root. Any other state
    /// returns `None` and takes the certified root solve.
    fn surveyed_reservoir_rest(
        &self,
        interior: &Interior<R>,
        reservoir: &Reservoir<R>,
        evaluation: &Evaluation<'_>,
    ) -> Result<Option<TracePoint<R>>, ChannelError> {
        let c = &self.client;
        let head = &reservoir.head;
        let zero = self.constant(head, 0.0)?;
        let rest = self.all(&c.eq(&interior.discharge_point.high, &zero)?)?
            && self.all(&c.eq(&interior.discharge_point.low, &zero)?)?
            && self.all(&c.eq(&reservoir.bounds.lo, head)?)?
            && self.all(&c.eq(&reservoir.bounds.hi, head)?)?
            && self.all(&c.ge(head, interior.depth.lower())?)?
            && self.all(&c.le(head, interior.depth.upper())?)?;
        if !rest {
            return Ok(None);
        }
        let enclosure = checked(c, "surveyed reservoir rest depth", |a| {
            Ok(a.bounds(head, head)?)
        })
        .map_err(trace_error)?;
        let point =
            self.surveyed_trace_point(interior, TraceDepth::Depth(&enclosure), evaluation)?;
        let still = self.all(&c.eq(point.velocity.lower(), &zero)?)?
            && self.all(&c.eq(point.velocity.upper(), &zero)?)?;
        Ok(still.then_some(point))
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::reach;
    use crate::channel::flow::{
        ChannelBoundary, ChannelForcingInterval,
        surveyed_update::tests::{TestResult, cpu},
    };
    use numr::runtime::cpu::CpuRuntime;

    /// A three-band end cell at rest at depth 1.5 m against a reservoir head of 1.5 m.
    ///
    /// Frozen criteria, at both ends and for `k = 0` and `k = 1`:
    /// - trace water flux and outward velocity exactly zero
    /// - trace depth and both reservoir depth bounds exactly 1.5 m
    /// - end face volume exactly zero
    #[test]
    fn cpu_reservoir_at_rest_takes_the_exact_head_depth() -> TestResult {
        for endpoint in [0, 1] {
            for loss in [0.0, 1.0] {
                let reach = reach::<CpuRuntime>(cpu(), 5.125, 0.0, 0.0)?;
                let reservoir = ChannelBoundary::QuiescentReservoir {
                    elevation_m: 1.5,
                    entrance_loss_coefficient: loss,
                };
                let held = ChannelBoundary::Exterior {
                    depth_m: 1.5,
                    velocity_m_per_s: 0.0,
                };
                let (left, right) = if endpoint == 0 {
                    (reservoir, held)
                } else {
                    (held, reservoir)
                };
                let forcing = ChannelForcingInterval {
                    start_s: 0.0,
                    end_s: 0.01,
                    left,
                    right,
                    lateral_inputs: Vec::new(),
                };
                let next = reach.step(0.01, &forcing)?.next;
                let receipt = next.last_step()?.ok_or("Missing channel receipt")?;
                let trace = receipt.boundary_traces[endpoint]
                    .as_ref()
                    .ok_or("Missing reservoir trace")?;
                let bounds = trace
                    .reservoir
                    .as_ref()
                    .ok_or("Missing reservoir receipt")?;
                assert_eq!(trace.outward_discharge_m3_per_s, 0.0);
                assert_eq!(trace.outward_velocity_m_per_s, 0.0);
                assert_eq!(trace.depth_m.to_bits(), 1.5_f64.to_bits());
                assert_eq!(bounds.depth_lower_m.to_bits(), 1.5_f64.to_bits());
                assert_eq!(bounds.depth_upper_m.to_bits(), 1.5_f64.to_bits());
                assert_eq!(receipt.face_volume_m3[endpoint], 0.0);
            }
        }
        Ok(())
    }
}
