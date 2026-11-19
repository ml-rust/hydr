//! Surveyed scalar endpoint traces: quadrature refinement and packed lanes.

use super::{
    client::TraceClient,
    depth::RootProgress,
    error::trace_error,
    interior::Interior,
    point::{Evaluation, TraceDepth},
    receipt::Unresolved,
    request::TraceRequest,
};
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    boundary::{Lane, ScalarFrame},
    context::Context,
    geometry::SurveyedGeometry,
    surveyed::{PrismaticRiemannWork, error::checked, work::WorkLedger},
};
use numr::{
    dtype::DType,
    ops::MidpointOps,
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
        let (point, depth) = match request {
            TraceRequest::Stage { depth, bounds } => {
                let enclosure = checked(c, "surveyed stage depth", |a| {
                    Ok(a.bounds(&bounds.lo, &bounds.hi)?)
                })
                .map_err(trace_error)?;
                let point =
                    self.surveyed_trace_point(interior, TraceDepth::Depth(&enclosure), evaluation)?;
                (point, depth.clone())
            }
            TraceRequest::Discharge { .. } | TraceRequest::Reservoir { .. } => {
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
}
