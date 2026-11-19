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
    boundary::{Lane, Reservoir, ScalarFrame, TraceLane},
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

/// Closing text of a trace rejection that no quadrature level within the ceilings resolves.
///
/// The coordinator reads it to tell a precision floor from a rejection a shorter interval fixes.
pub(in crate::channel::flow) const EXHAUSTED: &str =
    "the enclosure cannot meet its control within the quadrature ceilings";

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
            // A kept lane of the same fields and end policy holds this solve and its work.
            let key = self.lane_key(fields, endpoint, policy)?;
            let solved = match self.lane_memo.find(&key, |_| true) {
                Some(solved) => solved,
                None => {
                    let (lane, work) =
                        self.surveyed_trace_lane(geometry, fields, endpoint, policy)?;
                    self.lane_memo.keep(key, TraceLane { lane, work })
                }
            };
            lanes.push(solved.lane.clone());
            let lane_work = solved.work;
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
        let mut previous: Option<(usize, f64)> = None;
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
                Err(Unresolved { error, width }) => {
                    // A width that one doubling fails to halve sits at its rounding floor.
                    // Finer quadrature cannot meet the allowance, so the trace rejects now.
                    let stalled = matches!(
                        (previous, width),
                        (Some((last, before)), Some((check, now))) if last == check && now > 0.5 * before
                    );
                    let ceiling = refinement >= config.max_refinements
                        || panels > config.max_simpson_panels / 2;
                    if stalled || ceiling {
                        let work = work.snapshot();
                        let stop = if stalled {
                            "Refinement no longer narrows the enclosure"
                        } else {
                            "Refinement stops"
                        };
                        return Err(match error {
                            ChannelError::Numerical { cell, context } => ChannelError::Numerical {
                                cell,
                                context: format!(
                                    "{context} {stop} at Simpson panels={panels}, refinements={refinement}, work={work:?}: {EXHAUSTED}"
                                ),
                            },
                            error => error,
                        });
                    }
                    previous = width;
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
            TraceRequest::Reservoir { reservoir, cap } => {
                match self.surveyed_reservoir_rest(interior, reservoir, evaluation)? {
                    Some(point) => Some(point),
                    None if !*cap => {
                        self.surveyed_reservoir_leaving(interior, reservoir, evaluation)?
                    }
                    None => None,
                }
                .map(|point| (point, reservoir.head.clone()))
            }
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
                let root =
                    match self.surveyed_trace_depth(interior, request, evaluation, progress)? {
                        Ok(root) => root,
                        Err(error) => return Ok(Err(Unresolved::refinable(error))),
                    };
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

    /// The reservoir trace at the head depth when flow certainly leaves there, or `None`.
    ///
    /// Leaving flow takes `z_min + d = η_r`: its velocity head mixes into the reservoir, so
    /// `E(d) = d - (η - z_min)` wherever `u_b > 0`. The head-depth enclosure holds the exact
    /// head depth. When the trace velocity over that enclosure has a positive lower bound,
    /// `E` vanishes at the exact head depth, and `dE/dd >= 1` makes it the unique root. A
    /// head above the cap, or any other state, returns `None` and takes the certified root
    /// solve.
    fn surveyed_reservoir_leaving(
        &self,
        interior: &Interior<R>,
        reservoir: &Reservoir<R>,
        evaluation: &Evaluation<'_>,
    ) -> Result<Option<TracePoint<R>>, ChannelError> {
        let c = &self.client;
        if !self.all(&c.le(&reservoir.bounds.hi, interior.cap.lower())?)? {
            return Ok(None);
        }
        let enclosure = checked(c, "surveyed reservoir leaving depth", |a| {
            Ok(a.bounds(&reservoir.bounds.lo, &reservoir.bounds.hi)?)
        })
        .map_err(trace_error)?;
        let point =
            self.surveyed_trace_point(interior, TraceDepth::Depth(&enclosure), evaluation)?;
        let zero = self.constant(&reservoir.head, 0.0)?;
        let leaving = self.all(&c.gt(point.velocity.lower(), &zero)?)?;
        Ok(leaving.then_some(point))
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

    /// A three-band end cell at depth 2.47 m drains toward a reservoir head of 2 m.
    ///
    /// Frozen criteria, at both entrance losses:
    /// - the flow leaves the reach
    /// - trace depth and both reservoir depth bounds exactly 2 m: `z_min + d = η_r`
    /// - no root iteration runs
    #[test]
    fn cpu_leaving_reservoir_takes_the_exact_head_depth() -> TestResult {
        for loss in [0.0, 0.5] {
            let reach = reach::<CpuRuntime>(cpu(), 10.375, 1.0, 0.0)?;
            let forcing = ChannelForcingInterval {
                start_s: 0.0,
                end_s: 1.0 / 1048576.0,
                left: ChannelBoundary::Wall,
                right: ChannelBoundary::QuiescentReservoir {
                    elevation_m: 2.0,
                    entrance_loss_coefficient: loss,
                },
                lateral_inputs: Vec::new(),
            };
            let next = reach.step(forcing.end_s, &forcing)?.next;
            let receipt = next.last_step()?.ok_or("Missing channel receipt")?;
            let trace = receipt.boundary_traces[1]
                .as_ref()
                .ok_or("Missing reservoir trace")?;
            let bounds = trace
                .reservoir
                .as_ref()
                .ok_or("Missing reservoir receipt")?;
            assert!(trace.outward_discharge_m3_per_s > 0.0, "{trace:?}");
            assert_eq!(trace.depth_m.to_bits(), 2.0_f64.to_bits());
            assert_eq!(bounds.depth_lower_m.to_bits(), 2.0_f64.to_bits());
            assert_eq!(bounds.depth_upper_m.to_bits(), 2.0_f64.to_bits());
            assert_eq!(trace.root_iterations, 0);
        }
        Ok(())
    }

    /// Both ends hold scalar policies. Only the right reservoir head moves.
    ///
    /// Frozen criteria:
    /// - the second frame keeps the left lane of the first frame
    /// - every lane bit and the frame work match a fresh reach
    #[test]
    fn cpu_kept_lane_matches_a_fresh_reach() -> TestResult {
        use super::super::fixture::{geometry, policy, reach_cells};
        let left = ChannelBoundary::Discharge {
            discharge_m3_per_s: 0.75,
        };
        let head = |elevation_m| ChannelBoundary::QuiescentReservoir {
            elevation_m,
            entrance_loss_coefficient: 0.5,
        };
        let frame = |reach: &crate::channel::flow::ChannelReach<CpuRuntime>,
                     right|
         -> TestResult<super::ScalarFrame<CpuRuntime>> {
            let context = &reach.context;
            let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
            let left = policy(reach, &left)?;
            let right = policy(reach, &right)?;
            Ok(context
                .surveyed_scalar_frame(geometry(reach)?, &fields, &left, &right)?
                .ok_or("a scalar frame is required")?)
        };
        let bits = |frame: &super::ScalarFrame<CpuRuntime>| -> TestResult<Vec<u64>> {
            let mut out = Vec::new();
            for lane in &frame.lanes {
                let trace = &lane.trace;
                for value in [
                    &trace.depth_m,
                    &trace.area_m2,
                    &trace.outward_velocity_m_per_s,
                    &trace.outward_discharge_m3_per_s,
                    &trace.momentum_flux_m4_per_s2,
                    &trace.signal_speed_bound_m_per_s,
                    &lane.longitudinal,
                ] {
                    out.extend(value.try_to_vec::<f64>()?.iter().map(|v| v.to_bits()));
                }
            }
            Ok(out)
        };
        let kept = reach_cells::<CpuRuntime>(cpu(), 3, 5.125, 0.5, 0.0)?;
        let first = frame(&kept, head(1.7))?;
        let second = frame(&kept, head(1.75))?;
        assert!(
            second.lanes[0]
                .trace
                .depth_m
                .shares_storage_with(&first.lanes[0].trace.depth_m)
        );
        assert!(
            !second.lanes[1]
                .trace
                .depth_m
                .shares_storage_with(&first.lanes[1].trace.depth_m)
        );
        let fresh = reach_cells::<CpuRuntime>(cpu(), 3, 5.125, 0.5, 0.0)?;
        let fresh = frame(&fresh, head(1.75))?;
        assert_eq!(bits(&second)?, bits(&fresh)?);
        assert_eq!(second.work, fresh.work);
        Ok(())
    }
}
