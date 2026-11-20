//! Certified normalized-depth enclosures of `Discharge` and reservoir traces across quadrature levels.

use super::{
    bracket::Bracket,
    client::TraceClient,
    contract::SlopeFloor,
    interior::Interior,
    point::Evaluation,
    request::TraceRequest,
    roots::{Residual, RootEnclosure, RootOutcome},
};
use crate::channel::flow::{ChannelError, boundary::boundary_rejection, context::Context};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};

/// Certified facts of one trace root kept across quadrature levels.
///
/// Every certificate holds for the exact trace equation, so a finer level narrows the
/// kept enclosure instead of solving the whole bracket again.
#[derive(Default)]
pub(super) struct RootProgress {
    bracket: Option<Bracket>,
    enclosure: Option<RootEnclosure>,
    /// Largest midpoint round count among the lane's root solves.
    pub iterations: usize,
    /// Exact outgoing cap equality of a reservoir head equal to the cap depth.
    cap: Option<bool>,
}

impl RootProgress {
    fn record(&mut self, root: RootEnclosure) -> RootEnclosure {
        self.iterations = self.iterations.max(root.iterations);
        self.enclosure = Some(root);
        root
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// Certified normalized-depth enclosure of a `Discharge` or reservoir trace.
    ///
    /// `Discharge` solves `-G = σ Q - A u_b` over `[t_L, 1]` with `κ_Q`. A reservoir solves
    /// `E` over `[0, 1]` with `κ_E = H`. A kept enclosure narrows first; a failed narrowing
    /// solves the whole bracket. The inner error is a `Discharge` bracket rejection that
    /// finer quadrature can resolve.
    pub(super) fn surveyed_trace_depth(
        &self,
        interior: &Interior<R>,
        request: &TraceRequest<R>,
        evaluation: &Evaluation<'_>,
        progress: &mut RootProgress,
    ) -> Result<Result<RootEnclosure, ChannelError>, ChannelError> {
        if let TraceRequest::Reservoir { cap: true, .. } = request
            && progress.cap.is_none()
        {
            let point = self.surveyed_trace_at(interior, &self.tensor(&[1.0])?, evaluation)?;
            progress.cap = Some(point.velocity.lower().item::<f64>()? >= 0.0);
        }
        if progress.cap == Some(true) {
            return Ok(Ok(RootEnclosure {
                lower: 1.0,
                upper: 1.0,
                iterations: 0,
            }));
        }
        let residual: Box<Residual<'_, R>> = match request {
            TraceRequest::Discharge { outward, .. } => Box::new(move |t: &Tensor<R>| {
                let point = self.surveyed_trace_at(interior, t, evaluation)?;
                self.surveyed_discharge_residual(interior, &point, outward)
            }),
            TraceRequest::Reservoir { reservoir, .. } => Box::new(move |t: &Tensor<R>| {
                let point = self.surveyed_trace_at(interior, t, evaluation)?;
                self.surveyed_reservoir_residual(interior, &point, reservoir)
            }),
            TraceRequest::Stage { .. } => {
                return Err(ChannelError::numerical(
                    "surveyed stage trace has a known depth: inspect trace dispatch",
                ));
            }
        };
        let bracket = match progress.bracket {
            Some(bracket) => bracket,
            None => {
                let bracket = match request {
                    TraceRequest::Discharge { outward, .. } => match self
                        .surveyed_discharge_bracket(
                            interior,
                            outward,
                            evaluation,
                            &mut progress.iterations,
                        )? {
                        Ok(bracket) => bracket,
                        Err(error) => return Ok(Err(error)),
                    },
                    _ => Bracket {
                        lower: 0.0,
                        kappa: interior.cap.lower().item::<f64>()?,
                    },
                };
                progress.bracket = Some(bracket);
                bracket
            }
        };
        let discharge = matches!(request, TraceRequest::Discharge { .. });
        // The certified slope lower bound `bracket.kappa` holds at every certified depth above
        // d_s. At the lower end of an enclosure the local slope is far larger than at t_L, so
        // `root ± R / κ` contracts to the residual width instead of stopping at the first
        // ambiguous midpoint.
        let local = |lower: f64| -> Result<f64, ChannelError> {
            if !discharge {
                return Ok(bracket.kappa);
            }
            let point = self.surveyed_trace_at(interior, &self.tensor(&[lower])?, evaluation)?;
            Ok(match self.surveyed_discharge_slope(interior, &point)? {
                Some(local) => bracket.kappa.max(local),
                None => bracket.kappa,
            })
        };
        // Reservoir samples land on one side of the root, so the bracket contracts from both
        // ends first. A `Discharge` enclosure already contracts through its local floor.
        let slope: Option<Box<SlopeFloor<'_>>> = match request {
            TraceRequest::Reservoir { reservoir, .. } => Some(Box::new(move |a: f64, b: f64| {
                self.surveyed_reservoir_slope(interior, reservoir, evaluation, a, b, bracket.kappa)
            })),
            _ => None,
        };
        let contract = |lower: f64, upper: f64| match &slope {
            Some(slope) => {
                self.surveyed_trace_contract(evaluation, lower, upper, &*residual, &**slope)
            }
            None => Ok((lower, upper)),
        };
        let narrow = |root: RootEnclosure| -> Result<Option<RootEnclosure>, ChannelError> {
            let (lower, upper) = contract(root.lower, root.upper)?;
            let kappa = local(lower)?;
            Ok(
                match self.surveyed_trace_root(evaluation, lower, upper, kappa, &*residual)? {
                    RootOutcome::Root(root) => Some(root),
                    RootOutcome::NoRoot | RootOutcome::Unresolved => None,
                },
            )
        };
        if let Some(kept) = progress.enclosure
            && let Some(root) = narrow(kept)?
        {
            return Ok(Ok(progress.record(root)));
        }
        let (lower, upper) = contract(bracket.lower, 1.0)?;
        let root = match self.surveyed_trace_root(
            evaluation,
            lower,
            upper,
            bracket.kappa,
            &*residual,
        )? {
            RootOutcome::Root(root) => root,
            // The certified lower residual sign excludes absence there, so the root lies above H.
            RootOutcome::NoRoot => {
                return Err(boundary_rejection("trace depth exceeds the geometric cap"));
            }
            RootOutcome::Unresolved => {
                return Err(boundary_rejection(
                    "trace upper bracket lacks a directed residual sign or cap membership is ambiguous",
                ));
            }
        };
        progress.record(root);
        let root = if discharge {
            narrow(root)?.unwrap_or(root)
        } else {
            root
        };
        Ok(Ok(progress.record(root)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{
        CAP, PANELS, ends, geometry, interior, ledger, policy, reach, state,
    };
    use super::*;
    use crate::channel::flow::{
        ChannelBoundary, ChannelReach,
        surveyed_update::tests::{Client, TestResult, cpu},
    };
    use numr::runtime::cpu::CpuRuntime;

    /// One frozen three-band row: interior area and global discharge, end, policy, and the
    /// directed bits of its 150-digit trace depth.
    struct Row {
        area: f64,
        discharge: f64,
        endpoint: usize,
        policy: ChannelBoundary,
        depth: [u64; 2],
    }

    fn rows() -> [Row; 3] {
        [
            // Leaving above knot 2, root below the interior depth.
            Row {
                area: 10.375,
                discharge: 1.0,
                endpoint: 1,
                policy: ChannelBoundary::Discharge {
                    discharge_m3_per_s: 10.184547080871404,
                },
                depth: [0x4000000040000000, 0x4000000040000001],
            },
            // Entering through a shock above the interior depth.
            Row {
                area: 5.125,
                discharge: 0.5,
                endpoint: 0,
                policy: ChannelBoundary::Discharge {
                    discharge_m3_per_s: 5.06285120531442,
                },
                depth: [0x3ffc000000000000, 0x3ffc000000000001],
            },
            // Entering from a held reservoir head.
            Row {
                area: 5.125,
                discharge: 0.5,
                endpoint: 0,
                policy: ChannelBoundary::QuiescentReservoir {
                    elevation_m: 1.7831130485978541,
                    entrance_loss_coefficient: 0.0,
                },
                depth: [0x3ffc000000000000, 0x3ffc000000000001],
            },
        ]
    }

    /// The certified depth enclosure of `row` at the fixture panel count.
    fn enclosure<R>(reach: &ChannelReach<R>, row: &Row) -> TestResult<[f64; 2]>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let context = &reach.context;
        let interior = interior(reach, row.endpoint)?;
        let request = context.surveyed_trace_request(&interior, &policy(reach, &row.policy)?)?;
        let work = ledger(reach)?;
        let evaluation = Evaluation {
            config: &geometry(reach)?.riemann,
            panels: PANELS,
            work: &work,
        };
        let mut progress = RootProgress::default();
        let root =
            context.surveyed_trace_depth(&interior, &request, &evaluation, &mut progress)??;
        let point = state(reach, &interior, &evaluation, root.lower, root.upper)?;
        ends(&point.depth)
    }

    /// Each certified depth enclosure holds its 150-digit reference.
    fn roots_enclose_the_references<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        for (index, row) in rows().iter().enumerate() {
            let reach = reach::<R>(client.clone(), row.area, row.discharge, 0.0)?;
            let [lower, upper] = enclosure(&reach, row)?;
            let [floor, ceil] = row.depth.map(f64::from_bits);
            assert!(
                lower <= floor && ceil <= upper,
                "row {index}: [{lower:e}, {upper:e}] excludes [{floor:e}, {ceil:e}]"
            );
            assert!(
                upper - lower <= 1e-13 * CAP,
                "row {index}: width={:e}",
                upper - lower
            );
        }
        Ok(())
    }

    /// With the bracket's sonic-side slope floor, `root ± R / κ` stops at the first ambiguous
    /// midpoint. The floor at the enclosure's lower end contracts it to the residual width.
    ///
    /// Midpoint probes isolate the slope floor. A secant probe reaches the residual width
    /// without the local floor.
    fn local_slope_floor_contracts_the_enclosure<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let row = &rows()[0];
        let reach = reach::<R>(client, row.area, row.discharge, 0.0)?;
        let context = &reach.context;
        let interior = interior(&reach, row.endpoint)?;
        let work = ledger(&reach)?;
        let mut config = geometry(&reach)?.riemann.clone();
        config.root.probe = solvr::optimize::CertifiedRootProbe::Midpoint;
        let evaluation = Evaluation {
            config: &config,
            panels: PANELS,
            work: &work,
        };
        let target = context.tensor(&[10.184547080871404])?;
        let mut iterations = 0;
        let bracket = context.surveyed_discharge_bracket(
            &interior,
            &target,
            &evaluation,
            &mut iterations,
        )??;
        let residual = |t: &Tensor<R>| {
            let point = context.surveyed_trace_at(&interior, t, &evaluation)?;
            context.surveyed_discharge_residual(&interior, &point, &target)
        };
        let RootOutcome::Root(coarse) = context.surveyed_trace_root(
            &evaluation,
            bracket.lower,
            1.0,
            bracket.kappa,
            &residual,
        )?
        else {
            return Err("the discharge root is not established".into());
        };
        let point = state(&reach, &interior, &evaluation, coarse.lower, coarse.lower)?;
        let local = context
            .surveyed_discharge_slope(&interior, &point)?
            .ok_or("no slope floor above the sonic depth")?;
        assert!(
            local > 1e4 * bracket.kappa,
            "local={local:e}, bracket={:e}",
            bracket.kappa
        );
        let RootOutcome::Root(fine) = context.surveyed_trace_root(
            &evaluation,
            coarse.lower,
            coarse.upper,
            local,
            &residual,
        )?
        else {
            return Err("the narrowed root is not established".into());
        };
        assert!(coarse.lower <= fine.lower && fine.upper <= coarse.upper);
        let (coarse_width, fine_width) = (coarse.upper - coarse.lower, fine.upper - fine.lower);
        assert!(
            fine_width * 8.0 < coarse_width,
            "coarse={coarse_width:e}, fine={fine_width:e}"
        );
        Ok(())
    }

    /// `E(0) = -head < 0`, and `E(t2) - E(t1) >= κ_E (t2 - t1)` with `κ_E = H`.
    ///
    /// The enclosures contain the exact values, so a false floor makes the largest certified
    /// rise `E(t2).upper - E(t1).lower` fall below `κ_E Δt`.
    fn reservoir_bracket_and_slope_floor_hold<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let row = &rows()[2];
        let reach = reach::<R>(client, row.area, row.discharge, 0.0)?;
        let context = &reach.context;
        let interior = interior(&reach, row.endpoint)?;
        let TraceRequest::Reservoir { reservoir, cap } =
            context.surveyed_trace_request(&interior, &policy(&reach, &row.policy)?)?
        else {
            return Err("a reservoir request is required".into());
        };
        assert!(!cap);
        let work = ledger(&reach)?;
        let evaluation = Evaluation {
            config: &geometry(&reach)?.riemann,
            panels: PANELS,
            work: &work,
        };
        let samples = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut residuals = Vec::new();
        for &t in &samples {
            let point = state(&reach, &interior, &evaluation, t, t)?;
            let value = context.surveyed_reservoir_residual(&interior, &point, &reservoir)?;
            residuals.push(ends(&value)?);
        }
        let head = 1.7831130485978541;
        assert!(residuals[0][1] < 0.0 && residuals[0][0] <= -head && -head <= residuals[0][1]);
        let [kappa, _] = ends(&interior.cap)?;
        for pair in 0..samples.len() - 1 {
            let rise = residuals[pair + 1][1] - residuals[pair][0];
            let floor = kappa * (samples[pair + 1] - samples[pair]);
            assert!(
                rise >= floor,
                "pair {pair}: rise={rise:e} below H Δt={floor:e}"
            );
        }
        Ok(())
    }

    /// The entering reservoir row contracts from `[0, 1]` around its 150-digit depth.
    ///
    /// Frozen criteria:
    /// - the contracted bracket holds the reference depth and is narrower than `1e-6`
    /// - the local slope floor near the root exceeds `κ_E = H`
    /// - the largest certified residual rise over that bracket meets the local floor
    fn reservoir_contraction_holds_the_reference<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let row = &rows()[2];
        let reach = reach::<R>(client, row.area, row.discharge, 0.0)?;
        let context = &reach.context;
        let interior = interior(&reach, row.endpoint)?;
        let TraceRequest::Reservoir { reservoir, .. } =
            context.surveyed_trace_request(&interior, &policy(&reach, &row.policy)?)?
        else {
            return Err("a reservoir request is required".into());
        };
        let work = ledger(&reach)?;
        let evaluation = Evaluation {
            config: &geometry(&reach)?.riemann,
            panels: PANELS,
            work: &work,
        };
        let [kappa, _] = ends(&interior.cap)?;
        let residual = |t: &Tensor<R>| {
            let point = context.surveyed_trace_at(&interior, t, &evaluation)?;
            context.surveyed_reservoir_residual(&interior, &point, &reservoir)
        };
        let slope = |a: f64, b: f64| {
            context.surveyed_reservoir_slope(&interior, &reservoir, &evaluation, a, b, kappa)
        };
        let (lower, upper) =
            context.surveyed_trace_contract(&evaluation, 0.0, 1.0, &residual, &slope)?;
        let point = state(&reach, &interior, &evaluation, lower, upper)?;
        let [low, high] = ends(&point.depth)?;
        let [floor, ceil] = row.depth.map(f64::from_bits);
        assert!(low <= floor && ceil <= high, "[{low:e}, {high:e}]");
        assert!(upper - lower < 1e-6, "width={:e}", upper - lower);
        let (t1, t2) = (lower - 1e-3, upper + 1e-3);
        let local = slope(t1, t2)?;
        assert!(local > kappa, "local={local:e}, H={kappa:e}");
        let rise = |t: f64| -> TestResult<[f64; 2]> {
            let point = state(&reach, &interior, &evaluation, t, t)?;
            ends(&context.surveyed_reservoir_residual(&interior, &point, &reservoir)?)
        };
        let gain = rise(t2)?[1] - rise(t1)?[0];
        assert!(gain >= local * (t2 - t1), "rise={gain:e}");
        Ok(())
    }

    #[test]
    fn cpu_reservoir_contraction_holds_the_reference() -> TestResult {
        reservoir_contraction_holds_the_reference::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_trace_roots_enclose_the_frozen_references() -> TestResult {
        roots_enclose_the_references::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_local_slope_floor_contracts_the_discharge_enclosure() -> TestResult {
        local_slope_floor_contracts_the_enclosure::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_reservoir_bracket_and_slope_floor_hold() -> TestResult {
        reservoir_bracket_and_slope_floor_hold::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_trace_roots_enclose_the_frozen_references() -> TestResult {
            roots_enclose_the_references::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_local_slope_floor_contracts_the_discharge_enclosure() -> TestResult {
            local_slope_floor_contracts_the_enclosure::<CudaRuntime>(CudaClient::new(
                CudaDevice::new(0),
            )?)
        }

        #[test]
        fn cuda_reservoir_bracket_and_slope_floor_hold() -> TestResult {
            reservoir_bracket_and_slope_floor_hold::<CudaRuntime>(CudaClient::new(
                CudaDevice::new(0),
            )?)
        }
    }
}
