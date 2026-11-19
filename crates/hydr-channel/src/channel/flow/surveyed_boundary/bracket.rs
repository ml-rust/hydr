//! The `Discharge` trace bracket above the outflow sonic depth, and its slope floor.

use super::{
    client::TraceClient,
    error::trace_error,
    interior::Interior,
    point::{Evaluation, TraceDepth, TracePoint},
    roots::{RootEnclosure, RootOutcome},
};
use crate::channel::flow::{
    ChannelError, boundary::boundary_rejection, context::Context, surveyed::error::checked,
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// A normalized-depth bracket `[lower, 1]` and its slope lower bound over the whole bracket.
#[derive(Clone, Copy, Debug)]
pub(super) struct Bracket {
    pub lower: f64,
    pub kappa: f64,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// The `Discharge` bracket `[t_L, 1]` with `d_s < t_L H`, `G(t_L) >= 0` and its slope floor.
    ///
    /// `-G` increases above the outflow sonic depth `d_s`, with `-dG/dd >= T (c - u_b)`.
    /// Both factors are nondecreasing, so their certified product at `t_L` bounds the slope
    /// over the whole bracket. The interior depth serves as `t_L` when `G(d_i) >= 0`. Otherwise
    /// the sonic root of `c - u_b` below `d_i` gives `t_L`, rounded up by its enclosure.
    ///
    /// The inner error is a rejection that finer quadrature can resolve. A target above
    /// every certified discharge over the sonic enclosure exceeds the sonic outflow `A u_b(d_s)`,
    /// the peak of the side curve, and rejects at any quadrature level.
    pub(super) fn surveyed_discharge_bracket(
        &self,
        interior: &Interior<R>,
        target: &Tensor<R>,
        evaluation: &Evaluation<'_>,
        iterations: &mut usize,
    ) -> Result<Result<Bracket, ChannelError>, ChannelError> {
        let (interior_t, sonic_slope) = self.surveyed_sonic_setup(interior)?;
        // A candidate lower end: the upper bound of -G there and its slope floor.
        let accept = |t: f64| -> Result<Option<(f64, f64)>, ChannelError> {
            let point = self.surveyed_trace_at(interior, &self.tensor(&[t])?, evaluation)?;
            let Some(kappa) = self.surveyed_discharge_slope(interior, &point)? else {
                return Ok(None);
            };
            let discharge = self.surveyed_discharge_residual(interior, &point, target)?;
            Ok(Some((discharge.upper().item::<f64>()?, kappa)))
        };
        if interior_t > 0.0
            && let Some((discharge, kappa)) = accept(interior_t)?
            && discharge <= 0.0
        {
            return Ok(Ok(Bracket {
                lower: interior_t,
                kappa,
            }));
        }
        let root = match self.surveyed_sonic_root(interior, evaluation, interior_t, sonic_slope)? {
            Ok(root) => root,
            Err(error) => return Ok(Err(error)),
        };
        *iterations = (*iterations).max(root.iterations);
        // Step up from the sonic enclosure until c - u_b has a positive certified lower bound.
        let mut gap = (root.upper - root.lower).max(root.upper * f64::EPSILON);
        let mut t = root.upper;
        // The last depth without a certified subsonic gap, below every admissible lower end.
        let mut below = root.lower;
        let mut above = None;
        for _ in 0..64 {
            if let Some((discharge, kappa)) = accept(t)? {
                if discharge <= 0.0 {
                    return Ok(Ok(Bracket { lower: t, kappa }));
                }
                above = Some(t);
                break;
            }
            below = t;
            if t >= interior_t {
                break;
            }
            t = (t + gap).min(interior_t);
            gap *= 2.0;
        }
        // A doubled step can pass the root, where -G > 0. A target below the sonic outflow
        // keeps G(t) >= 0 on (d_s, d_root], so bisection between the two ends finds it.
        if let Some(mut above) = above {
            for _ in 0..64 {
                let middle = 0.5 * (below + above);
                if !(below < middle && middle < above) {
                    break;
                }
                match accept(middle)? {
                    Some((discharge, kappa)) if discharge <= 0.0 => {
                        return Ok(Ok(Bracket {
                            lower: middle,
                            kappa,
                        }));
                    }
                    Some(_) => above = middle,
                    None => below = middle,
                }
            }
        }
        // The sonic depth lies in the root enclosure and A u_b peaks there. A residual
        // q - A u_b certainly positive over that enclosure puts q above the peak.
        let (lower, upper) = (self.tensor(&[root.lower])?, self.tensor(&[root.upper])?);
        let t = checked(&self.client, "surveyed sonic enclosure", |a| {
            Ok(a.bounds(&lower, &upper)?)
        })
        .map_err(trace_error)?;
        let peak = self.surveyed_trace_point(interior, TraceDepth::Normalized(&t), evaluation)?;
        let residual = self.surveyed_discharge_residual(interior, &peak, target)?;
        if residual.lower().item::<f64>()? > 0.0 {
            return Err(boundary_rejection(
                "requested discharge reaches the lower sonic uncertainty band: it exceeds the sonic outflow of the end cell",
            ));
        }
        Ok(Err(unresolved_sonic("requested discharge")))
    }

    /// The interior normalized depth `t_i` and the sonic slope floor `H sqrt(g / d_i)`.
    pub(super) fn surveyed_sonic_setup(
        &self,
        interior: &Interior<R>,
    ) -> Result<(f64, f64), ChannelError> {
        let c = interior.sources.geometry.client();
        let gravity = self.definition.config.gravity;
        let (interior_t, sonic_slope) = checked(c, "surveyed discharge bracket", |a| {
            let depth = a.point(interior.depth.lower())?;
            let cap = a.point(interior.cap.upper())?;
            let ratio = a.div_positive_interval(&depth, &cap)?;
            let g = a.point(&Tensor::<R>::full_scalar(
                &[1],
                DType::F64,
                gravity,
                c.device(),
            )?)?;
            let deepest = a.point(interior.depth.upper())?;
            let ratio_g = a.div_positive_interval(&g, &deepest)?;
            let slope = a.sqrt_nonnegative(&ratio_g)?;
            let floor = a.point(interior.cap.lower())?;
            let slope = a.mul(&slope, &floor)?;
            Ok((ratio.lower().clone(), slope.lower().clone()))
        })
        .map_err(trace_error)?;
        Ok((
            interior_t.item::<f64>()?.min(1.0),
            sonic_slope.item::<f64>()?,
        ))
    }

    /// The certified outflow sonic root of `c - u_b` over `[0, t_i]`.
    ///
    /// On the rarefaction branch `dS/dd >= c T / A = sqrt(g T / A)`, and `A <= T d` gives
    /// `sqrt(g / d)`. Below `d_i` it is at least `sqrt(g / d_i)`, times `H` for `t = d / H`.
    /// The inner error marks a root that finer quadrature can establish.
    pub(super) fn surveyed_sonic_root(
        &self,
        interior: &Interior<R>,
        evaluation: &Evaluation<'_>,
        interior_t: f64,
        sonic_slope: f64,
    ) -> Result<Result<RootEnclosure, ChannelError>, ChannelError> {
        if !(interior_t > 0.0 && sonic_slope > 0.0 && sonic_slope.is_finite()) {
            return Err(unresolved_sonic("sonic slope floor"));
        }
        let sonic = |t: &Tensor<R>| {
            let point = self.surveyed_trace_at(interior, t, evaluation)?;
            self.surveyed_sonic_residual(interior, &point)
        };
        Ok(
            match self.surveyed_trace_root(evaluation, 0.0, interior_t, sonic_slope, &sonic)? {
                RootOutcome::Root(root) => Ok(root),
                RootOutcome::NoRoot | RootOutcome::Unresolved => {
                    Err(unresolved_sonic("outflow sonic depth"))
                }
            },
        )
    }

    /// Normalized slope floor `κ_Q = H T(d) (c - u_b)(d)` of `-G` at and above the state `point`.
    ///
    /// `T` and `c - u_b` are nondecreasing above `d_s`, so the product of their certified
    /// lower bounds at `point` holds over every deeper depth. Returns `None` unless the
    /// product is finite and positive, which also certifies `d > d_s`.
    pub(super) fn surveyed_discharge_slope(
        &self,
        interior: &Interior<R>,
        point: &TracePoint<R>,
    ) -> Result<Option<f64>, ChannelError> {
        let c = interior.sources.geometry.client();
        let sonic = self.surveyed_sonic_residual(interior, point)?;
        if sonic.lower().item::<f64>()? <= 0.0 {
            return Ok(None);
        }
        let kappa = checked(c, "surveyed discharge slope floor", |a| {
            let width = a.point(point.top_width.lower())?;
            let gap = a.point(sonic.lower())?;
            let cap = a.point(interior.cap.lower())?;
            let product = a.mul(&width, &gap)?;
            Ok(a.mul(&product, &cap)?.lower().clone())
        })
        .map_err(trace_error)?
        .item::<f64>()?;
        Ok((kappa > 0.0 && kappa.is_finite()).then_some(kappa))
    }
}

fn unresolved_sonic(quantity: &str) -> ChannelError {
    boundary_rejection(&format!(
        "requested discharge reaches the lower sonic uncertainty band at the {quantity}"
    ))
}

#[cfg(test)]
mod tests {
    use super::super::{
        fixture::{CAP, GRAVITY, PANELS, ends, geometry, interior, ledger, reach, state},
        point::Evaluation,
    };
    use super::*;
    use crate::channel::flow::surveyed_update::tests::{Client, TestResult, cpu};
    use numr::runtime::cpu::CpuRuntime;

    /// Frozen leaving row: three bands, `A = 10.375`, `Q = 1`, right end, trace 4.8e-7 m above
    /// knot 2. The 150-digit outflow sonic depth is `1.350778904582332`.
    const LEAVING: (f64, f64, f64) = (10.375, 1.0, 10.184547080871404);
    const SONIC_CEIL: u64 = 0x3ff59cca5734ecd5;
    /// `H T (c - u_b)` at the 150-digit leaving root `d = 2.000000476837158`.
    const LEAVING_ROOT_SLOPE: f64 =
        CAP * 5.000001430511474 * (3.8360136167351895 - 1.3579391791055526);

    /// `-G(t2) - (-G(t1)) >= κ (t2 - t1)` at sample pairs over the bracket.
    ///
    /// The enclosures contain the exact values, so a false floor makes the largest certified
    /// rise `-G(t2).upper - (-G(t1)).lower` fall below `κ Δt`.
    fn check_slope<R>(
        reach: &crate::channel::flow::ChannelReach<R>,
        evaluation: &Evaluation<'_>,
        target: f64,
        bracket: Bracket,
    ) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let context = &reach.context;
        let interior = interior(reach, 1)?;
        let target = context.tensor(&[target])?;
        let span = 1.0 - bracket.lower;
        let samples: Vec<f64> = (0..=4)
            .map(|k| bracket.lower + span * f64::from(k) / 4.0)
            .collect();
        let mut residuals = Vec::new();
        for &t in &samples {
            let point = state(reach, &interior, evaluation, t, t)?;
            let value = context.surveyed_discharge_residual(&interior, &point, &target)?;
            residuals.push(ends(&value)?);
        }
        for pair in 0..samples.len() - 1 {
            let rise = residuals[pair + 1][1] - residuals[pair][0];
            let floor = bracket.kappa * (samples[pair + 1] - samples[pair]);
            assert!(
                rise >= floor,
                "pair {pair}: rise={rise:e} below κ Δt={floor:e}"
            );
        }
        Ok(())
    }

    fn leaving_bracket_lies_above_the_sonic_depth<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let (area, discharge, target) = LEAVING;
        let reach = reach::<R>(client, area, discharge, 0.0)?;
        let context = &reach.context;
        let work = ledger(&reach)?;
        let evaluation = Evaluation {
            config: &geometry(&reach)?.riemann,
            panels: PANELS,
            work: &work,
        };
        let interior = interior(&reach, 1)?;
        let mut iterations = 0;
        let bracket = context.surveyed_discharge_bracket(
            &interior,
            &context.tensor(&[target])?,
            &evaluation,
            &mut iterations,
        )??;
        // The certified lower end lies strictly above the 150-digit sonic depth.
        assert!(
            bracket.lower * CAP > f64::from_bits(SONIC_CEIL),
            "{bracket:?}"
        );
        // Below the interior depth 2.47 m, because the target exceeds the interior discharge.
        assert!(bracket.lower < 2.47 / CAP);
        assert!(
            iterations > 0,
            "the sonic root solves below the interior depth"
        );
        assert!(bracket.kappa > 0.0 && bracket.kappa <= LEAVING_ROOT_SLOPE);
        check_slope(&reach, &evaluation, target, bracket)
    }

    /// The sonic root of `c - u_b` below the interior depth encloses the 150-digit sonic depth.
    fn sonic_root_encloses_the_reference<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let (area, discharge, _) = LEAVING;
        let reach = reach::<R>(client, area, discharge, 0.0)?;
        let context = &reach.context;
        let work = ledger(&reach)?;
        let evaluation = Evaluation {
            config: &geometry(&reach)?.riemann,
            panels: PANELS,
            work: &work,
        };
        let interior = interior(&reach, 1)?;
        let [_, deepest] = ends(&interior.depth)?;
        // H sqrt(g / d_i), rounded down by a relative margin far above rounding.
        let kappa = CAP * (GRAVITY / deepest).sqrt() * (1.0 - 1e-12);
        let upper = 2.47 / CAP;
        let sonic = |t: &Tensor<R>| {
            let point = context.surveyed_trace_at(&interior, t, &evaluation)?;
            context.surveyed_sonic_residual(&interior, &point)
        };
        let RootOutcome::Root(root) =
            context.surveyed_trace_root(&evaluation, 0.0, upper, kappa, &sonic)?
        else {
            return Err("the sonic root is not established".into());
        };
        let floor = 0x3ff59cca5734ecd4_u64;
        let enclosure = state(&reach, &interior, &evaluation, root.lower, root.upper)?;
        let [lower, upper] = ends(&enclosure.depth)?;
        assert!(lower <= f64::from_bits(floor) && f64::from_bits(SONIC_CEIL) <= upper);
        assert!(
            upper - lower <= 1e-11,
            "sonic enclosure [{lower:e}, {upper:e}]"
        );
        Ok(())
    }

    /// Entering flow at the interior depth already has `G >= 0`, so no sonic root is solved.
    fn entering_bracket_starts_at_the_interior<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        // Frozen entering row: A = 5.125, Q = 0.5, left end, global Q = 5.06285120531442.
        let reach = reach::<R>(client, 5.125, 0.5, 0.0)?;
        let context = &reach.context;
        let work = ledger(&reach)?;
        let evaluation = Evaluation {
            config: &geometry(&reach)?.riemann,
            panels: PANELS,
            work: &work,
        };
        let interior = interior(&reach, 0)?;
        let mut iterations = 0;
        let bracket = context.surveyed_discharge_bracket(
            &interior,
            &context.tensor(&[-5.06285120531442])?,
            &evaluation,
            &mut iterations,
        )??;
        let [depth, _] = ends(&interior.depth)?;
        assert!(bracket.lower <= depth / CAP && bracket.lower >= (depth / CAP).next_down());
        assert_eq!(iterations, 0);
        assert!(bracket.kappa > 0.0);
        Ok(())
    }

    #[test]
    fn cpu_leaving_discharge_bracket_lies_above_the_sonic_depth() -> TestResult {
        leaving_bracket_lies_above_the_sonic_depth::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_sonic_root_encloses_the_reference_sonic_depth() -> TestResult {
        sonic_root_encloses_the_reference::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_entering_discharge_bracket_starts_at_the_interior_depth() -> TestResult {
        entering_bracket_starts_at_the_interior::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_leaving_discharge_bracket_lies_above_the_sonic_depth() -> TestResult {
            leaving_bracket_lies_above_the_sonic_depth::<CudaRuntime>(CudaClient::new(
                CudaDevice::new(0),
            )?)
        }

        #[test]
        fn cuda_sonic_root_encloses_the_reference_sonic_depth() -> TestResult {
            sonic_root_encloses_the_reference::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_entering_discharge_bracket_starts_at_the_interior_depth() -> TestResult {
            entering_bracket_starts_at_the_interior::<CudaRuntime>(CudaClient::new(
                CudaDevice::new(0),
            )?)
        }
    }
}
