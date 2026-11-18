use super::{
    point_quadrature_progress,
    point_quadrature_types::PointQuadratureProgress,
    point_root_session::{PointRootSession, invalid},
    wet_depth_types::WetDepthDomain,
};
use crate::channel::flow::surveyed::error::HydraulicResult;
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{integrate::CertifiedSimpsonAlgorithms, optimize::CertifiedMappedBisectionAlgorithms};

impl<R> PointRootSession<'_, '_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>,
{
    /// Advances each requested wet star row one Simpson level and re-refines its root bracket.
    /// Rows without a wet depth domain record `NotApplicable`. Star endpoints and candidates
    /// follow the new level at the next selection.
    pub(super) fn advance_wet_star_quadrature(
        &mut self,
        owners: &Tensor<R>,
    ) -> HydraulicResult<PointQuadratureProgress<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.wet.terminal {
            return Err(invalid(
                "wet star quadrature remains sealed after an earlier numerical error",
            ));
        }
        let mut progress = point_quadrature_progress::start(
            self,
            owners,
            None,
            "wet star quadrature complete request admission",
        )?;
        if progress.invocation_owner.numel() == 0 {
            return Ok(progress);
        }
        let (demand, association) = point_quadrature_progress::demand(
            self,
            &progress,
            "wet star quadrature nonempty demand",
        )?;
        let tolerance = self.wet.tolerance;
        let before = self.sources.evaluation.work.snapshot();
        let old_mapped = mapped(&self.wet.domains)?;
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            let mut output = Vec::with_capacity(self.wet.domains.len());
            for domain in std::mem::take(&mut self.wet.domains) {
                output.extend(super::wet_star_quadrature_request::advance(
                    self.sources,
                    &mut self.invocation,
                    domain,
                    &tolerance,
                    &demand,
                    &association,
                    &mut progress,
                )?);
            }
            self.wet.domains = output;
            Ok(())
        })();
        if result.is_err() {
            self.wet.terminal = true;
        }
        self.reconcile(checkpoint, result.is_ok())?;
        result?;
        point_quadrature_progress::finish(
            &mut progress,
            before,
            self.sources.evaluation.work.snapshot(),
            [old_mapped, mapped(&self.wet.domains)?],
            self.callback_points(),
        )?;
        Ok(progress)
    }
}

/// Mapped-coordinate work summed over every wet domain.
pub(super) fn mapped<R: Runtime<DType = DType>>(
    domains: &[WetDepthDomain<R>],
) -> HydraulicResult<usize> {
    domains.iter().try_fold(0usize, |sum, domain| {
        sum.checked_add(domain.mapped_points)
            .ok_or_else(|| invalid("wet mapped work sum overflows usize"))
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        corpus_coverage::Coverage,
        point_candidate_types::PointCandidate,
        point_flux_types::PointFluxCertificates,
        point_quadrature_types::{PointQuadratureOutcome as Outcome, PointQuadratureProgress},
        point_root_session::PointRootSession,
        source_fixture_support as frozen,
        wet_depth::fixture as support,
        wet_star_levels::tests::{MIRROR, trapezoid},
    };
    use crate::channel::flow::surveyed::{PrismaticRiemannConfig, work::PrismaticRiemannWork};
    use frozen::{Result, hydraulic};
    use numr::{dtype::DType, runtime::Runtime};
    use solvr::optimize::CertifiedRootTolerance;

    /// The reach allowances: water and momentum atol and the relative allowance are all 1e-12.
    const ALLOWANCE: f64 = 1e-12;

    /// Normalized-depth allowance of the narrowing runs, below F64 resolution at these depths.
    /// Bisection then stops only at its arithmetic floor.
    const DEPTH_ATOL: f64 = 1e-17;

    /// A gate-like star face: deep flow upstream of a shallow reach with the same discharge.
    const GATE: [[f64; 4]; 1] = [[24.0, 12.0, 12.0, 12.0]];

    fn strict(mut config: PrismaticRiemannConfig) -> PrismaticRiemannConfig {
        config.water_flux_atol_m3_per_s = ALLOWANCE;
        config.momentum_flux_atol_m4_per_s2 = ALLOWANCE;
        config.flux_rtol = ALLOWANCE;
        config
    }

    fn width(value: [f64; 2]) -> f64 {
        value[1] - value[0]
    }

    fn accepted(value: [f64; 2], atol: f64, rtol: f64) -> bool {
        width(value) <= atol + rtol * value[0].abs().max(value[1].abs())
    }

    /// Widths of one invocation row: the star Q* candidate, the joint water and momentum hulls.
    #[derive(Clone, Copy, Debug)]
    struct Widths {
        star: f64,
        water: [f64; 2],
        momentum: [f64; 2],
    }

    fn widths<R>(result: &PointFluxCertificates<R>, owner: i64) -> Result<Widths>
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let owners = result.candidates.invocation_owner.try_to_vec::<i64>()?;
        let row = owners
            .iter()
            .position(|value| *value == owner)
            .ok_or("Result lacks the requested owner")?;
        let mut star = None;
        for candidate in &result.candidates.candidates {
            if let PointCandidate::Star(s) = candidate {
                let wave = &result.candidates.waves[s.wave];
                let local = s.rows.try_to_vec::<i64>()?;
                let wave_owners = wave.star.invocation_owner.try_to_vec::<i64>()?;
                for index in local {
                    let index = usize::try_from(index)?;
                    if wave_owners[index] == owner {
                        star = Some(width(frozen::endpoints(&wave.star.discharge)?[index]));
                    }
                }
            }
        }
        Ok(Widths {
            star: star.ok_or("Result lacks a star candidate for the owner")?,
            water: frozen::endpoints(&result.joint_water)?[row],
            momentum: frozen::endpoints(&result.joint_momentum)?[row],
        })
    }

    fn work_delta<R: Runtime<DType = DType>>(
        progress: &PointQuadratureProgress<R>,
        before: PrismaticRiemannWork,
        after: PrismaticRiemannWork,
    ) {
        assert_eq!(
            after.residual_points.checked_sub(before.residual_points),
            Some(progress.additional_residual_points)
        );
        assert_eq!(
            after
                .simpson_value_points
                .checked_sub(before.simpson_value_points),
            Some(progress.additional_simpson_value_points)
        );
        assert_eq!(
            after
                .simpson_derivative_points
                .checked_sub(before.simpson_derivative_points),
            Some(progress.additional_simpson_derivative_points)
        );
    }

    /// Explicit calls advance one level each, until water meets the allowances or a ceiling stops them.
    fn narrow<R>(
        session: &mut PointRootSession<'_, '_, R>,
        work: &crate::channel::flow::surveyed::work::WorkLedger,
        config: &PrismaticRiemannConfig,
        label: &str,
    ) -> Result<bool>
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let c = session.sources.geometry.client().clone();
        let owners = support::rows::<R>(&c, &[0])?;
        hydraulic(session.select_point_fluxes(&owners))?;
        // At the configured root allowance the depth bracket converges while water stays over 1e-12.
        // A stricter depth allowance lets the panel level set the remaining width.
        hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance {
            atol: DEPTH_ATOL,
            rtol: 0.,
        }))?;
        let base = hydraulic(session.select_point_fluxes(&owners))?;
        let mut previous = widths(&base, 0)?;
        eprintln!(
            "WET_STAR_QUADRATURE\t{label}\tlevel=0\tstar={:e}\twater={:e}\tmomentum={:e}",
            previous.star,
            width(previous.water),
            width(previous.momentum)
        );
        let mut level = 0_i64;
        let mut water = accepted(previous.water, ALLOWANCE, ALLOWANCE);
        while !water {
            let before = work.snapshot();
            let old_points = session.callback_points();
            let progress = hydraulic(session.advance_wet_star_quadrature(&owners))?;
            work_delta(&progress, before, work.snapshot());
            assert!(progress.side.is_none());
            assert_eq!(progress.callback_points, session.callback_points());
            assert_eq!(
                progress.callback_points.checked_sub(old_points),
                Some(progress.additional_residual_points)
            );
            assert_eq!(progress.point_limit, config.root.max_points);
            assert_eq!(progress.old_level.try_to_vec::<i64>()?, [level]);
            let outcome = progress.outcome.item::<i64>()?;
            assert_eq!(
                outcome,
                Outcome::Advanced as i64,
                "{label}: water stays over its allowance at the ceiling, level={level}"
            );
            level += 1;
            assert_eq!(progress.new_level.try_to_vec::<i64>()?, [level]);
            assert!(usize::try_from(level)? <= config.max_refinements);
            // A converged bracket can need no midpoint at the new level, so Simpson work can be zero.
            assert!(work.snapshot().maximum_simpson_panels <= config.max_simpson_panels);
            let result = hydraulic(session.select_point_fluxes(&owners))?;
            let current = widths(&result, 0)?;
            let domain = session.wet.domains.first().ok_or("Missing wet domain")?;
            eprintln!(
                "WET_STAR_QUADRATURE\t{label}\tlevel={level}\tstar={:e}\twater={:e}\tmomentum={:e}\timproved={}\tdepth={:e}\tstatus={}",
                current.star,
                width(current.water),
                width(current.momentum),
                progress.bracket_improved.try_to_vec::<f64>()?[0],
                width(frozen::endpoints(&domain.depth)?[0]),
                domain.status.try_to_vec::<i64>()?[0],
            );
            assert!(
                current.star <= previous.star,
                "{label}: star Q* width at {level}"
            );
            assert!(width(current.water) <= width(previous.water));
            assert!(width(current.momentum) <= width(previous.momentum));
            let improved = progress.bracket_improved.try_to_vec::<f64>()?[0] != 0.;
            if improved {
                assert!(
                    current.star < previous.star,
                    "{label}: star Q* width at {level}"
                );
                assert!(
                    width(current.water) < width(previous.water),
                    "{label}: water width at {level}"
                );
                assert!(
                    width(current.momentum) < width(previous.momentum),
                    "{label}: momentum width at {level}"
                );
            }
            water = accepted(current.water, ALLOWANCE, ALLOWANCE);
            previous = current;
            // A level that leaves the bracket unchanged marks the residual arithmetic floor.
            if !improved {
                break;
            }
        }
        eprintln!(
            "WET_STAR_QUADRATURE\t{label}\tlevel={level}\twater_accepted={water}\tmomentum_accepted={}",
            accepted(previous.momentum, ALLOWANCE, ALLOWANCE)
        );
        Ok(water)
    }

    fn narrowing<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let section = trapezoid::<R>(client)?;
        let config = strict(PrismaticRiemannConfig::default());
        support::with_section::<R, _>(client, &section, &MIRROR, &config, |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            narrow(&mut session, work, &config, "mirror")
        })?;
        let gate =
            support::with_section::<R, _>(client, &section, &GATE, &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                narrow(&mut session, work, &config, "gate")
            })?;
        assert!(
            gate,
            "gate: water stays over its allowance within the ceilings"
        );
        Ok(())
    }

    fn run<R>(client: &R::Client, coverage: Coverage) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        narrowing::<R>(client)?;
        if coverage.is_full() {
            controls::level_limit::<R>(client)?;
            controls::work_limit::<R>(client)?;
            controls::not_applicable::<R>(client)?;
            controls::sealed::<R>(client)?;
        }
        Ok(())
    }

    #[test]
    fn configured_cpu_wet_star_quadrature_narrows_star_rows() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        run::<CpuRuntime>(&CpuClient::new(CpuDevice::new()), Coverage::Full)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_wet_star_quadrature_narrows_star_rows() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
    }

    mod controls {
        use super::*;
        use crate::channel::flow::surveyed::{
            PrismaticRiemannRejection, error::HydraulicError, work::WorkLedger,
        };
        use numr::{runtime::RuntimeClient, tensor::Tensor};

        /// Stored bits of one wet row: bracket, depth, normalized depth, status and level.
        #[derive(Debug, PartialEq, Eq)]
        struct RowState {
            bracket: [u64; 2],
            depth: [u64; 2],
            normalized: [u64; 2],
            status: i64,
            level: i64,
        }

        fn row_state<R>(session: &PointRootSession<'_, '_, R>, owner: i64) -> Result<RowState>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            for domain in &session.wet.domains {
                let owners = domain.invocation_owner.try_to_vec::<i64>()?;
                let Some(row) = owners.iter().position(|value| *value == owner) else {
                    continue;
                };
                let bits = |value: [f64; 2]| value.map(f64::to_bits);
                let parameter = &domain.query.query.parameter;
                return Ok(RowState {
                    bracket: [
                        parameter.lower().try_to_vec::<f64>()?[row].to_bits(),
                        parameter.upper().try_to_vec::<f64>()?[row].to_bits(),
                    ],
                    depth: bits(frozen::endpoints(&domain.depth)?[row]),
                    normalized: bits(frozen::endpoints(&domain.normalized_depth)?[row]),
                    status: domain.status.try_to_vec::<i64>()?[row],
                    level: domain.levels.levels().try_to_vec::<i64>()?[row],
                });
            }
            Err("Session lacks a wet domain for the owner".into())
        }

        fn mirror<R, T>(
            client: &R::Client,
            states: &[[f64; 4]],
            config: &PrismaticRiemannConfig,
            operation: impl FnOnce(
                &mut crate::channel::flow::surveyed_points::paired_sources::PairedSources<'_, R>,
                &WorkLedger,
            ) -> Result<T>,
        ) -> Result<T>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let section = trapezoid::<R>(client)?;
            support::with_section::<R, _>(client, &section, states, config, operation)
        }

        fn total(work: PrismaticRiemannWork) -> Result<usize> {
            work.residual_points
                .checked_add(work.simpson_value_points)
                .and_then(|points| points.checked_add(work.simpson_derivative_points))
                .ok_or("Wet quadrature control work sum exceeds usize".into())
        }

        /// `max_refinements = 1` admits one level, and the next call charges nothing.
        pub(super) fn level_limit<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = strict(PrismaticRiemannConfig::default());
            config.max_refinements = 1;
            mirror::<R, _>(client, &MIRROR, &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                hydraulic(session.select_point_fluxes(&owners))?;
                let first = hydraulic(session.advance_wet_star_quadrature(&owners))?;
                assert_eq!(
                    first.outcome.try_to_vec::<i64>()?,
                    [Outcome::Advanced as i64]
                );
                assert_eq!(first.old_level.try_to_vec::<i64>()?, [0]);
                assert_eq!(first.new_level.try_to_vec::<i64>()?, [1]);
                let state = row_state(&session, 0)?;
                let before = work.snapshot();
                let points = session.callback_points();
                let limited = hydraulic(session.advance_wet_star_quadrature(&owners))?;
                assert_eq!(
                    limited.outcome.try_to_vec::<i64>()?,
                    [Outcome::LevelLimit as i64]
                );
                assert_eq!(limited.old_level.try_to_vec::<i64>()?, [1]);
                assert_eq!(limited.new_level.try_to_vec::<i64>()?, [1]);
                assert_eq!(limited.bracket_improved.try_to_vec::<f64>()?, [0.]);
                assert_eq!(limited.additional_residual_points, 0);
                assert_eq!(limited.additional_simpson_value_points, 0);
                assert_eq!(limited.additional_mapped_points, 0);
                assert_eq!(work.snapshot(), before);
                assert_eq!(session.callback_points(), points);
                assert_eq!(row_state(&session, 0)?, state);
                assert_eq!(session.point_limit(), config.root.max_points);
                Ok(())
            })
        }

        /// An exhausted or nearly exhausted point limit stops the call, and executed work matches recorded work.
        pub(super) fn work_limit<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = strict(PrismaticRiemannConfig::default());
            let used = mirror::<R, _>(client, &MIRROR, &config, |pair, _| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                hydraulic(session.select_point_fluxes(&owners))?;
                Ok(session.callback_points())
            })?;
            assert!(used > 0);
            for extra in [0, 1] {
                let limit = used + extra;
                mirror::<R, _>(client, &MIRROR, &config, |pair, work| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let mut session = hydraulic(PointRootSession::new(pair, &owners, limit))?;
                    hydraulic(session.select_point_fluxes(&owners))?;
                    assert_eq!(session.callback_points(), used);
                    let state = row_state(&session, 0)?;
                    let before = work.snapshot();
                    let progress = hydraulic(session.advance_wet_star_quadrature(&owners))?;
                    let after = work.snapshot();
                    work_delta(&progress, before, after);
                    assert_eq!(
                        session.callback_points().checked_sub(used),
                        Some(progress.additional_residual_points)
                    );
                    assert!(session.callback_points() <= limit);
                    assert_eq!(progress.point_limit, limit);
                    assert_eq!(session.point_limit(), limit);
                    let outcome = progress.outcome.item::<i64>()?;
                    eprintln!(
                        "WET_STAR_WORK_LIMIT\textra={extra}\toutcome={outcome}\tresidual={}",
                        progress.additional_residual_points
                    );
                    if extra == 0 {
                        assert_eq!(outcome, Outcome::WorkLimit as i64);
                        assert_eq!(progress.new_level.try_to_vec::<i64>()?, [0]);
                        assert_eq!(after, before);
                        assert_eq!(row_state(&session, 0)?, state);
                    } else {
                        assert!(
                            outcome == Outcome::WorkLimit as i64
                                || outcome == Outcome::Advanced as i64,
                            "Tiny point limit outcome={outcome}"
                        );
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }

        /// A row without a wet star is untouched, and an undemanded wet row keeps every stored bit.
        pub(super) fn not_applicable<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = strict(PrismaticRiemannConfig::default());
            let states = [MIRROR[0], [12.0, 0.0, 0.0, 0.0], [12.0, -30.0, 12.0, 36.0]];
            mirror::<R, _>(client, &states, &config, |pair, _| {
                let all = support::rows::<R>(client, &[0, 1, 2])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &all, config.root.max_points))?;
                let base = hydraulic(session.select_point_fluxes(&all))?;
                let dry = [
                    frozen::endpoints(&base.joint_water)?[1].map(f64::to_bits),
                    frozen::endpoints(&base.joint_momentum)?[1].map(f64::to_bits),
                ];
                assert!(row_state(&session, 1).is_err(), "Dry row owns a wet domain");
                let undemanded = row_state(&session, 2)?;
                let owners = support::rows::<R>(client, &[0, 1])?;
                let progress = hydraulic(session.advance_wet_star_quadrature(&owners))?;
                assert_eq!(progress.invocation_owner.try_to_vec::<i64>()?, [0, 1]);
                assert_eq!(
                    progress.outcome.try_to_vec::<i64>()?,
                    [Outcome::Advanced as i64, Outcome::NotApplicable as i64]
                );
                assert_eq!(progress.old_level.try_to_vec::<i64>()?, [0, -1]);
                assert_eq!(progress.new_level.try_to_vec::<i64>()?, [1, -1]);
                assert_eq!(row_state(&session, 2)?, undemanded);
                assert_eq!(row_state(&session, 0)?.level, 1);
                let refined = hydraulic(session.select_point_fluxes(&all))?;
                assert_eq!(
                    [
                        frozen::endpoints(&refined.joint_water)?[1].map(f64::to_bits),
                        frozen::endpoints(&refined.joint_momentum)?[1].map(f64::to_bits),
                    ],
                    dry
                );
                Ok(())
            })
        }

        /// A ledger rejection inside the refresh consumes the continuation and seals the session.
        pub(super) fn sealed<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = strict(PrismaticRiemannConfig::default());
            let seed = mirror::<R, _>(client, &MIRROR, &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                hydraulic(session.select_point_fluxes(&owners))?;
                total(work.snapshot())
            })?;
            let ceiling = seed
                .checked_add(1)
                .ok_or("Measured wet quadrature error ceiling exceeds usize")?;
            config.max_native_query_points = ceiling;
            mirror::<R, _>(client, &MIRROR, &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                hydraulic(session.select_point_fluxes(&owners))?;
                assert_eq!(total(work.snapshot())?, seed);
                let before = work.snapshot();
                let points = session.callback_points();
                let result = session.advance_wet_star_quadrature(&owners);
                assert!(
                    matches!(
                        result,
                        Err(HydraulicError::Rejected {
                            reason: PrismaticRiemannRejection::WorkLimit,
                            ..
                        })
                    ),
                    "Measured ceiling does not exercise a consumed refresh"
                );
                let after = work.snapshot();
                let residual = after
                    .residual_points
                    .checked_sub(before.residual_points)
                    .ok_or("Failed wet quadrature decreases executed residual work")?;
                assert!(
                    residual > 0,
                    "Error control never reaches the residual callback"
                );
                assert!(total(after)? <= ceiling);
                assert_eq!(
                    session.callback_points().checked_sub(points),
                    Some(residual)
                );
                assert_eq!(session.point_limit(), config.root.max_points);
                assert!(session.wet.terminal);
                assert!(session.advance_wet_star_quadrature(&owners).is_err());
                assert!(session.select_point_fluxes(&owners).is_err());
                assert_eq!(work.snapshot(), after);
                let empty = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
                assert!(session.advance_wet_star_quadrature(&empty).is_err());
                assert_eq!(work.snapshot(), after);
                Ok(())
            })
        }
    }
}
