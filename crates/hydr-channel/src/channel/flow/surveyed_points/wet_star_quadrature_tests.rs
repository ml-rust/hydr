use super::super::{
    corpus_coverage::Coverage,
    point_candidate_types::PointCandidate,
    point_flux_types::PointFluxCertificates,
    point_quadrature_types::{PointQuadratureOutcome as Outcome, PointQuadratureProgress},
    point_root_session::PointRootSession,
    source_tests::support as frozen,
    wet_depth::tests::support,
    wet_star_levels::tests::{MIRROR, trapezoid},
};
use crate::channel::flow::surveyed::{PrismaticRiemannConfig, work::PrismaticRiemannWork};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

#[path = "wet_star_quadrature_tests_controls.rs"]
mod controls;

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
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        narrow(&mut session, work, &config, "mirror")
    })?;
    let gate = support::with_section::<R, _>(client, &section, &GATE, &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
