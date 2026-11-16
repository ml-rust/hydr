use super::super::super::{
    corpus_coverage::Coverage,
    point_flux_types::PointFluxStatus,
    point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
    point_quadrature_types::PointQuadratureOutcome as QuadratureOutcome,
    point_root_session::PointRootSession,
    source_tests::support::{self as frozen, Result, hydraulic},
    wet_depth::tests::support,
    wet_star_levels::tests::{MIRROR, trapezoid},
};
use crate::channel::flow::surveyed::{
    PrismaticRiemannConfig, PrismaticRiemannError, PrismaticRiemannRejection,
};
use numr::{dtype::DType, runtime::Runtime};

/// The reach allowances: water and momentum atol and the relative allowance are all 1e-12.
const ALLOWANCE: f64 = 1e-12;

/// A gate-like star face: deep flow upstream of a shallow reach with the same discharge.
const GATE: [f64; 4] = [24.0, 12.0, 12.0, 12.0];

/// A dam break onto a shallow pool. The left rarefaction straddles the interface.
const FAN: [f64; 4] = [24.0, 0.0, 1.0, 0.0];

type Outcome<R> = std::result::Result<PrismaticPointRiemannResult<R>, PrismaticRiemannError>;

fn config() -> PrismaticRiemannConfig {
    PrismaticRiemannConfig {
        water_flux_atol_m3_per_s: ALLOWANCE,
        momentum_flux_atol_m4_per_s2: ALLOWANCE,
        flux_rtol: ALLOWANCE,
        ..PrismaticRiemannConfig::default()
    }
}

/// One public call on the exact trapezoid, with every face on section 0.
fn call<R>(client: &R::Client, states: &[[f64; 4]]) -> Result<Outcome<R>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let section = trapezoid::<R>(client)?;
    let owners = support::rows::<R>(client, &vec![0; states.len()])?;
    let inputs = (0..4)
        .map(|field| {
            frozen::expansion::<R>(
                client,
                &states
                    .iter()
                    .map(|state| [state[field].to_bits(), 0, 0])
                    .collect::<Vec<_>>(),
                true,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(section.riemann_flux_points_tensor(
        &owners,
        &inputs[0],
        &inputs[1],
        &inputs[2],
        &inputs[3],
        &config(),
    ))
}

fn within(value: [f64; 2]) -> bool {
    value[1] - value[0] <= ALLOWANCE + ALLOWANCE * value[0].abs().max(value[1].abs())
}

/// Accepted faces carry the expected trace kinds and flux bounds within the allowances.
fn accepted<R>(
    client: &R::Client,
    label: &str,
    states: &[[f64; 4]],
    kinds: &[PrismaticPointTraceKind],
) -> Result<PrismaticPointRiemannResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let result = call::<R>(client, states)?
        .map_err(|error| format!("{label} rejects at the public entry: {error:?}"))?;
    let codes = result.trace_kind().try_to_vec::<i64>()?;
    let actual = codes
        .iter()
        .map(|code| PrismaticPointTraceKind::from_code(*code))
        .collect::<Vec<_>>();
    let expected = kinds.iter().copied().map(Some).collect::<Vec<_>>();
    assert_eq!(actual, expected, "{label} trace kinds");
    let water = frozen::endpoints(result.water_flux_bound_m3_per_s())?;
    let momentum = frozen::endpoints(result.momentum_flux_bound_m4_per_s2())?;
    for face in 0..states.len() {
        assert!(
            within(water[face]),
            "{label} face {face} water {:?}",
            water[face]
        );
        assert!(
            within(momentum[face]),
            "{label} face {face} momentum {:?}",
            momentum[face]
        );
    }
    let work = result.work();
    eprintln!(
        "POINT_RIEMANN_STAR\t{label}\trefinements={}\tpanels={}\tresidual_points={}",
        work.refinements, work.maximum_simpson_panels, work.residual_points
    );
    assert!(work.refinements <= config().max_refinements);
    assert!(work.maximum_simpson_panels <= config().max_simpson_panels);
    Ok(result)
}

/// The gate star face is over budget at the initial panels and accepted after star rounds.
fn gate<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let result = accepted::<R>(client, "gate", &[GATE], &[PrismaticPointTraceKind::Star])?;
    assert!(result.work().refinements >= 1);
    Ok(())
}

/// A star face and a sonic-fan face share their rounds, so the batch needs no more rounds
/// than the slower face alone.
fn shared_rounds<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let star = accepted::<R>(client, "gate", &[GATE], &[PrismaticPointTraceKind::Star])?;
    let fan = accepted::<R>(client, "fan", &[FAN], &[PrismaticPointTraceKind::LeftFan])?;
    let both = accepted::<R>(
        client,
        "gate and fan",
        &[GATE, FAN],
        &[
            PrismaticPointTraceKind::Star,
            PrismaticPointTraceKind::LeftFan,
        ],
    )?;
    let [star, fan] = [star.work().refinements, fan.work().refinements];
    assert!(star >= 1 && fan >= 1, "gate={star} fan={fan}");
    assert_eq!(both.work().refinements, star.max(fan));
    Ok(())
}

/// The exact mirror stops at its residual floor: its star continuation ends over budget.
fn mirror<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = config();
    let section = trapezoid::<R>(client)?;
    support::with_section::<R, _>(client, &section, &MIRROR, &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        let result = hydraulic(session.resolve_point_fluxes(&owners))?;
        let outcome = result.star_outcome.try_to_vec::<i64>()?;
        let level = result.star_level.try_to_vec::<i64>()?;
        eprintln!(
            "POINT_RIEMANN_STAR\tmirror\trounds={}\tstar_outcome={outcome:?}\tstar_level={level:?}",
            result.rounds
        );
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [PointFluxStatus::FluxBudgetExceeded as i64]
        );
        // The star stops at the last admissible level and names the level ceiling.
        assert_eq!(outcome, [QuadratureOutcome::LevelLimit as i64]);
        assert_eq!(level, [i64::try_from(config.max_refinements)?]);
        assert_eq!(result.sonic_level.try_to_vec::<i64>()?, [-1]);
        assert_eq!(session.callback_points(), work.snapshot().residual_points);
        Ok(())
    })?;
    match call::<R>(client, &MIRROR)? {
        Err(PrismaticRiemannError::Rejected { reason, query, .. }) => {
            assert_eq!(reason, PrismaticRiemannRejection::FluxBudgetExceeded);
            assert_eq!(query, Some(0));
            Ok(())
        }
        other => Err(format!("mirror returns {:?}", other.err()).into()),
    }
}

pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    gate::<R>(client)?;
    if coverage.is_full() {
        shared_rounds::<R>(client)?;
        mirror::<R>(client)?;
    }
    Ok(())
}

#[test]
fn configured_cpu_public_point_fluxes_refine_star_and_fan_faces() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    run::<CpuRuntime>(&CpuClient::new(CpuDevice::new()), Coverage::Full)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_public_point_fluxes_refine_star_and_fan_faces() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
}
