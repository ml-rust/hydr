use super::super::super::{
    corpus_coverage::Coverage,
    point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
    source_tests::support::{self as frozen, Result, hydraulic},
    wet_depth::tests::support,
    wet_star_levels::tests::{MIRROR, WALLS, integer_trapezoid, trapezoid},
};
use crate::channel::{
    flow::surveyed::{PrismaticRiemannConfig, PrismaticRiemannError, error::checked},
    section::PrismaticSectionBatch,
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

/// One public call with every face on section 0.
fn call<R>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    states: &[[f64; 4]],
) -> Result<Outcome<R>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
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
    section: &PrismaticSectionBatch<R>,
    label: &str,
    states: &[[f64; 4]],
    kinds: &[PrismaticPointTraceKind],
) -> Result<PrismaticPointRiemannResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let result = call::<R>(client, section, states)?
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
    let section = trapezoid::<R>(client)?;
    let result = accepted::<R>(
        client,
        &section,
        "gate",
        &[GATE],
        &[PrismaticPointTraceKind::Star],
    )?;
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
    let section = trapezoid::<R>(client)?;
    let star = accepted::<R>(
        client,
        &section,
        "gate",
        &[GATE],
        &[PrismaticPointTraceKind::Star],
    )?;
    let fan = accepted::<R>(
        client,
        &section,
        "fan",
        &[FAN],
        &[PrismaticPointTraceKind::LeftFan],
    )?;
    let both = accepted::<R>(
        client,
        &section,
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

/// Exact mirror faces take the reflection identity at the public entry: zero water in every
/// component, and a selected momentum inside its bound.
fn mirror<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let level = trapezoid::<R>(client)?;
    let integer = integer_trapezoid::<R>(client)?;
    let faces = std::iter::once((&level, "mirror", MIRROR[0])).chain(
        WALLS
            .into_iter()
            .map(|(label, state)| (&integer, label, state)),
    );
    for (section, label, state) in faces {
        let result = accepted::<R>(
            client,
            section,
            label,
            &[state],
            &[PrismaticPointTraceKind::Star],
        )?;
        for value in [
            result.water_flux_m3_per_s(),
            result.trace_discharge_m3_per_s(),
        ] {
            assert_eq!(frozen::components(value)?, [[0; 3]], "{label}");
        }
        for water in [
            result.water_flux_bound_m3_per_s(),
            result.physical_water_flux_m3_per_s(),
        ] {
            for value in [water.lower(), water.upper()] {
                let bits = value.try_to_vec::<f64>()?.into_iter().map(f64::to_bits);
                assert_eq!(bits.collect::<Vec<_>>(), [0], "{label}");
            }
        }
        let bound = frozen::endpoints(result.momentum_flux_bound_m4_per_s2())?[0];
        let selected = hydraulic(checked::<R, _>(client, "mirror momentum point", |a| {
            Ok(a.expansion_interval(result.momentum_flux_m4_per_s2())?)
        }))?;
        let selected = frozen::endpoints(&selected)?[0];
        assert!(
            bound[0] <= selected[0] && selected[1] <= bound[1],
            "{label} momentum {selected:?} outside {bound:?}"
        );
    }
    Ok(())
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
