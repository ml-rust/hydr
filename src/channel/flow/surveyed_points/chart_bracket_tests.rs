use super::{ChartBrackets, bracket_chart};
use crate::channel::flow::{
    surveyed::error,
    surveyed_points::{
        chart_roots::solve_chart,
        chart_test_support as support,
        paired_sources::PairedSources,
        potential_range_tests::frozen::original_film,
        source_tests::support::{Client, Result, hydraulic},
    },
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms, CertifiedUpperBracketStatus,
    },
};

#[path = "chart_bracket_tests_controls.rs"]
mod controls;

fn parameters<R: Runtime<DType = DType>>(bracket: &ChartBrackets<R>) -> Result {
    let owners = bracket.bracket_owner.try_to_vec::<i64>()?;
    let lower = bracket.result.lower().try_to_vec::<f64>()?;
    let upper = bracket.result.upper().try_to_vec::<f64>()?;
    let selected_lower = owners
        .iter()
        .map(|&row| usize::try_from(row).map(|row| lower[row]))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let selected_upper = owners
        .iter()
        .map(|&row| usize::try_from(row).map(|row| upper[row]))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    assert_eq!(
        bracket.query.query.parameter.lower().try_to_vec::<f64>()?,
        selected_lower
    );
    assert_eq!(
        bracket.query.query.parameter.upper().try_to_vec::<f64>()?,
        selected_upper
    );
    Ok(())
}

fn contains<R>(pair: &PairedSources<'_, R>, bracket: &ChartBrackets<R>, truth: f64) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    parameters(bracket)?;
    let [lower, upper] = support::endpoint_areas(pair, &bracket.bracket_rows, &bracket.query)?;
    hydraulic(error::checked::<R, _>(
        pair.geometry.client(),
        "retained bracket physical containment",
        |a| {
            assert!(a.expansion_interval(&lower)?.lower().item::<f64>()? <= truth);
            assert!(a.expansion_interval(&upper)?.upper().item::<f64>()? >= truth);
            Ok(())
        },
    ))
}

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = support::config();
    support::with_pair::<R, _>(
        client,
        &[
            "scalar_rectangle_uniform_rest",
            "scalar_triangle_uniform_flow",
            "scalar_rectangle_uniform_rest",
        ],
        &config,
        |pair, work| {
            let full = support::chart(
                pair,
                support::Anchor::Source,
                [1.0_f64.to_bits(), 0],
                &[[-0.25, 0.25], [0.25, 0.5], [-0.25, 0.25]],
            )?;
            let parents = Tensor::from_slice(&[2_i64, 0, 2, 1], &[4], client.device())?;
            let chart = support::select(pair, &full, &parents)?;
            let seed = Tensor::from_slice(
                &[9.0, -0.125, 9.0, -0.125, 9.0, -0.125, 9.0, 0.375],
                &[4, 2],
                client.device(),
            )?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
            assert!(seed.offset() > 0 && !seed.is_contiguous());
            let bracket = hydraulic(bracket_chart(
                pair,
                &parents,
                &chart,
                &seed,
                config.root.max_points,
            ))?;
            let found = CertifiedUpperBracketStatus::Bracketed as i64;
            let absent = CertifiedUpperBracketStatus::NoRoot as i64;
            assert_eq!(
                bracket.result.status().try_to_vec::<i64>()?,
                [found, found, found, absent]
            );
            assert_eq!(bracket.invocation_rows.try_to_vec::<i64>()?, [2, 0, 2, 1]);
            assert_eq!(bracket.bracket_owner.try_to_vec::<i64>()?, [0, 1, 2]);
            assert_eq!(bracket.bracket_rows.try_to_vec::<i64>()?, [2, 0, 2]);
            assert_eq!(
                bracket.query.query.original_rows.try_to_vec::<i64>()?,
                [103, 101, 103]
            );
            parameters(&bracket)?;
            assert!(bracket.result.expansion_rounds() > 0);
            assert_eq!(work.snapshot().bracket_solves, 1);
            assert_eq!(
                work.snapshot().bracket_expansion_rounds,
                bracket.result.expansion_rounds()
            );
            assert_eq!(
                work.snapshot().residual_points,
                bracket.result.callback_points()
            );
            let endpoints = bracket.result.upper().try_to_vec::<f64>()?;
            client.copy_into(&parents, &Tensor::zeros(&[4], DType::I64, client.device())?)?;
            client.copy_into(&seed, &Tensor::zeros(&[4], DType::F64, client.device())?)?;
            assert_eq!(bracket.invocation_rows.try_to_vec::<i64>()?, [2, 0, 2, 1]);
            assert_eq!(bracket.result.upper().try_to_vec::<f64>()?, endpoints);
            Ok(())
        },
    )?;
    stationary::<R>(client)?;
    film::<R>(client)?;
    deficit::<R>(client)?;
    controls::run::<R>(client)
}

fn stationary<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let mut config = support::config();
    config.gravity_m_per_s2 = 3.0;
    support::with_pair::<R, _>(
        client,
        &["scalar_rectangle_stationary_shock"],
        &config,
        |pair, work| {
            let chart = support::chart(
                pair,
                support::Anchor::Zero,
                [1.0_f64.to_bits(), 0],
                &[[1.0, 3.0]],
            )?;
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let seed = Tensor::from_slice(&[1.25], &[1], client.device())?;
            let bracket = hydraulic(bracket_chart(
                pair,
                &rows,
                &chart,
                &seed,
                config.root.max_points,
            ))?;
            assert_eq!(
                bracket.result.status().try_to_vec::<i64>()?,
                [CertifiedUpperBracketStatus::Bracketed as i64]
            );
            assert!(bracket.result.expansion_rounds() > 0);
            contains(pair, &bracket, 2.0)?;
            assert_eq!(
                work.snapshot().residual_points,
                bracket.result.callback_points()
            );
            Ok(())
        },
    )
}

fn film<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = support::config();
    support::with_pair::<R, _>(
        client,
        &["retained_vacuum_inside_film"],
        &config,
        |pair, work| {
            let film = &original_film::SONIC_BRACKET;
            let limit = f64::from_bits(film.parameter_bounds[0]);
            let chart = support::chart(
                pair,
                support::Anchor::Zero,
                film.scale_components,
                &[[0.0, limit]],
            )?;
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let seed = Tensor::from_slice(&[limit / 32.0], &[1], client.device())?;
            let bracket = hydraulic(bracket_chart(
                pair,
                &rows,
                &chart,
                &seed,
                config.root.max_points,
            ))?;
            assert_eq!(
                bracket.result.status().try_to_vec::<i64>()?,
                [CertifiedUpperBracketStatus::Bracketed as i64]
            );
            assert!(bracket.result.expansion_rounds() > 0);
            let star = f64::from_bits(original_film::STAR.area_bounds[0]);
            contains(pair, &bracket, star)?;
            let remaining = config
                .root
                .max_points
                .checked_sub(bracket.result.callback_points())
                .ok_or("Bracket callback count exceeds the configured budget")?;
            let roots = hydraulic(solve_chart(
                pair,
                &bracket.bracket_rows,
                &bracket.query,
                remaining,
            ))?;
            let [lower, upper] = support::endpoint_areas(pair, &roots.root_rows, &roots.query)?;
            hydraulic(error::checked::<R, _>(
                client,
                "film bracket and root physical precision",
                |a| {
                    let lo = a.expansion_interval(&lower)?.lower().item::<f64>()?;
                    let hi = a.expansion_interval(&upper)?.upper().item::<f64>()?;
                    assert!(lo <= star && star <= hi);
                    let gate = f64::from_bits(0x3ddb7cdfd9d7bdba);
                    assert!((hi - lo) / star <= gate);
                    assert!((star - lo).max(hi - star) / star <= gate);
                    Ok(())
                },
            ))?;
            assert_eq!(
                work.snapshot().residual_points,
                bracket.result.callback_points() + roots.result.evaluated_points()
            );
            Ok(())
        },
    )
}

fn deficit<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = support::config();
    support::with_pair::<R, _>(
        client,
        &["uniform_cap_retained_deficit_dyadic_velocity"],
        &config,
        |pair, _| {
            let deficit = f64::from_bits(0x3af0000000000000);
            let chart = support::chart(
                pair,
                support::Anchor::Cap,
                [1.0_f64.to_bits(), 0],
                &[[0.0, 2.0 * deficit]],
            )?;
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let seed = Tensor::from_slice(&[deficit / 4.0], &[1], client.device())?;
            let bracket = hydraulic(bracket_chart(
                pair,
                &rows,
                &chart,
                &seed,
                config.root.max_points,
            ))?;
            assert_eq!(
                bracket.result.status().try_to_vec::<i64>()?,
                [CertifiedUpperBracketStatus::Bracketed as i64]
            );
            assert!(bracket.result.expansion_rounds() > 0);
            parameters(&bracket)?;
            let [lower, upper] =
                support::endpoint_areas(pair, &bracket.bracket_rows, &bracket.query)?;
            hydraulic(error::checked::<R, _>(
                client,
                "retained deficit bracket physical endpoints",
                |a| {
                    let high = Tensor::from_slice(&[14.0], &[1], client.device())?;
                    let low = Tensor::from_slice(&[-deficit], &[1], client.device())?;
                    let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                    let truth = a.expansion_bounds(&high, &low, &zero)?;
                    let lo = a.expansion_sub(&lower, &truth)?;
                    let hi = a.expansion_sub(&upper, &truth)?;
                    assert!(a.expansion_interval(&lo)?.lower().item::<f64>()? <= 0.0);
                    assert!(a.expansion_interval(&hi)?.upper().item::<f64>()? >= 0.0);
                    Ok(())
                },
            ))
        },
    )
}

#[test]
fn configured_cpu_chart_brackets_preserve_domains_and_shared_work() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let base = CpuClient::new(CpuDevice::new());
    let before = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    matrix::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(base.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_chart_brackets_preserve_domains_and_shared_work() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
