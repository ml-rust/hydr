use crate::channel::flow::surveyed_points::{chart_test_support, paired_sources::PairedSources};
use crate::channel::flow::{
    surveyed::{PrismaticRiemannConfig, error, work::WorkLedger},
    surveyed_points::source_tests::support::{
        self, Client, Result, components, expansion, hydraulic,
    },
};
use crate::channel::section::SupportedAreaQuery;
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{integrate::CertifiedSimpsonAlgorithms, optimize::CertifiedBisectionAlgorithms};

pub(super) fn source_bits<R>(pair: &PairedSources<'_, R>) -> Result<Vec<Vec<[u64; 3]>>>
where
    R: Runtime<DType = DType>,
{
    let input = pair.velocity.input();
    [
        &input.left_area,
        &input.left_discharge,
        &input.right_area,
        &input.right_discharge,
        &pair.velocity.left,
        &pair.velocity.right,
        &pair.velocity.difference,
    ]
    .into_iter()
    .map(components)
    .collect()
}

pub(super) fn physical_bounds<R>(
    pair: &PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
) -> Result<Vec<[f64; 2]>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let [lower, upper] = chart_test_support::endpoint_areas(pair, rows, query)?;
    hydraulic(error::checked::<R, _>(
        pair.geometry.client(),
        "locator retained physical domain bounds",
        |a| {
            let lower = a.expansion_interval(&lower)?.lower().try_to_vec::<f64>()?;
            let upper = a.expansion_interval(&upper)?.upper().try_to_vec::<f64>()?;
            Ok(lower
                .into_iter()
                .zip(upper)
                .map(|(lo, hi)| [lo, hi])
                .collect())
        },
    ))
}

pub(super) fn star_contains(actual: [f64; 2], reference: [u64; 2], id: &str) {
    let [lo, hi] = reference.map(f64::from_bits);
    assert!(
        actual[0] <= hi && lo <= actual[1],
        "{id}: {actual:?}, star [{lo:e}, {hi:e}]"
    );
    assert!(
        actual[0] >= 0.0 && actual[0] <= actual[1],
        "{id}: {actual:?}"
    );
}

pub(super) fn cap_deficit_contains<R>(
    pair: &PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let [lower, upper] = chart_test_support::endpoint_areas(pair, rows, query)?;
    let client = pair.geometry.client();
    let truth = expansion::<R>(client, &[[14.0_f64.to_bits(), 0xbaf0000000000000, 0]], true)?;
    hydraulic(error::checked::<R, _>(
        client,
        "locator cap deficit retained physical difference",
        |a| {
            let lower = a.expansion_sub(&lower, &truth)?;
            let upper = a.expansion_sub(&upper, &truth)?;
            assert!(a.expansion_interval(&lower)?.lower().item::<f64>()? <= 0.0);
            assert!(a.expansion_interval(&upper)?.upper().item::<f64>()? >= 0.0);
            Ok(())
        },
    ))
}

pub(super) fn pointed_film<R, T>(
    client: &R::Client,
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(&mut PairedSources<'_, R>, &WorkLedger) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let section = support::batch::<R>(client)?;
    let owner = support::cases()
        .iter()
        .find(|case| case.id == "scalar_triangle_uniform_flow")
        .ok_or("Missing independently declared triangle profile")?
        .owner;
    let owners = Tensor::from_slice(&[i64::try_from(owner)?], &[1], client.device())?;
    let inputs = [
        [1.0_f64.to_bits(), 0, 0],
        [(-4.0_f64).to_bits(), 0, 0],
        [1.0_f64.to_bits(), 0, 0],
        [4.0_f64.to_bits(), 0xbaf0000000000000, 0],
    ]
    .into_iter()
    .map(|value| expansion::<R>(client, &[value], true))
    .collect::<Result<Vec<_>>>()?;
    chart_test_support::with_inputs(client, section, owners, inputs, config, operation)
}

pub(super) fn uncertain_dry_limit<R, T>(
    client: &R::Client,
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(&mut PairedSources<'_, R>, &WorkLedger) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let section = chart_test_support::rectangle::<R>(client, 9.0, 1.0)?;
    let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
    let inputs = [
        9.0_f64.to_bits(),
        (-3.0_f64).to_bits(),
        4.0_f64.to_bits(),
        12.0_f64.to_bits(),
    ]
    .into_iter()
    .map(|high| expansion::<R>(client, &[[high, 0, 0]], true))
    .collect::<Result<Vec<_>>>()?;
    chart_test_support::with_inputs(client, section, owners, inputs, config, operation)
}
