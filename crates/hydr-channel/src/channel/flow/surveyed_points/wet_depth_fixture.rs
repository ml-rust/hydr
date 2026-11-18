use crate::channel::{
    flow::{
        surveyed::PrismaticRiemannConfig,
        surveyed::work::WorkLedger,
        surveyed_points::{
            chart_fixture,
            paired_sources::PairedSources,
            source_fixture_support::{self as frozen, Client, Result, components, expansion},
            wet_root_types::WetRootDomain,
        },
    },
    section::{PrismaticSectionBatch, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedBisectionAlgorithms, CertifiedMappedBisectionAlgorithms,
        CertifiedUpperBracketAlgorithms,
    },
};

pub(in crate::channel::flow::surveyed_points) trait DepthClient<R: Runtime<DType = DType>>:
    Client<R>
    + CertifiedSimpsonAlgorithms<R>
    + CertifiedBisectionAlgorithms<R>
    + CertifiedMappedBisectionAlgorithms<R>
    + CertifiedUpperBracketAlgorithms<R>
{
}

impl<R, C> DepthClient<R> for C
where
    R: Runtime<DType = DType>,
    C: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
}

pub(in crate::channel::flow::surveyed_points) fn rows<R>(
    client: &R::Client,
    values: &[i64],
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    Ok(Tensor::from_slice(
        values,
        &[values.len()],
        client.device(),
    )?)
}

pub(in crate::channel::flow::surveyed_points) fn with_section<R, T>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    states: &[[f64; 4]],
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(&mut PairedSources<'_, R>, &WorkLedger) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    with_section_policy(client, section, states, config, config, |pair, _, work| {
        operation(pair, work)
    })
}

fn with_section_policy<R, T>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    states: &[[f64; 4]],
    config: &PrismaticRiemannConfig,
    replacement: &PrismaticRiemannConfig,
    operation: impl for<'a> FnOnce(
        &mut PairedSources<'a, R>,
        &'a PrismaticRiemannConfig,
        &WorkLedger,
    ) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let owners = rows::<R>(client, &vec![0; states.len()])?;
    let inputs = (0..4)
        .map(|field| {
            expansion::<R>(
                client,
                &states
                    .iter()
                    .map(|state| [state[field].to_bits(), 0, 0])
                    .collect::<Vec<_>>(),
                true,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    chart_fixture::with_inputs_ref_policy(
        client,
        section,
        owners,
        inputs,
        config,
        replacement,
        operation,
    )
}

pub(in crate::channel::flow::surveyed_points) fn with_rectangle<R, T>(
    client: &R::Client,
    width: f64,
    cap_depth: f64,
    states: &[[f64; 4]],
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(&mut PairedSources<'_, R>, &WorkLedger) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let section = chart_fixture::rectangle::<R>(client, width, cap_depth)?;
    with_section(client, &section, states, config, operation)
}

pub(super) fn with_rectangle_policy<R, T>(
    client: &R::Client,
    width: f64,
    cap_depth: f64,
    states: &[[f64; 4]],
    config: &PrismaticRiemannConfig,
    replacement: &PrismaticRiemannConfig,
    operation: impl for<'a> FnOnce(
        &mut PairedSources<'a, R>,
        &'a PrismaticRiemannConfig,
        &WorkLedger,
    ) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let section = chart_fixture::rectangle::<R>(client, width, cap_depth)?;
    with_section_policy(client, &section, states, config, replacement, operation)
}

pub(super) fn with_pointed_film<R, T>(
    client: &R::Client,
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(&mut PairedSources<'_, R>, &WorkLedger) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let section = frozen::batch::<R>(client)?;
    let owner = frozen::cases()
        .iter()
        .find(|case| case.id == "scalar_triangle_uniform_flow")
        .ok_or("Missing independent pointed profile")?
        .owner;
    let owners = rows::<R>(client, &[i64::try_from(owner)?])?;
    let inputs = [
        [1.0_f64.to_bits(), 0, 0],
        [(-4.0_f64).to_bits(), 0, 0],
        [1.0_f64.to_bits(), 0, 0],
        [4.0_f64.to_bits(), (-2.0_f64.powi(-80)).to_bits(), 0],
    ]
    .iter()
    .map(|value| expansion::<R>(client, &[*value], true))
    .collect::<Result<Vec<_>>>()?;
    chart_fixture::with_inputs(client, section, owners, inputs, config, operation)
}

pub(super) fn area_bounds<R>(
    pair: &PairedSources<'_, R>,
    parents: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
) -> Result<Vec<[f64; 2]>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let [lower, upper] = chart_fixture::endpoint_areas(pair, parents, query)?;
    let lower = frozen::bounds(pair.geometry.client(), &lower)?;
    let upper = frozen::bounds(pair.geometry.client(), &upper)?;
    Ok(lower
        .lower()
        .try_to_vec::<f64>()?
        .into_iter()
        .zip(upper.upper().try_to_vec::<f64>()?)
        .map(|(lower, upper)| [lower, upper])
        .collect())
}

#[derive(Debug, PartialEq)]
pub(super) struct DomainSnapshot {
    parents: Vec<i64>,
    owners: Vec<i64>,
    diagnostics: Vec<i64>,
    parameter: Vec<[u64; 2]>,
    scale: Vec<[u64; 3]>,
    lower_area: Vec<[u64; 3]>,
    upper_area: Vec<[u64; 3]>,
    lower_residual: Vec<[u64; 3]>,
    upper_residual: Vec<[u64; 3]>,
}

pub(super) fn snapshot<R>(
    pair: &PairedSources<'_, R>,
    domain: &WetRootDomain<R>,
) -> Result<DomainSnapshot>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let [lower, upper] = chart_fixture::endpoint_areas(pair, &domain.parent_rows, &domain.query)?;
    let parameter = frozen::endpoints(&domain.query.query.parameter)?;
    Ok(DomainSnapshot {
        parents: domain.parent_rows.try_to_vec::<i64>()?,
        owners: domain.invocation_owner.try_to_vec::<i64>()?,
        diagnostics: domain.query.query.original_rows.try_to_vec::<i64>()?,
        parameter: parameter
            .into_iter()
            .map(|pair| pair.map(f64::to_bits))
            .collect(),
        scale: components(&domain.query.query.scale)?,
        lower_area: components(&lower)?,
        upper_area: components(&upper)?,
        lower_residual: components(&domain.lower_retained)?,
        upper_residual: components(&domain.upper_retained)?,
    })
}
