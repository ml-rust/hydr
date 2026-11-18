use crate::channel::flow::surveyed_points::{
    paired_sources::PairedSources,
    point_sources::PointSourceParts,
    source_fixture_support::{
        self as support, Client, Result, components, endpoints, expansion, hydraulic,
    },
};
use crate::channel::{
    flow::surveyed::{PrismaticRiemannConfig, error, work::WorkLedger},
    section::{
        AreaAnchor, AreaDirection, AreaQuery, CrossSectionBatch, CrossSectionDefinition,
        CrossSectionPoint, PhysicalEndpoint, PrismaticSectionBatch, SupportedAreaQuery,
    },
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;
use solvr::{integrate::CertifiedSimpsonAlgorithms, optimize::CertifiedBisectionAlgorithms};

#[derive(Debug, PartialEq)]
pub(super) struct CacheSnapshot {
    pub(super) ready: Vec<f64>,
    retained: Vec<[u64; 3]>,
    physical: Vec<[f64; 2]>,
}

pub(super) fn cache<R: Runtime<DType = DType>>(
    pair: &PairedSources<'_, R>,
) -> Result<Vec<CacheSnapshot>> {
    pair.cache
        .iter()
        .map(|entry| {
            Ok(CacheSnapshot {
                ready: entry.ready.try_to_vec::<f64>()?,
                retained: components(&entry.value.retained)?,
                physical: endpoints(&entry.value.physical)?,
            })
        })
        .collect()
}

pub(super) fn with_pair<R, T>(
    client: &R::Client,
    ids: &[&str],
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(&mut PairedSources<'_, R>, &WorkLedger) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let cases = support::cases();
    let selected = ids
        .iter()
        .map(|id| {
            cases
                .iter()
                .find(|case| case.id == *id)
                .ok_or("Missing independently frozen root source")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let section = support::batch::<R>(client)?;
    let owners = Tensor::from_slice(
        &selected
            .iter()
            .map(|case| i64::try_from(case.owner))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        &[ids.len()],
        client.device(),
    )?;
    let inputs = (0..4)
        .map(|field| {
            expansion::<R>(
                client,
                &selected
                    .iter()
                    .map(|case| case.inputs[field])
                    .collect::<Vec<_>>(),
                true,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    with_inputs(client, section, owners, inputs, config, operation)
}

pub(super) fn with_inputs<R, T>(
    client: &R::Client,
    section: PrismaticSectionBatch<R>,
    owners: Tensor<R>,
    inputs: Vec<CertifiedExpansion<R>>,
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(&mut PairedSources<'_, R>, &WorkLedger) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    with_inputs_ref(client, &section, owners, inputs, config, operation)
}

pub(super) fn with_inputs_ref<R, T>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    owners: Tensor<R>,
    inputs: Vec<CertifiedExpansion<R>>,
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(&mut PairedSources<'_, R>, &WorkLedger) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    with_inputs_ref_policy(
        client,
        section,
        owners,
        inputs,
        config,
        config,
        |pair, _, work| operation(pair, work),
    )
}

pub(super) fn with_inputs_ref_policy<R, T>(
    client: &R::Client,
    section: &PrismaticSectionBatch<R>,
    owners: Tensor<R>,
    inputs: Vec<CertifiedExpansion<R>>,
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
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let parts = hydraulic(PointSourceParts::capture(
        section,
        &owners,
        [&inputs[0], &inputs[1], &inputs[2], &inputs[3]],
        config,
    ))?;
    let originals = Tensor::from_slice(
        &(0..owners.numel())
            .map(|index| i64::try_from(index).map(|index| 101 + index))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        &[owners.numel()],
        client.device(),
    )?;
    let work = WorkLedger::new(config);
    let mut pair = hydraulic(parts.pair(config, &work, &originals))?;
    operation(&mut pair, replacement, &work)
}

pub(super) fn rectangle<R>(
    client: &R::Client,
    width: f64,
    depth: f64,
) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definition = CrossSectionDefinition {
        id: "declared_rectangle".into(),
        vertical_datum: "Exact rectangle datum".into(),
        points: [(0.0, depth), (0.0, 0.0), (width, 0.0), (width, depth)]
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: depth,
        profile_provenance: "Independent dyadic or exact integer proof".into(),
    };
    Ok(CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?)
}

pub(super) fn uncertain_pair<R, T>(
    client: &R::Client,
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(&mut PairedSources<'_, R>, &WorkLedger) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let section = rectangle::<R>(client, 9.0, 1.0)?;
    let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
    let inputs = [9.0_f64, 0.0, 4.0, 8.0]
        .into_iter()
        .map(|value| expansion::<R>(client, &[[value.to_bits(), 0, 0]], true))
        .collect::<Result<Vec<_>>>()?;
    with_inputs(client, section, owners, inputs, config, operation)
}

#[derive(Clone, Copy)]
pub(super) enum Anchor {
    Source,
    Zero,
    Cap,
}

pub(super) fn chart<R>(
    pair: &PairedSources<'_, R>,
    anchor: Anchor,
    scale: [u64; 2],
    parameters: &[[f64; 2]],
) -> Result<SupportedAreaQuery<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let client = pair.geometry.client();
    hydraulic(error::checked::<R, _>(
        client,
        "independent root chart",
        |a| {
            let n = parameters.len();
            let high =
                Tensor::from_slice(&vec![f64::from_bits(scale[0]); n], &[n], client.device())?;
            let low =
                Tensor::from_slice(&vec![f64::from_bits(scale[1]); n], &[n], client.device())?;
            let zero = Tensor::zeros(&[n], DType::F64, client.device())?;
            let scale = a.expansion_bounds(&high, &low, &zero)?;
            let lower = Tensor::from_slice(
                &parameters.iter().map(|p| p[0]).collect::<Vec<_>>(),
                &[n],
                client.device(),
            )?;
            let upper = Tensor::from_slice(
                &parameters.iter().map(|p| p[1]).collect::<Vec<_>>(),
                &[n],
                client.device(),
            )?;
            let parameter = a.bounds(&lower, &upper)?;
            let query = match anchor {
                Anchor::Source => AreaQuery::source_affine(
                    a,
                    pair.bands,
                    pair.sources[0],
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?,
                Anchor::Zero => AreaQuery::band_affine(
                    a,
                    pair.bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?,
                Anchor::Cap => AreaQuery::cap_deficit(a, pair.bands, &scale, &parameter)?,
            };
            client.copy_into(&query.original_rows, &pair.original_rows)?;
            query.prove_support(a, pair.bands)
        },
    ))
}

pub(super) fn select<R>(
    pair: &PairedSources<'_, R>,
    query: &SupportedAreaQuery<R>,
    rows: &Tensor<R>,
) -> Result<SupportedAreaQuery<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    hydraulic(error::checked::<R, _>(
        pair.geometry.client(),
        "root invocation selection",
        |a| query.select(a, rows),
    ))
}

pub(super) fn endpoint_areas<R>(
    pair: &PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
) -> Result<[CertifiedExpansion<R>; 2]>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    hydraulic(error::checked::<R, _>(
        pair.geometry.client(),
        "returned chart retained physical expressions",
        |a| {
            let bands = pair.bands.select(a, rows)?;
            let boundaries = pair.boundaries.select(a, rows)?;
            let lower = query.endpoint(a, &bands, PhysicalEndpoint::Lower)?;
            let upper = query.endpoint(a, &bands, PhysicalEndpoint::Upper)?;
            Ok([
                lower.retained_endpoint_area(a, &bands, &boundaries)?,
                upper.retained_endpoint_area(a, &bands, &boundaries)?,
            ])
        },
    ))
}

pub(super) fn config() -> PrismaticRiemannConfig {
    PrismaticRiemannConfig {
        gravity_m_per_s2: 1.0,
        ..Default::default()
    }
}
