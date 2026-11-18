use super::analytic_bottom::AnalyticSource;
use super::analytic_bottom_arithmetic::{AnalyticOperands, evaluate as evaluate_bottom};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult},
    section::ClippedBandEndpoints,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

#[derive(Clone, Copy)]
pub(super) enum EndpointSide {
    Source,
    Target,
}

pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    endpoints: &ClippedBandEndpoints<R>,
    side: EndpointSide,
    gravity: f64,
) -> HydraulicResult<AnalyticSource<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let endpoint = match side {
        EndpointSide::Source => endpoints.source(),
        EndpointSide::Target => endpoints.target(),
    };
    let raw = endpoint.retained_area();
    let available = endpoint.raw_available_mask();
    let zero = Tensor::zeros(raw.high().shape(), DType::F64, c.device())?;
    let one = Tensor::ones(raw.high().shape(), DType::F64, c.device())?;
    let safe_area = a.expansion_bounds(
        &c.where_cond(available, raw.high(), &zero)?,
        &c.where_cond(available, raw.low(), &zero)?,
        &c.where_cond(available, raw.abs_error(), &zero)?,
    )?;
    let raw_bounds = a.expansion_interval(&safe_area)?;
    let rectangular = c.mul(endpoints.rectangular_band0_mask(), available)?;
    let rectangular = c.mul(&rectangular, &c.ge(raw_bounds.lower(), &zero)?)?;
    let pointed = c.mul(endpoints.pointed_band0_mask(), available)?;
    let eligible = c.maximum(&rectangular, &pointed)?;
    let local_depth = endpoint.coordinates().depth_above_start_m();
    let local_depth = a.bounds(
        &local_depth.lower().reshape(raw.high().shape())?,
        &local_depth.upper().reshape(raw.high().shape())?,
    )?;
    evaluate_bottom(
        a,
        AnalyticOperands {
            eligible,
            rectangular,
            pointed,
            exact_area: &safe_area,
            complete_depth: &local_depth,
            width: endpoints.compiled_bottom_width(),
            depth_identity: None,
            zero: &zero,
            one: &one,
            gravity,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::super::potential_analytic;
    use super::super::source_tests::support::{
        Client, Result, bounds, components, endpoints, hydraulic, zero,
    };
    use super::{self as analytic_endpoint, EndpointSide};
    use crate::channel::{
        flow::surveyed::error,
        section::{
            AreaAnchor, AreaDirection, AreaQuery, ConditionalBandSegments, CrossSectionBatch,
            CrossSectionDefinition, CrossSectionPoint, HydraulicGeometry, PrismaticRecoveryConfig,
            PrismaticSectionBatch,
        },
    };
    use numr::{
        dtype::DType,
        runtime::{Runtime, RuntimeClient},
        tensor::Tensor,
    };
    use solvr::common::arithmetic::CertifiedExpansion;

    fn snapshot<R: Runtime<DType = DType>>(
        value: &CertifiedExpansion<R>,
    ) -> error::HydraulicResult<Vec<[u64; 3]>> {
        components(value).map_err(|error| error::HydraulicError::Numerical {
            context: format!("Analytic endpoint snapshot rejected its tensors: {error}"),
        })
    }

    fn section<R>(client: &R::Client) -> Result<PrismaticSectionBatch<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let profiles: [(&str, &[(f64, f64)]); 3] = [
            (
                "rectangle",
                &[(0.0, 3.0), (0.0, 0.0), (2.0, 0.0), (2.0, 3.0)],
            ),
            ("pointed", &[(-3.0, 3.0), (0.0, 0.0), (3.0, 3.0)]),
            (
                "rectangle with widening banks",
                &[
                    (-2.0, 3.0),
                    (0.0, 1.0),
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (2.0, 1.0),
                    (4.0, 3.0),
                ],
            ),
        ];
        let definitions = profiles
            .into_iter()
            .map(|(id, coordinates)| CrossSectionDefinition {
                id: id.into(),
                vertical_datum: "Analytic endpoint datum".into(),
                points: coordinates
                    .iter()
                    .map(|&(station_m, elevation_m)| CrossSectionPoint {
                        station_m,
                        elevation_m,
                    })
                    .collect(),
                maximum_stage_m: 3.0,
                profile_provenance: "Exact dyadic analytic endpoint controls".into(),
            })
            .collect();
        Ok(CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?)
    }

    fn dry_traversals<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        let table = hydraulic(error::checked::<R, _>(
            client,
            "analytic dry traversal boundaries",
            |a| geometry.boundary_areas(a),
        ))?;
        let owners = Tensor::from_slice(&[0_i64, 1], &[2], client.device())?;
        for reverse in [false, true] {
            let source_value = if reverse { 1.0 } else { 0.0 };
            let target_value = if reverse { 0.0 } else { 1.0 };
            let areas = hydraulic(error::checked::<R, _>(client, "dry endpoint source", |a| {
                Ok(a.expansion_point(&Tensor::from_slice(
                    &[source_value; 2],
                    &[2],
                    client.device(),
                )?)?)
            }))?;
            let source = section.recover_area_points_tensor(
                &owners,
                &areas,
                &PrismaticRecoveryConfig::default(),
            )?;
            hydraulic(error::checked::<R, _>(
                client,
                "analytic dry endpoint traversal",
                |a| {
                    let bands = geometry.gather(a, &owners)?;
                    let boundaries = table.gather(a, &owners)?;
                    let scale = a.expansion_point(&Tensor::from_slice(
                        &[target_value; 2],
                        &[2],
                        client.device(),
                    )?)?;
                    let parameter = a.point(&Tensor::ones(&[2], DType::F64, client.device())?)?;
                    let target = AreaQuery::band_affine(
                        a,
                        &bands,
                        AreaAnchor::Zero,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    )?
                    .prove_support(a, &bands)?;
                    let segments =
                        ConditionalBandSegments::prepare(a, &bands, &source, &target, &boundaries)?;
                    let b = section.maximum_band_count();
                    let rows = Tensor::from_slice(&[b as i64, 0, b as i64], &[3], client.device())?;
                    let clipped = segments.clipped_endpoints(a, &rows)?;
                    let left = analytic_endpoint::evaluate(a, &clipped, EndpointSide::Source, 2.0)?;
                    let right =
                        analytic_endpoint::evaluate(a, &clipped, EndpointSide::Target, 2.0)?;
                    assert_eq!(left.eligible.try_to_vec::<f64>()?, [1.0; 3]);
                    assert_eq!(right.eligible.try_to_vec::<f64>()?, [1.0; 3]);
                    let selected = segments.select(a, &rows)?;
                    let difference = potential_analytic::difference(
                        a,
                        &left,
                        &right,
                        clipped.source().top_width(),
                        clipped.target().top_width(),
                        selected.retained_gap(),
                        2.0,
                    )?;
                    let expected = if reverse {
                        [-4.0_f64, -2.0, -4.0]
                    } else {
                        [4.0_f64, 2.0, 4.0]
                    };
                    let projected = a.expansion_interval(&difference.retained_increment)?;
                    let retained_bounds = [
                        projected.lower().try_to_vec::<f64>()?,
                        projected.upper().try_to_vec::<f64>()?,
                    ];
                    let physical_bounds = [
                        difference.physical_increment.lower().try_to_vec::<f64>()?,
                        difference.physical_increment.upper().try_to_vec::<f64>()?,
                    ];
                    for (index, (actual, expected)) in snapshot(&difference.retained_increment)?
                        .into_iter()
                        .zip(expected)
                        .enumerate()
                    {
                        let point = actual.map(f64::from_bits);
                        if index == 1 {
                            assert_eq!(point, [expected, 0.0, 0.0]);
                        } else {
                            // Pointed inverse depth and width certificates retain their rounding uncertainty.
                            assert!(point[2].is_finite() && point[2] > 0.0);
                            for bounds in [&retained_bounds, &physical_bounds] {
                                let [lower, upper] = [bounds[0][index], bounds[1][index]];
                                assert!(lower <= expected && expected <= upper);
                                let excess = (lower - expected).abs().max((upper - expected).abs());
                                assert!(excess <= 32.0 * f64::EPSILON * 4.0);
                            }
                        }
                    }
                    let dry = if reverse { &right } else { &left };
                    for value in [&dry.potential, &dry.sound, &dry.sound_squared] {
                        for point in snapshot(value)? {
                            zero(point, "structural dry analytic endpoint");
                        }
                    }
                    assert_eq!(
                        clipped
                            .compiled_bottom_width()
                            .lower()
                            .try_to_vec::<f64>()?,
                        [0.0, 2.0, 0.0]
                    );
                    Ok(())
                },
            ))?;
        }
        Ok(())
    }

    fn clipped_above<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        let owners = Tensor::from_slice(&[2_i64], &[1], client.device())?;
        let area = hydraulic(error::checked::<R, _>(client, "above-band source", |a| {
            Ok(a.expansion_point(&Tensor::from_slice(&[4.0], &[1], client.device())?)?)
        }))?;
        let source = section.recover_area_points_tensor(
            &owners,
            &area,
            &PrismaticRecoveryConfig::default(),
        )?;
        hydraulic(error::checked::<R, _>(
            client,
            "clipped analytic band-zero endpoint",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let scale =
                    a.expansion_point(&Tensor::from_slice(&[0.5], &[1], client.device())?)?;
                let parameter = a.point(&Tensor::ones(&[1], DType::F64, client.device())?)?;
                let target = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let segments =
                    ConditionalBandSegments::prepare(a, &bands, &source, &target, &boundaries)?;
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let selected = segments.select(a, &rows)?;
                let clipped = selected.clipped_endpoints(a, &rows)?;
                assert_eq!(
                    clipped
                        .source()
                        .retained_area()
                        .high()
                        .try_to_vec::<f64>()?,
                    [2.0]
                );
                let left = analytic_endpoint::evaluate(a, &clipped, EndpointSide::Source, 1.0)?;
                let right = analytic_endpoint::evaluate(a, &clipped, EndpointSide::Target, 1.0)?;
                assert_eq!(
                    snapshot(&left.potential)?[0].map(f64::from_bits),
                    [2.0, 0.0, 0.0]
                );
                assert_eq!(
                    snapshot(&right.potential)?[0].map(f64::from_bits),
                    [1.0, 0.0, 0.0]
                );
                let difference = potential_analytic::difference(
                    a,
                    &left,
                    &right,
                    clipped.source().top_width(),
                    clipped.target().top_width(),
                    selected.retained_gap(),
                    1.0,
                )?;
                assert_eq!(
                    snapshot(&difference.retained_increment)?[0].map(f64::from_bits),
                    [-1.0, 0.0, 0.0]
                );
                Ok(())
            },
        ))
    }

    fn retained_gap<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        let area = hydraulic(error::checked::<R, _>(client, "tiny-gap source", |a| {
            Ok(a.expansion_point(&Tensor::ones(&[1], DType::F64, client.device())?)?)
        }))?;
        let source = section.recover_area_points_tensor(
            &owners,
            &area,
            &PrismaticRecoveryConfig::default(),
        )?;
        for sign in [-1.0, 1.0] {
            let distance = sign * 2.0_f64.powi(-80);
            let result = hydraulic(error::checked::<R, _>(
                client,
                "retained analytic endpoint gap",
                |a| {
                    let bands = geometry.gather(a, &owners)?;
                    let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                    let scale = a.expansion_point(&Tensor::from_slice(
                        &[distance.abs()],
                        &[1],
                        client.device(),
                    )?)?;
                    let parameter =
                        a.point(&Tensor::from_slice(&[sign], &[1], client.device())?)?;
                    let target = AreaQuery::source_affine(
                        a,
                        &bands,
                        &source,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    )?
                    .prove_support(a, &bands)?;
                    let segments =
                        ConditionalBandSegments::prepare(a, &bands, &source, &target, &boundaries)?;
                    let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                    let selected = segments.select(a, &rows)?;
                    let clipped = selected.clipped_endpoints(a, &rows)?;
                    assert_eq!(
                        clipped
                            .source()
                            .retained_area()
                            .high()
                            .try_to_vec::<f64>()?,
                        [1.0]
                    );
                    assert_eq!(
                        clipped
                            .target()
                            .retained_area()
                            .high()
                            .try_to_vec::<f64>()?,
                        [1.0]
                    );
                    assert_eq!(
                        clipped.target().retained_area().low().try_to_vec::<f64>()?,
                        [distance]
                    );
                    let left = analytic_endpoint::evaluate(a, &clipped, EndpointSide::Source, 2.0)?;
                    let right =
                        analytic_endpoint::evaluate(a, &clipped, EndpointSide::Target, 2.0)?;
                    potential_analytic::difference(
                        a,
                        &left,
                        &right,
                        clipped.source().top_width(),
                        clipped.target().top_width(),
                        selected.retained_gap(),
                        2.0,
                    )
                },
            ))?;
            let actual = endpoints(&result.physical_increment)?[0];
            assert!(actual[0] <= distance && distance <= actual[1], "{actual:?}");
            assert!(
                (actual[0] - distance)
                    .abs()
                    .max((actual[1] - distance).abs())
                    <= distance.abs() * 1e-10
            );
            assert!(components(&result.retained_increment)?[0][2] != 0);
            let projected = endpoints(&bounds::<R>(client, &result.retained_increment)?)?[0];
            assert!(projected[0] <= distance && distance <= projected[1]);
        }
        Ok(())
    }

    fn uncertain_zero<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        hydraulic(error::checked::<R, _>(
            client,
            "uncertain zero analytic admission",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let scale = a.expansion_point(&Tensor::from_slice(
                    &[f64::from_bits(1)],
                    &[1],
                    client.device(),
                )?)?;
                let parameter = a.point(&Tensor::from_slice(&[0.5], &[1], client.device())?)?;
                let target = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let prefix = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &target,
                    &boundaries,
                )?;
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let clipped = prefix.segments.clipped_endpoints(a, &rows)?;
                assert_eq!(
                    clipped.target().exact_dry_mask().try_to_vec::<f64>()?,
                    [0.0]
                );
                assert!(clipped.target().retained_area().abs_error().item::<f64>()? > 0.0);
                let value = analytic_endpoint::evaluate(a, &clipped, EndpointSide::Target, 1.0)?;
                assert_eq!(value.eligible.try_to_vec::<f64>()?, [0.0]);
                for value in [&value.potential, &value.sound, &value.sound_squared] {
                    zero(
                        snapshot(value)?[0],
                        "unavailable uncertain-zero analytic endpoint",
                    );
                }
                Ok(())
            },
        ))
    }

    fn matrix<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = section::<R>(&client)?;
        dry_traversals::<R>(&client, &section)?;
        clipped_above::<R>(&client, &section)?;
        retained_gap::<R>(&client, &section)?;
        uncertain_zero::<R>(&client, &section)
    }

    #[test]
    fn configured_cpu_analytic_endpoints_preserve_clipped_geometry_and_retained_gaps() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        let retained = client.clone();
        matrix::<CpuRuntime>(client)?;
        assert_eq!(retained.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_analytic_endpoints_preserve_clipped_geometry_and_retained_gaps() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
