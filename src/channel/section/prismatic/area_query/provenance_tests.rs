use super::tests::test_support::{Client, Result, batch, hydraulic};
use super::{
    AreaAnchor, AreaDirection, AreaQuery, endpoints::PhysicalEndpoint,
    source_difference::source_difference,
};
use crate::channel::{
    flow::surveyed::error::{self, HydraulicError, HydraulicResult},
    section::{CrossSectionBatch, CrossSectionError, HydraulicGeometry, PrismaticRecoveryConfig},
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

fn rejects_foreign_model<T>(result: HydraulicResult<T>, label: &str) {
    assert!(
        matches!(result,
            Err(HydraulicError::Geometry(CrossSectionError::InvalidInput { ref context, .. }))
                if context.contains("different compiled models")
        ),
        "{label}"
    );
}

fn provenance<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = batch::<R>(&client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let same_geometry = HydraulicGeometry::from_batch(&section);
    let equal = batch::<R>(&client)?;
    assert_eq!(section.definitions(), equal.definitions());
    let mut definitions = section.definitions().to_vec();
    for definition in &mut definitions {
        for point in &mut definition.points {
            point.station_m *= 2.0;
        }
    }
    let wider = CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?;
    let owners = Tensor::<R>::from_slice(&[1_i64, 2, 1], &[3], client.device())?;
    let area_tensor = Tensor::from_slice(&[1.0; 3], &[3], client.device())?;
    let areas = hydraulic(error::checked::<R, _>(
        &client,
        "provenance exact areas",
        |a| Ok(a.expansion_point(&area_tensor)?),
    ))?;
    let source =
        section.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;

    for foreign_section in [&wider, &equal] {
        let foreign_geometry = HydraulicGeometry::from_batch(foreign_section);
        let foreign_source = foreign_section.recover_area_points_tensor(
            &owners,
            &areas,
            &PrismaticRecoveryConfig::default(),
        )?;
        hydraulic(error::checked::<R, _>(
            &client,
            "compiled model provenance",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let cloned_bands = same_geometry.gather(a, &owners)?;
                let foreign_bands = foreign_geometry.gather(a, &owners)?;
                assert_eq!(
                    bands.section_owner.try_to_vec::<i64>()?,
                    foreign_bands.section_owner.try_to_vec::<i64>()?
                );
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let foreign_boundaries = foreign_geometry.boundary_areas(a)?.gather(a, &owners)?;
                let zero = Tensor::zeros(&[3], DType::F64, client.device())?;
                let scale = a.expansion_point(&zero)?;
                let parameter = a.point(&zero)?;
                let foreign_anchor = AreaAnchor::Source {
                    area: Box::new(a.expansion_bounds(
                        foreign_source.requested_areas_m2().high(),
                        foreign_source.requested_areas_m2().low(),
                        foreign_source.requested_areas_m2().abs_error(),
                    )?),
                    coordinates: Box::new(super::selection::coordinates(
                        a,
                        foreign_source.band_coordinates(),
                        None,
                    )?),
                };
                assert!(matches!(
                    AreaQuery::band_affine(
                        a,
                        &bands,
                        foreign_anchor,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    ),
                    Err(HydraulicError::Geometry(CrossSectionError::InvalidInput { ref context, .. }))
                        if context.contains("Use source_affine")
                ));
                rejects_foreign_model(
                    AreaQuery::source_affine(
                        a,
                        &bands,
                        &foreign_source,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    ),
                    "source-affine foreign recovery",
                );

                let query = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?;
                rejects_foreign_model(
                    query.prove_support(a, &foreign_bands),
                    "support foreign bands",
                );
                let query = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &cloned_bands)?;
                let foreign_query = AreaQuery::source_affine(
                    a,
                    &foreign_bands,
                    &foreign_source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &foreign_bands)?;
                rejects_foreign_model(
                    geometry.recover_query(a, &bands, &foreign_query),
                    "recovery foreign query",
                );
                rejects_foreign_model(
                    geometry.recover_query(a, &foreign_bands, &query),
                    "recovery foreign bands",
                );
                rejects_foreign_model(
                    foreign_geometry.recover_query(a, &bands, &query),
                    "recovery foreign receiver",
                );
                rejects_foreign_model(
                    query.endpoint(a, &foreign_bands, PhysicalEndpoint::Lower),
                    "endpoint foreign bands",
                );
                rejects_foreign_model(
                    source_difference(a, &foreign_bands, &query, &source, &boundaries),
                    "difference foreign bands",
                );
                rejects_foreign_model(
                    source_difference(a, &bands, &foreign_query, &source, &boundaries),
                    "difference foreign query",
                );
                rejects_foreign_model(
                    source_difference(a, &bands, &query, &foreign_source, &boundaries),
                    "difference foreign recovery",
                );
                rejects_foreign_model(
                    source_difference(a, &bands, &query, &source, &foreign_boundaries),
                    "difference foreign witnesses",
                );

                same_geometry.recover_query(a, &cloned_bands, &query)?;
                source_difference(a, &bands, &query, &source, &boundaries)?;
                let selected_rows = Tensor::from_slice(&[2_i64, 1, 2, 0], &[4], client.device())?;
                let selected_bands = bands.select(a, &selected_rows)?;
                let selected = query.select(a, &selected_rows)?;
                assert_eq!(
                    selected.query.original_rows.try_to_vec::<i64>()?,
                    [2, 1, 2, 0]
                );
                let selected_boundaries = boundaries.select(a, &selected_rows)?;
                assert!(selected_boundaries.same_model(&boundaries));
                for physical in [PhysicalEndpoint::Lower, PhysicalEndpoint::Upper] {
                    let endpoint = selected.endpoint(a, &selected_bands, physical)?;
                    same_geometry.recover_query(a, &selected_bands, &endpoint)?;
                    assert_eq!(
                        endpoint.query.original_rows.try_to_vec::<i64>()?,
                        [2, 1, 2, 0]
                    );
                }

                let wrong_owners = Tensor::from_slice(&[2_i64, 1, 2], &[3], client.device())?;
                let wrong_bands = geometry.gather(a, &wrong_owners)?;
                assert!(
                    matches!(query.endpoint(a, &wrong_bands, PhysicalEndpoint::Lower), Err(HydraulicError::Geometry(CrossSectionError::InvalidInput { ref context, .. })) if context.contains("owners differ"))
                );
                client.copy_into(
                    &query.query.owners,
                    &Tensor::zeros(&[3], DType::I64, client.device())?,
                )?;
                client.copy_into(
                    query.query.scale.high(),
                    &Tensor::ones(&[3], DType::F64, client.device())?,
                )?;
                client.copy_into(
                    query.query.parameter.lower(),
                    &Tensor::ones(&[3], DType::F64, client.device())?,
                )?;
                assert_eq!(selected.query.owners.try_to_vec::<i64>()?, [1, 2, 1, 1]);
                assert_eq!(selected.query.scale.high().try_to_vec::<f64>()?, [0.0; 4]);
                assert_eq!(
                    selected.query.parameter.lower().try_to_vec::<f64>()?,
                    [0.0; 4]
                );
                same_geometry.recover_query(a, &selected_bands, &selected)?;
                Ok(())
            },
        ))?;
    }
    Ok(())
}

#[test]
fn cpu_compiled_model_provenance_preserves_owned_query_rows() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    provenance::<CpuRuntime>(CpuClient::new(CpuDevice::new()))
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_compiled_model_provenance_preserves_owned_query_rows() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    provenance::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
