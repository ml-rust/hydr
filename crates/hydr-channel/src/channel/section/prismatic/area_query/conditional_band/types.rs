use super::super::super::{
    area_point_query::PrismaticAreaPointBandCoordinates, hydraulic_geometry::GatheredBands,
};
use super::endpoint_authority::StoredEndpointArea;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct LocalEndpoint<R: Runtime<DType = DType>> {
    pub above_start: Interval<R>,
    pub below_end: Interval<R>,
    pub at_start: Tensor<R>,
    pub at_end: Tensor<R>,
}

/// Each lane represents one clipped physical band, never an alternative inverse representation.
pub(crate) struct ConditionalBandSegments<R: Runtime<DType = DType>> {
    pub(super) rows: GatheredBands<R>,
    pub(super) original_rows: Tensor<R>,
    pub(super) band_index: Tensor<R>,
    pub(super) active: Tensor<R>,
    pub(super) zero_gap: Tensor<R>,
    pub(super) start: LocalEndpoint<R>,
    pub(super) end: LocalEndpoint<R>,
    pub(super) start_area: StoredEndpointArea<R>,
    pub(super) end_area: StoredEndpointArea<R>,
    pub(super) gap: CertifiedExpansion<R>,
    pub(super) physical_gap: Interval<R>,
    pub(super) endpoint_hull: Interval<R>,
}

pub(crate) struct PreparedAreaPrefix<R: Runtime<DType = DType>> {
    pub(crate) segments: ConditionalBandSegments<R>,
    pub(crate) endpoint_area: CertifiedExpansion<R>,
    pub(super) band_count: usize,
}

pub(crate) struct PrefixIntegrationGeometry<R: Runtime<DType = DType>> {
    pub(crate) face_count: usize,
    pub(crate) band_count: usize,
    pub(crate) section_owner: Tensor<R>,
    pub(crate) original_rows: Tensor<R>,
    pub(crate) band_index: Tensor<R>,
    pub(crate) valid: Tensor<R>,
    pub(crate) pointed: Tensor<R>,
    pub(crate) extent: Interval<R>,
    pub(crate) start_width: Interval<R>,
    pub(crate) slope: Interval<R>,
    pub(crate) start_area: Interval<R>,
}

pub(crate) struct ConditionalBandPanel<R: Runtime<DType = DType>> {
    pub(crate) area: Interval<R>,
    pub(crate) depth: Interval<R>,
    pub(crate) integral: Interval<R>,
    pub(crate) top_width: Interval<R>,
    pub(crate) hydraulic_depth: Interval<R>,
    pub(crate) coordinates: PrismaticAreaPointBandCoordinates<R>,
    pub(crate) gap: Interval<R>,
    pub(crate) active: Tensor<R>,
}

impl<R: Runtime<DType = DType>> ConditionalBandSegments<R> {
    pub(crate) fn owners(&self) -> &Tensor<R> {
        &self.rows.section_owner
    }
    pub(crate) fn original_rows(&self) -> &Tensor<R> {
        &self.original_rows
    }
    #[cfg(test)]
    pub(crate) fn band_indices(&self) -> &Tensor<R> {
        &self.band_index
    }
    pub(crate) fn active_mask(&self) -> &Tensor<R> {
        &self.active
    }
    pub(crate) fn zero_gap_mask(&self) -> &Tensor<R> {
        &self.zero_gap
    }
    pub(crate) fn retained_gap(&self) -> &CertifiedExpansion<R> {
        &self.gap
    }
    pub(crate) fn physical_gap(&self) -> &Interval<R> {
        &self.physical_gap
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        AreaDirection, AreaQuery, endpoints::PhysicalEndpoint, fixtures, test_support as support,
    };
    use super::super::preparation::origin::tests as tiny_gap_tests;
    use super::*;
    use crate::channel::{
        flow::surveyed::error::{self, HydraulicError, PrismaticRiemannRejection},
        section::{HydraulicGeometry, PrismaticRecoveryConfig},
    };
    use numr::{ops::BinaryOps, runtime::RuntimeClient};
    use support::{Client, Result, hydraulic};

    fn frozen_endpoints<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let table = hydraulic(error::checked::<R, _>(
            client,
            "conditional endpoint boundary table",
            |a| geometry.boundary_areas(a),
        ))?;
        for case in fixtures::CASES {
            let input = support::input::<R>(client, &section, &[case.chart], true, false)?;
            let areas = support::expansion::<R>(client, &[case.chart.source], true)?;
            let source = section.recover_area_points_tensor(
                &input.owners,
                &areas,
                &PrismaticRecoveryConfig::default(),
            )?;
            for (physical, upper) in [
                (PhysicalEndpoint::Lower, false),
                (PhysicalEndpoint::Upper, true),
            ] {
                hydraulic(error::checked::<R, _>(
                    client,
                    "conditional original-coordinate endpoints",
                    |a| {
                        let bands = geometry.gather(a, &input.owners)?;
                        let supported =
                            support::query(a, &bands, &input)?.prove_support(a, &bands)?;
                        let endpoint = supported.endpoint(a, &bands, physical)?;
                        let boundaries = table.gather(a, &input.owners)?;
                        let segments = ConditionalBandSegments::prepare(
                            a,
                            &bands,
                            &source,
                            &endpoint,
                            &boundaries,
                        )?;
                        let b = section.maximum_band_count();
                        assert_eq!(segments.rows.band_count, 1);
                        assert_eq!(segments.rows.valid.shape(), &[b, 1]);
                        let lanes = Tensor::from_slice(
                            &(0..b).map(|i| i as i64).collect::<Vec<_>>(),
                            &[b],
                            client.device(),
                        )?;
                        let parameter =
                            a.point(&Tensor::ones(&[b], DType::F64, client.device())?)?;
                        let panel = segments.panel(a, &lanes, &parameter)?;
                        let active = panel.active.try_to_vec::<f64>()?;
                        for (field, value) in [
                            panel.coordinates.area_above_start_m2(),
                            panel.coordinates.area_below_end_m2(),
                            panel.coordinates.depth_above_start_m(),
                            panel.coordinates.depth_below_end_m(),
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            let lower = value.lower().try_to_vec::<f64>()?;
                            let higher = value.upper().try_to_vec::<f64>()?;
                            for band in case.bands {
                                if active[band.index] == 0.0 {
                                    continue;
                                }
                                let extrema = band.local[field];
                                let high_endpoint = upper ^ (field == 1 || field == 3);
                                let expected = if high_endpoint {
                                    extrema.maximum
                                } else {
                                    extrema.minimum
                                }
                                .map(f64::from_bits);
                                let actual = [lower[band.index], higher[band.index]];
                                assert!(
                                    actual[0] <= expected[0] && expected[1] <= actual[1],
                                    "{}, endpoint={upper}, band={}, field={field}: {actual:?} excludes {expected:?}",
                                    case.id,
                                    band.index
                                );
                                let error = (actual[0] - expected[1])
                                    .abs()
                                    .max((actual[1] - expected[0]).abs());
                                assert!(
                                    error
                                        <= f64::from_bits(
                                            extrema.allowances[usize::from(high_endpoint)]
                                        ),
                                    "{}, band={}, field={field}, error={error}",
                                    case.id,
                                    band.index
                                );
                            }
                        }
                        Ok(())
                    },
                ))?;
            }
        }
        Ok(())
    }

    fn controls<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let foreign = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let foreign_geometry = HydraulicGeometry::from_batch(&foreign);
        let owners = Tensor::<R>::from_slice(&[1_i64, 1, 2], &[3], client.device())?;
        let areas = support::expansion::<R>(
            client,
            &[
                [4.0_f64.to_bits(), 0, 0],
                [6.0_f64.to_bits(), 0, 0],
                [4.0_f64.to_bits(), 0, 0],
            ],
            false,
        )?;
        let source = section.recover_area_points_tensor(
            &owners,
            &areas,
            &PrismaticRecoveryConfig::default(),
        )?;
        hydraulic(error::checked::<R, _>(
            client,
            "conditional signed gaps and owned selection",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let foreign_boundaries = foreign_geometry.boundary_areas(a)?.gather(a, &owners)?;
                let scale =
                    a.expansion_point(&Tensor::from_slice(&[2.0; 3], &[3], client.device())?)?;
                let parameter = a.point(&Tensor::from_slice(
                    &[1.0, -1.0, 1.0],
                    &[3],
                    client.device(),
                )?)?;
                let query = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let endpoint = query.endpoint(a, &bands, PhysicalEndpoint::Lower)?;
                assert!(
                    ConditionalBandSegments::prepare(
                        a,
                        &bands,
                        &source,
                        &endpoint,
                        &foreign_boundaries
                    )
                    .is_err()
                );
                let segments =
                    ConditionalBandSegments::prepare(a, &bands, &source, &endpoint, &boundaries)?;
                assert_eq!(
                    segments.owners().try_to_vec::<i64>()?,
                    [1, 1, 1, 1, 1, 1, 2, 2, 2]
                );
                assert_eq!(
                    segments.original_rows().try_to_vec::<i64>()?,
                    [0, 0, 0, 1, 1, 1, 2, 2, 2]
                );
                assert_eq!(
                    segments.band_indices().try_to_vec::<i64>()?,
                    [0, 1, 2, 0, 1, 2, 0, 1, 2]
                );
                assert_eq!(
                    segments.active_mask().try_to_vec::<f64>()?,
                    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                );
                assert_eq!(
                    segments.zero_gap_mask().try_to_vec::<f64>()?,
                    [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
                );
                assert_eq!(
                    segments.retained_gap().high().try_to_vec::<f64>()?,
                    [2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 2.0, 0.0]
                );
                assert!(
                    segments
                        .retained_gap()
                        .abs_error()
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|&r| r == 0.0)
                );
                assert_eq!(segments.physical_gap().lower().shape(), &[9]);

                let selected_rows = Tensor::from_slice(&[3_i64, 0, 7, 0], &[4], client.device())?;
                let selected = segments.select(a, &selected_rows)?;
                assert_eq!(selected.original_rows().try_to_vec::<i64>()?, [1, 0, 2, 0]);
                assert_eq!(selected.rows.start_area.lower().shape(), &[4, 1]);
                let lanes = Tensor::from_slice(&[0_i64, 1, 3], &[3], client.device())?;
                let half = a.point(&Tensor::from_slice(&[0.5; 3], &[3], client.device())?)?;
                let panel = selected.panel(a, &lanes, &half)?;
                assert_eq!(panel.area.lower().try_to_vec::<f64>()?, [5.0; 3]);
                assert_eq!(panel.area.upper().try_to_vec::<f64>()?, [5.0; 3]);
                assert_eq!(panel.top_width.lower().try_to_vec::<f64>()?, [2.0; 3]);
                assert_eq!(panel.hydraulic_depth.lower().try_to_vec::<f64>()?, [2.5; 3]);
                assert_eq!(panel.depth.lower().try_to_vec::<f64>()?, [2.5; 3]);
                assert_eq!(panel.integral.lower().try_to_vec::<f64>()?, [6.25; 3]);
                assert_eq!(panel.gap.lower().try_to_vec::<f64>()?, [-2.0, 2.0, 2.0]);
                for value in [
                    segments.retained_gap().high(),
                    segments.physical_gap().lower(),
                    segments.start.above_start.lower(),
                    bands.start_area.lower(),
                    boundaries.areas().high(),
                ] {
                    client.copy_into(
                        value,
                        &Tensor::zeros(value.shape(), DType::F64, client.device())?,
                    )?;
                }
                let fresh = selected.panel(a, &lanes, &half)?;
                assert_eq!(fresh.area.lower().try_to_vec::<f64>()?, [5.0; 3]);
                assert_eq!(fresh.hydraulic_depth.lower().try_to_vec::<f64>()?, [2.5; 3]);
                let empty_rows = Tensor::from_slice::<i64>(&[], &[0], client.device())?;
                let empty_s = a.point(&Tensor::zeros(&[0], DType::F64, client.device())?)?;
                assert_eq!(
                    selected
                        .panel(a, &empty_rows, &empty_s)?
                        .area
                        .lower()
                        .shape(),
                    &[0]
                );
                Ok(())
            },
        ))?;
        dry_and_uncertain_boundaries::<R>(client)?;
        tiny_gap_tests::run::<R>(client)
    }

    fn dry_and_uncertain_boundaries<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let owners = Tensor::<R>::from_slice(&[1_i64, 2], &[2], client.device())?;
        let areas = support::expansion::<R>(
            client,
            &[[2.0_f64.to_bits(), 0, 0], [4.0_f64.to_bits(), 0, 0]],
            false,
        )?;
        let source = section.recover_area_points_tensor(
            &owners,
            &areas,
            &PrismaticRecoveryConfig::default(),
        )?;
        hydraulic(error::checked::<R, _>(
            client,
            "conditional dry bottom and uncertain boundaries",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                client.copy_into(
                    boundaries.areas().abs_error(),
                    &Tensor::full_scalar(
                        boundaries.areas().abs_error().shape(),
                        DType::F64,
                        f64::EPSILON,
                        client.device(),
                    )?,
                )?;
                let scale =
                    a.expansion_point(&Tensor::from_slice(&[2.0; 2], &[2], client.device())?)?;
                let parameter =
                    a.point(&Tensor::from_slice(&[-1.0, 1.0], &[2], client.device())?)?;
                let endpoint = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let segments =
                    ConditionalBandSegments::prepare(a, &bands, &source, &endpoint, &boundaries)?;
                let zero = segments.zero_gap_mask().try_to_vec::<f64>()?;
                assert_eq!((zero[3], zero[5]), (1.0, 1.0));
                assert_eq!(
                    segments.retained_gap().abs_error().try_to_vec::<f64>()?[3],
                    0.0
                );
                let first = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let at_dry = a.point(&Tensor::ones(&[1], DType::F64, client.device())?)?;
                assert!(
                    matches!(segments.panel(a, &first, &at_dry), Err(HydraulicError::Rejected { reason: PrismaticRiemannRejection::QuadratureUnresolved, ref context, .. }) if context.contains("bottom regularization"))
                );
                let middle = a.point(&Tensor::from_slice(&[0.5], &[1], client.device())?)?;
                let wet = segments.panel(a, &first, &middle)?;
                assert!(
                    wet.area.lower().item::<f64>()? <= 1.0
                        && wet.area.upper().item::<f64>()? >= 1.0
                );
                Ok(())
            },
        ))
    }

    #[test]
    fn cpu_conditional_band_panels_preserve_local_geometry_and_owned_lanes() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        frozen_endpoints::<CpuRuntime>(&client)?;
        controls::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_conditional_band_panels_preserve_local_geometry_and_owned_lanes() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        frozen_endpoints::<CudaRuntime>(&client)?;
        controls::<CudaRuntime>(&client)
    }
}
