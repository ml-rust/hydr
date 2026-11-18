use super::super::{
    area_point_query::{PrismaticAreaPointBandCoordinates, PrismaticAreaPointRecovery},
    context::Context,
    hydraulic_geometry::GatheredBands,
};
use super::{admission, range};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};
use std::sync::Arc;

pub(crate) enum AreaAnchor<R: Runtime<DType = DType>> {
    Source {
        area: Box<CertifiedExpansion<R>>,
        coordinates: Box<PrismaticAreaPointBandCoordinates<R>>,
    },
    Zero,
    #[cfg(test)]
    BandStart(Tensor<R>),
    #[cfg(test)]
    BandEnd(Tensor<R>),
    Cap,
}

#[derive(Clone, Copy)]
pub(crate) enum AreaDirection {
    Add,
    Subtract,
}

pub(crate) struct ExpandedBounds<R: Runtime<DType = DType>> {
    pub(crate) lower: CertifiedExpansion<R>,
    pub(crate) upper: CertifiedExpansion<R>,
}

pub(super) struct EndpointDifferences<R: Runtime<DType = DType>> {
    pub above_start: Interval<R>,
    pub below_end: Interval<R>,
}

pub(crate) struct AreaQuery<R: Runtime<DType = DType>> {
    pub(super) model: Arc<Context<R>>,
    pub(super) anchor: AreaAnchor<R>,
    pub(super) offsets: ExpandedBounds<R>,
    pub(crate) direction: AreaDirection,
    pub(crate) scale: CertifiedExpansion<R>,
    pub(crate) parameter: Interval<R>,
    pub(crate) owners: Tensor<R>,
    pub(crate) original_rows: Tensor<R>,
    pub(super) band_count: usize,
}

/// Construction proves whole-image support before geometry consumes these certificates.
pub(crate) struct SupportedAreaQuery<R: Runtime<DType = DType>> {
    pub(crate) query: AreaQuery<R>,
    pub(crate) area: Interval<R>,
    pub(crate) above_start: Interval<R>,
    pub(crate) below_end: Interval<R>,
    pub(crate) candidate: Tensor<R>,
    pub(super) lower_endpoint: EndpointDifferences<R>,
    pub(super) upper_endpoint: EndpointDifferences<R>,
}

impl<R> AreaQuery<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn source_affine(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        source: &PrismaticAreaPointRecovery<R>,
        direction: AreaDirection,
        scale: &CertifiedExpansion<R>,
        parameter: &Interval<R>,
    ) -> HydraulicResult<Self> {
        admission::matching_model(source.same_context(&bands.model))?;
        admission::matching_owners(a, bands, source.owners())?;
        let areas = source.requested_areas_m2();
        let area = a.expansion_bounds(areas.high(), areas.low(), areas.abs_error())?;
        let coordinates = super::selection::coordinates(a, source.band_coordinates(), None)?;
        let anchor = AreaAnchor::Source {
            area: Box::new(area),
            coordinates: Box::new(coordinates),
        };
        admission::chart(
            a,
            bands,
            #[cfg(test)]
            &anchor,
            scale,
            parameter,
        )?;
        Self::finish(a, bands, anchor, direction, scale, parameter)
    }

    pub(crate) fn band_affine(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        anchor: AreaAnchor<R>,
        direction: AreaDirection,
        scale: &CertifiedExpansion<R>,
        parameter: &Interval<R>,
    ) -> HydraulicResult<Self> {
        if matches!(&anchor, AreaAnchor::Source { .. }) {
            return Err(crate::channel::section::CrossSectionError::input(
                "band_affine received a Source anchor. Use source_affine with a recovery from the same compiled model",
            ).into());
        }
        admission::chart(
            a,
            bands,
            #[cfg(test)]
            &anchor,
            scale,
            parameter,
        )?;
        let anchor = super::selection::capture_anchor(a, &anchor)?;
        Self::finish(a, bands, anchor, direction, scale, parameter)
    }

    fn finish(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        anchor: AreaAnchor<R>,
        direction: AreaDirection,
        scale: &CertifiedExpansion<R>,
        parameter: &Interval<R>,
    ) -> HydraulicResult<Self> {
        let offsets = range::offsets(a, scale, parameter, direction)?;
        let c = a.client();
        let owners = Tensor::empty(bands.section_owner.shape(), DType::I64, c.device())?;
        c.copy_into(&owners, &bands.section_owner)?;
        let original_rows = c.arange(0.0, owners.numel() as f64, 1.0, DType::I64)?;
        let scale = a.expansion_bounds(scale.high(), scale.low(), scale.abs_error())?;
        let parameter = a.bounds(parameter.lower(), parameter.upper())?;
        Ok(Self {
            model: Arc::clone(&bands.model),
            anchor,
            offsets,
            direction,
            scale,
            parameter,
            owners,
            original_rows,
            band_count: bands.band_count,
        })
    }

    pub(crate) fn cap_deficit(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        scale: &CertifiedExpansion<R>,
        parameter: &Interval<R>,
    ) -> HydraulicResult<Self> {
        let query = Self::band_affine(
            a,
            bands,
            AreaAnchor::Cap,
            AreaDirection::Subtract,
            scale,
            parameter,
        )?;
        admission::nonnegative_parameter(a, &query.parameter)?;
        Ok(query)
    }
}

#[cfg(test)]
mod tests {
    use super::super::admission_tests;
    use super::super::fixtures::{AnchorKind, CASES, Case};
    use super::super::test_support::{
        Client, Result, batch, check, hydraulic, input, query, snapshot,
    };
    use super::*;
    use crate::channel::flow::surveyed::error;
    use crate::channel::section::HydraulicGeometry;
    use std::collections::{BTreeMap, BTreeSet};

    fn matrix<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        assert_eq!(CASES.len(), 17);
        assert_eq!(
            CASES
                .iter()
                .map(|case| case.group)
                .collect::<BTreeSet<_>>()
                .len(),
            15
        );
        assert!(CASES.iter().any(|left| {
            CASES
                .iter()
                .any(|right| left.group == right.group && left.chart.add != right.chart.add)
        }));
        let section = batch::<R>(&client)?;
        let definitions = section.definitions().to_vec();
        let geometry = HydraulicGeometry::from_batch(&section);
        let mut groups: BTreeMap<(AnchorKind, bool), Vec<&Case>> = BTreeMap::new();
        for case in CASES.iter().rev() {
            groups
                .entry((case.chart.kind, case.chart.add))
                .or_default()
                .push(case);
        }
        let mut exercised = BTreeSet::new();
        for cases in groups.values() {
            let charts: Vec<_> = cases.iter().map(|case| case.chart).collect();
            let count = cases.len();
            let last = i64::try_from(count - 1)?;
            let row_values = [last, 0, last, 0];
            let row_indices = [count - 1, 0, count - 1, 0];
            let selected_cases: Vec<_> = row_indices.iter().map(|&index| cases[index]).collect();
            let expected_owners: Vec<_> = selected_cases
                .iter()
                .map(|case| i64::try_from(case.chart.owner))
                .collect::<std::result::Result<_, _>>()?;
            let rows = Tensor::<R>::from_slice(&row_values, &[row_values.len()], client.device())?;
            let mut contiguous = None;
            for strided in [false, true] {
                let input = input::<R>(&client, &section, &charts, strided, false)?;
                let (
                    full,
                    selected,
                    selected_raw,
                    selected_owners,
                    original_rows,
                    raw_original_rows,
                ) = hydraulic(error::checked::<R, _>(
                    &client,
                    "private area chart matrix",
                    |a| {
                        let bands = geometry.gather(a, &input.owners)?;
                        let query = query(a, &bands, &input)?;
                        let selected_bands = bands.select(a, &rows)?;
                        let selected_raw =
                            query.select(a, &rows)?.prove_support(a, &selected_bands)?;
                        let supported = query.prove_support(a, &bands)?;
                        let selected = supported.select(a, &rows)?;
                        let selected_owners = selected.query.owners.try_to_vec::<i64>()?;
                        let original_rows = selected.query.original_rows.try_to_vec::<i64>()?;
                        let raw_original_rows =
                            selected_raw.query.original_rows.try_to_vec::<i64>()?;
                        Ok((
                            geometry.recover_query(a, &bands, &supported)?,
                            geometry.recover_query(a, &selected_bands, &selected)?,
                            geometry.recover_query(a, &selected_bands, &selected_raw)?,
                            selected_owners,
                            original_rows,
                            raw_original_rows,
                        ))
                    },
                ))?;
                assert_eq!(selected_owners, expected_owners);
                assert_eq!(original_rows, row_values);
                assert_eq!(raw_original_rows, row_values);
                let expected = snapshot(&full)?;
                let retained_selected = snapshot(&selected)?;
                let retained_raw = snapshot(&selected_raw)?;
                check(&expected, cases);
                check(&retained_selected, &selected_cases);
                check(&retained_raw, &selected_cases);
                if let Some(contiguous) = &contiguous {
                    assert_eq!(&expected, contiguous);
                }
                for interval in [
                    &full.state.area,
                    &full.state.forward_area,
                    &full.state.depth,
                    &full.state.top_width,
                    &full.state.integral,
                    &full.state.hydraulic_depth,
                    full.coordinates.area_above_start_m2(),
                    full.coordinates.area_below_end_m2(),
                    full.coordinates.depth_above_start_m(),
                    full.coordinates.depth_below_end_m(),
                ] {
                    for endpoint in [interval.lower(), interval.upper()] {
                        client.copy_into(
                            endpoint,
                            &Tensor::zeros(endpoint.shape(), DType::F64, client.device())?,
                        )?;
                    }
                }
                assert_eq!(snapshot(&selected)?, retained_selected);
                assert_eq!(snapshot(&selected_raw)?, retained_raw);
                let fresh = hydraulic(error::checked::<R, _>(
                    &client,
                    "independent private area chart",
                    |a| {
                        let bands = geometry.gather(a, &input.owners)?;
                        let supported = query(a, &bands, &input)?.prove_support(a, &bands)?;
                        geometry.recover_query(a, &bands, &supported)
                    },
                ))?;
                assert_eq!(snapshot(&fresh)?, expected);
                assert_eq!(section.definitions(), definitions.as_slice());
                contiguous = Some(expected);
            }
            exercised.extend(cases.iter().map(|case| case.id));
        }
        assert_eq!(exercised.len(), 17);
        admission_tests::run::<R>(&client, &section)
    }

    #[test]
    fn configured_cpu_private_area_charts_enclose_independent_extrema() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let previous = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        let retained = client.clone();
        matrix::<CpuRuntime>(client)?;
        assert_eq!(retained.parallelism(), config);
        assert_eq!(original.parallelism(), previous);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_private_area_charts_enclose_independent_extrema() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
