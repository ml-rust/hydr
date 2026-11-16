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
