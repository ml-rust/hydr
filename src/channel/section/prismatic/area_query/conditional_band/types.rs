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
