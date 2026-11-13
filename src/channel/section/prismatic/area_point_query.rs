use super::super::CrossSectionError;
use super::{area_point_projection, context::Context, query::Fields};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};
use std::sync::Arc;

/// Retained local coordinates with shape `[query, band]` and zero inactive placeholders.
///
/// Keep tensors, endpoints, and storage aliases immutable. Every active column encloses a compatible physical band.
/// Dry rows retain geometric endpoints at depth zero, with exact zero distances above the band start.
#[derive(Debug)]
pub struct PrismaticAreaPointBandCoordinates<R: Runtime<DType = DType>> {
    pub(super) candidate: Tensor<R>,
    pub(super) area_above_start: Interval<R>,
    pub(super) area_below_end: Interval<R>,
    pub(super) depth_above_start: Interval<R>,
    pub(super) depth_below_end: Interval<R>,
}

impl<R: Runtime<DType = DType>> PrismaticAreaPointBandCoordinates<R> {
    /// F64 mask containing one for compatible bands and zero for inactive columns.
    pub fn candidate_mask(&self) -> &Tensor<R> {
        &self.candidate
    }

    /// Requested area minus band-start area, in square metres.
    pub fn area_above_start_m2(&self) -> &Interval<R> {
        &self.area_above_start
    }

    /// Band-end area minus requested area, in square metres.
    pub fn area_below_end_m2(&self) -> &Interval<R> {
        &self.area_below_end
    }

    /// Physical depth above the band start, in metres.
    pub fn depth_above_start_m(&self) -> &Interval<R> {
        &self.depth_above_start
    }

    /// Physical depth below the band end, in metres.
    pub fn depth_below_end_m(&self) -> &Interval<R> {
        &self.depth_below_end
    }
}

/// Host bounds for one compatible physical band.
#[derive(Clone, Debug, PartialEq)]
pub struct PrismaticAreaPointBandProperties {
    /// Zero-based band column in the compiled section.
    pub band_index: usize,
    /// Requested area above the band start, in square metres.
    pub area_above_start_m2: [f64; 2],
    /// Remaining area below the band end, in square metres.
    pub area_below_end_m2: [f64; 2],
    /// Physical depth above the band start, in metres.
    pub depth_above_start_m: [f64; 2],
    /// Physical depth below the band end, in metres.
    pub depth_below_end_m: [f64; 2],
}

/// Host geometry conditioned on an exact two-component area point.
#[derive(Clone, Debug, PartialEq)]
pub struct PrismaticAreaPointRecoveredProperties {
    /// Stable identity of the queried section.
    pub section_id: String,
    /// Original requested high component, in square metres.
    pub requested_area_high_m2: f64,
    /// Original requested low component, in square metres.
    pub requested_area_low_m2: f64,
    /// True when the exact requested sum is zero.
    pub dry: bool,
    /// Recovered depth bounds `[lower, upper]`, in metres.
    pub depth_m: [f64; 2],
    /// Forward area bounds `[lower, upper]`, in square metres.
    pub forward_area_m2: [f64; 2],
    /// Wet width bounds, in metres. Absent when dry.
    pub top_width_m: Option<[f64; 2]>,
    /// Hydrostatic integral bounds, in cubic metres.
    pub hydrostatic_integral_m3: [f64; 2],
    /// Original requested area divided by width, in metres. Absent when dry.
    pub hydraulic_depth_m: Option<[f64; 2]>,
    /// Compatible physical bands in compiled column order.
    pub bands: Vec<PrismaticAreaPointBandProperties>,
}

/// Native geometry conditioned on exact high-plus-low area points.
///
/// Geometry tensors have shape `[N]`. Band tensors have shape `[N, maximum_band_count]`.
/// Keep native storage and endpoint aliases immutable. Dry width and hydraulic depth contain zero placeholders.
/// Dry band coordinates describe geometric endpoints at depth zero and retain certified distances below band ends.
/// Results own their storage and preserve original requested component bits.
#[derive(Debug)]
pub struct PrismaticAreaPointRecovery<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) owners: Tensor<R>,
    pub(super) requested_areas: CertifiedExpansion<R>,
    pub(super) dry: Tensor<R>,
    pub(super) fields: Fields<R>,
    pub(super) coordinates: PrismaticAreaPointBandCoordinates<R>,
}

impl<R: Runtime<DType = DType>> PrismaticAreaPointRecovery<R> {
    #[cfg(test)]
    pub(in crate::channel::section::prismatic) fn same_context(
        &self,
        context: &Arc<Context<R>>,
    ) -> bool {
        Arc::ptr_eq(&self.context, context)
    }

    /// Original captured I64 section owners. Keep storage immutable.
    pub fn owners(&self) -> &Tensor<R> {
        &self.owners
    }

    /// Original captured area components, in square metres. Keep storage immutable.
    pub fn requested_areas_m2(&self) -> &CertifiedExpansion<R> {
        &self.requested_areas
    }

    /// F64 mask containing one for exact dry points and zero for wet points.
    pub fn dry_mask(&self) -> &Tensor<R> {
        &self.dry
    }

    /// Recovered geometric depth bounds, in metres. Keep endpoints immutable.
    pub fn depth_m(&self) -> &Interval<R> {
        &self.fields.depth
    }

    /// Forward geometry area bounds, in square metres. Keep endpoints immutable.
    pub fn forward_area_m2(&self) -> &Interval<R> {
        &self.fields.area
    }

    /// Wet width bounds, in metres. Dry rows contain zero placeholders.
    pub fn top_width_m(&self) -> &Interval<R> {
        &self.fields.top_width
    }

    /// Hydrostatic integral bounds, in cubic metres. Keep endpoints immutable.
    pub fn hydrostatic_integral_m3(&self) -> &Interval<R> {
        &self.fields.integral
    }

    /// Original area point divided by wet width, in metres. Dry rows contain zero placeholders.
    pub fn hydraulic_depth_m(&self) -> &Interval<R> {
        &self.fields.hydraulic_depth
    }

    /// Retained coordinates before projection onto absolute depth.
    pub fn band_coordinates(&self) -> &PrismaticAreaPointBandCoordinates<R> {
        &self.coordinates
    }

    /// Copy original components, geometry bounds, and compatible bands to host memory.
    pub fn properties(
        &self,
    ) -> Result<Vec<PrismaticAreaPointRecoveredProperties>, CrossSectionError> {
        area_point_projection::properties(self, self.context.source.client.device())
    }
}
