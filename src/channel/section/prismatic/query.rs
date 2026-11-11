use super::super::CrossSectionError;
use super::context::Context;
use super::output::{PrismaticRecoveredProperties, PrismaticSectionProperties};
use super::projection::Projection;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct Fields<R: Runtime<DType = DType>> {
    pub(super) depth: Interval<R>,
    pub(super) area: Interval<R>,
    pub(super) top_width: Interval<R>,
    pub(super) integral: Interval<R>,
    pub(super) hydraulic_depth: Interval<R>,
}

/// Native geometry bounds for original represented depth operands.
///
/// All tensors have shape `[N]`. Treat their storage and endpoint aliases as immutable.
/// Native dry width and hydraulic-depth fields contain zero placeholders. Use the dry mask before reading them.
/// Results own query storage and never expose compiled coefficient storage.
#[derive(Debug)]
pub struct PrismaticSectionQuery<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) owners: Tensor<R>,
    pub(super) requested_depths: Tensor<R>,
    pub(super) dry: Tensor<R>,
    pub(super) fields: Fields<R>,
}

impl<R: Runtime<DType = DType>> PrismaticSectionQuery<R> {
    /// Original I64 section-owner snapshot. Keep its storage immutable.
    pub fn owners(&self) -> &Tensor<R> {
        &self.owners
    }

    /// Original F64 depth operands, in metres, including signed-zero bits. Keep their storage immutable.
    pub fn requested_depths_m(&self) -> &Tensor<R> {
        &self.requested_depths
    }

    /// F64 mask containing one for dry rows and zero for wet rows. Keep its storage immutable.
    pub fn dry_mask(&self) -> &Tensor<R> {
        &self.dry
    }

    /// Geometric depth bounds above the bed, in metres. Keep endpoints immutable.
    pub fn depth_m(&self) -> &Interval<R> {
        &self.fields.depth
    }

    /// Wetted area bounds, in square metres. Keep endpoints immutable.
    pub fn area_m2(&self) -> &Interval<R> {
        &self.fields.area
    }

    /// Wet horizontal width bounds, in metres. Dry rows contain zero placeholders. Keep endpoints immutable.
    pub fn top_width_m(&self) -> &Interval<R> {
        &self.fields.top_width
    }

    /// Hydrostatic integral bounds, in cubic metres. Keep endpoints immutable.
    pub fn hydrostatic_integral_m3(&self) -> &Interval<R> {
        &self.fields.integral
    }

    /// Area divided by wet width, in metres. Dry rows contain zero placeholders. Keep endpoints immutable.
    pub fn hydraulic_depth_m(&self) -> &Interval<R> {
        &self.fields.hydraulic_depth
    }

    /// Copy every field to host memory in original query order.
    pub fn properties(&self) -> Result<Vec<PrismaticSectionProperties>, CrossSectionError> {
        let projection = Projection::read(
            &self.owners,
            &self.requested_depths,
            &self.dry,
            &self.fields,
            self.context.source.client.device(),
        )?;
        (0..projection.len())
            .map(|index| {
                let row = projection.row(index)?;
                let definition = self.context.source.definitions.get(row.owner).ok_or_else(|| {
                    CrossSectionError::numerical(format!(
                        "prismatic output owner={}, query={index}: rebuild the query with admitted section owners",
                        row.owner
                    ))
                })?;
                Ok(PrismaticSectionProperties {
                    section_id: definition.id.clone(),
                    query_depth_m: row.requested,
                    dry: row.dry,
                    depth_m: row.depth,
                    area_m2: row.area,
                    top_width_m: (!row.dry).then_some(row.top_width),
                    hydrostatic_integral_m3: row.integral,
                    hydraulic_depth_m: (!row.dry).then_some(row.hydraulic_depth),
                })
            })
            .collect()
    }
}

/// Native geometry bounds conditioned on original represented conserved areas.
///
/// All tensors have shape `[N]`. Treat their storage and endpoint aliases as immutable.
/// Native dry width and hydraulic-depth fields contain zero placeholders. Use the dry mask before reading them.
/// Forward-area bounds describe recovered geometry. Original requested areas retain their conservation role.
/// Results own query storage and never expose compiled coefficient storage.
#[derive(Debug)]
pub struct PrismaticSectionRecovery<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) owners: Tensor<R>,
    pub(super) requested_areas: Tensor<R>,
    pub(super) dry: Tensor<R>,
    pub(super) fields: Fields<R>,
}

impl<R: Runtime<DType = DType>> PrismaticSectionRecovery<R> {
    /// Original I64 section-owner snapshot. Keep its storage immutable.
    pub fn owners(&self) -> &Tensor<R> {
        &self.owners
    }

    /// Original conserved F64 areas, in square metres, including signed-zero bits. Keep their storage immutable.
    pub fn requested_areas_m2(&self) -> &Tensor<R> {
        &self.requested_areas
    }

    /// F64 mask containing one for dry rows and zero for wet rows. Keep its storage immutable.
    pub fn dry_mask(&self) -> &Tensor<R> {
        &self.dry
    }

    /// Recovered geometric depth bounds above the bed, in metres. Keep endpoints immutable.
    pub fn depth_m(&self) -> &Interval<R> {
        &self.fields.depth
    }

    /// Forward area bounds across recovered depths, in square metres. Keep endpoints immutable.
    pub fn forward_area_m2(&self) -> &Interval<R> {
        &self.fields.area
    }

    /// Wet horizontal width bounds, in metres. Dry rows contain zero placeholders. Keep endpoints immutable.
    pub fn top_width_m(&self) -> &Interval<R> {
        &self.fields.top_width
    }

    /// Hydrostatic integral bounds, in cubic metres. Keep endpoints immutable.
    pub fn hydrostatic_integral_m3(&self) -> &Interval<R> {
        &self.fields.integral
    }

    /// Requested area divided by wet width, in metres. Dry rows contain zero placeholders. Keep endpoints immutable.
    pub fn hydraulic_depth_m(&self) -> &Interval<R> {
        &self.fields.hydraulic_depth
    }

    /// Copy original areas and recovered bounds to host memory in original query order.
    pub fn properties(&self) -> Result<Vec<PrismaticRecoveredProperties>, CrossSectionError> {
        let projection = Projection::read(
            &self.owners,
            &self.requested_areas,
            &self.dry,
            &self.fields,
            self.context.source.client.device(),
        )?;
        (0..projection.len())
            .map(|index| {
                let row = projection.row(index)?;
                let definition = self.context.source.definitions.get(row.owner).ok_or_else(|| {
                    CrossSectionError::numerical(format!(
                        "prismatic output owner={}, query={index}: rebuild the recovery with admitted section owners",
                        row.owner
                    ))
                })?;
                Ok(PrismaticRecoveredProperties {
                    section_id: definition.id.clone(),
                    requested_area_m2: row.requested,
                    dry: row.dry,
                    depth_m: row.depth,
                    forward_area_m2: row.area,
                    top_width_m: (!row.dry).then_some(row.top_width),
                    hydrostatic_integral_m3: row.integral,
                    hydraulic_depth_m: (!row.dry).then_some(row.hydraulic_depth),
                })
            })
            .collect()
    }
}
