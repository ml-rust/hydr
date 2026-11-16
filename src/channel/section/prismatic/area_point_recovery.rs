use super::super::CrossSectionError;
use super::{
    area_point_fields, area_point_hydrostatic, area_point_input, area_point_inverse,
    area_point_query::{PrismaticAreaPointBandCoordinates, PrismaticAreaPointRecovery},
    band_geometry::{BandRows, empty_fields},
    batch::PrismaticSectionBatch,
    config::PrismaticRecoveryConfig,
    query_input,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

impl<R> PrismaticSectionBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Recover geometry at exact high-plus-low area points with numerically zero radii.
    ///
    /// Owners use rank-one I64 tensors. Area components use matching F64 tensors on the configured device.
    /// Positive points require certified represented positive depths. Unresolved support returns a numerical error.
    pub fn recover_area_points_tensor(
        &self,
        owners: &Tensor<R>,
        areas_m2: &CertifiedExpansion<R>,
        config: &PrismaticRecoveryConfig,
    ) -> Result<PrismaticAreaPointRecovery<R>, CrossSectionError> {
        config.check()?;
        let count = area_point_input::metadata(&self.context, owners, areas_m2)?;
        let context = &self.context;
        query_input::checked(context, "prismatic area-point inverse", |a| {
            let qualification = Tensor::zeros(&[0], DType::F64, a.client().device())?;
            let qualification = a.point(&qualification)?;
            a.sqrt_nonnegative(&qualification)?;
            let inputs = area_point_input::capture(a, context, owners, areas_m2)?;
            let c = a.client();
            let zero = &context.source.zero;
            let (fields, coordinates, hydrostatic_point) = if count == 0 {
                let value = Tensor::zeros(&[0], DType::F64, c.device())?;
                let bands = Tensor::zeros(&[0, context.maximum_bands], DType::F64, c.device())?;
                (
                    empty_fields(a, &value)?,
                    PrismaticAreaPointBandCoordinates {
                        candidate: bands.clone(),
                        area_above_start: a.point(&bands)?,
                        area_below_end: a.point(&bands)?,
                        depth_above_start: a.point(&bands)?,
                        depth_below_end: a.point(&bands)?,
                    },
                    a.expansion_point(&value)?,
                )
            } else {
                let rows = BandRows::gather(a, context, &inputs.owners)?;
                let cap_difference =
                    area_point_input::difference(a, &inputs.areas, &rows.cap_area, false)?;
                query_input::reject(
                    context,
                    &inputs.owners,
                    &c.mul(&inputs.wet, &c.gt(cap_difference.lower(), zero)?)?,
                    false,
                    "requested area point exceeds surveyed support. Check units or extend the profile",
                )?;
                let overlap = c.mul(&inputs.wet, &c.gt(cap_difference.upper(), zero)?)?;
                query_input::reject(
                    context,
                    &inputs.owners,
                    &overlap,
                    true,
                    "cap membership is unresolved for the requested area point. Review profile conditioning",
                )?;
                let coordinates = area_point_inverse::recover(a, context, &rows, &inputs)?;
                let fields = area_point_fields::geometry(a, context, &rows, &inputs, &coordinates)?;
                query_input::reject(
                    context,
                    &inputs.owners,
                    &c.mul(&inputs.wet, &c.le(fields.depth.lower(), zero)?)?,
                    true,
                    "positive area point has no certified represented positive depth. Keep the area and review profile conditioning",
                )?;
                let low = a.point(fields.depth.lower())?;
                let high = a.point(fields.depth.upper())?;
                let width = a.sub(&high, &low)?;
                let width_upper = c.where_cond(
                    &c.eq(fields.depth.lower(), fields.depth.upper())?,
                    zero,
                    width.upper(),
                )?;
                let scale = a.point(
                    &c.maximum(&c.abs(fields.depth.lower())?, &c.abs(fields.depth.upper())?)?,
                )?;
                let absolute = a.point(&Tensor::from_slice(
                    &[config.depth_atol_m],
                    &[],
                    c.device(),
                )?)?;
                let relative =
                    a.point(&Tensor::from_slice(&[config.depth_rtol], &[], c.device())?)?;
                let relative_width = a.mul(&relative, &scale)?;
                let allowance = a.add(&absolute, &relative_width)?;
                query_input::reject(
                    context,
                    &inputs.owners,
                    &c.gt(&width_upper, &c.maximum(allowance.lower(), zero)?)?,
                    true,
                    "depth enclosure exceeds its width allowance. Review profile conditioning or widen the allowance",
                )?;
                let residual = area_point_input::difference(a, &inputs.areas, &fields.area, false)?;
                let outside = c.maximum(
                    &c.gt(residual.lower(), zero)?,
                    &c.lt(residual.upper(), zero)?,
                )?;
                query_input::reject(
                    context,
                    &inputs.owners,
                    &outside,
                    true,
                    "recovered forward area excludes the requested area point. Keep the original point and inspect geometry",
                )?;
                let hydrostatic_point = area_point_hydrostatic::point(
                    a,
                    context,
                    &inputs,
                    &coordinates,
                    &fields.integral,
                )?;
                (fields, coordinates, hydrostatic_point)
            };
            Ok(PrismaticAreaPointRecovery {
                context: Arc::clone(context),
                owners: inputs.owners,
                requested_areas: inputs.areas,
                dry: inputs.dry,
                fields,
                hydrostatic_point,
                coordinates,
            })
        })
    }
}
