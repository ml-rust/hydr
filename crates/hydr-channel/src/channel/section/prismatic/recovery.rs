use super::super::CrossSectionError;
use super::{
    band_geometry::{BandRows, empty_fields, geometry},
    batch::PrismaticSectionBatch,
    config::PrismaticRecoveryConfig,
    inverse_bands,
    query::PrismaticSectionRecovery,
    query_input,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
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
    /// Enclose inverse depths while preserving every original represented area operand.
    ///
    /// Owners use I64 and areas use F64 square metres. Both inputs have matching rank-one shape.
    /// Positive areas require a positive certified depth. Unresolved support or width returns a numerical error.
    pub fn recover_depths_tensor(
        &self,
        owners: &Tensor<R>,
        areas_m2: &Tensor<R>,
        config: &PrismaticRecoveryConfig,
    ) -> Result<PrismaticSectionRecovery<R>, CrossSectionError> {
        config.check()?;
        let count = query_input::metadata(&self.context, owners, areas_m2)?;
        let context = &self.context;
        query_input::checked(context, "prismatic conserved-area inverse", |a| {
            let inputs = query_input::capture(context, owners, areas_m2, "area_m2")?;
            let fields = if count == 0 {
                empty_fields(a, &inputs.values)?
            } else {
                let c = a.client();
                let zero = &context.source.zero;
                let rows = BandRows::gather(a, context, &inputs.owners)?;
                query_input::reject_cap(
                    context,
                    &inputs,
                    &c.mul(&inputs.wet, &c.gt(&inputs.values, rows.cap_area.upper())?)?,
                    &rows.cap_area,
                    "area_m2",
                    false,
                )?;
                let overlap = c.mul(
                    &inputs.wet,
                    &c.mul(
                        &c.gt(&inputs.values, rows.cap_area.lower())?,
                        &c.le(&inputs.values, rows.cap_area.upper())?,
                    )?,
                )?;
                query_input::reject_cap(
                    context,
                    &inputs,
                    &overlap,
                    &rows.cap_area,
                    "area_m2",
                    true,
                )?;
                let depth = inverse_bands::recover(a, context, &rows, &inputs)?;
                query_input::reject(
                    context,
                    &inputs.owners,
                    &c.mul(&inputs.wet, &c.le(depth.lower(), zero)?)?,
                    true,
                    "positive conserved area has no certified represented positive depth. Keep the area and reconsider profile conditioning",
                )?;
                let lower_depth = a.point(depth.lower())?;
                let upper_depth = a.point(depth.upper())?;
                let width = a.sub(&upper_depth, &lower_depth)?;
                let width_upper =
                    c.where_cond(&c.eq(depth.lower(), depth.upper())?, zero, width.upper())?;
                let scale = a.point(&c.maximum(&c.abs(depth.lower())?, &c.abs(depth.upper())?)?)?;
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
                    "depth enclosure exceeds its width allowance. Check profile conditioning or choose a wider allowance",
                )?;
                let fields = geometry(
                    a,
                    context,
                    &rows,
                    &inputs.owners,
                    depth,
                    &inputs.wet,
                    Some(&inputs.values),
                )?;
                let outside = c.maximum(
                    &c.gt(fields.area.lower(), &inputs.values)?,
                    &c.lt(fields.area.upper(), &inputs.values)?,
                )?;
                query_input::reject(
                    context,
                    &inputs.owners,
                    &outside,
                    true,
                    "recovered forward area excludes the conserved input. Keep the original area and inspect the geometry",
                )?;
                fields
            };
            Ok(PrismaticSectionRecovery {
                context: Arc::clone(context),
                owners: inputs.owners,
                requested_areas: inputs.values,
                dry: inputs.dry,
                fields,
            })
        })
    }
}
