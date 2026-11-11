use super::super::CrossSectionError;
use super::{
    band_geometry::{BandRows, empty_fields, geometry},
    batch::PrismaticSectionBatch,
    query::PrismaticSectionQuery,
    query_input,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
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
    /// Enclose geometry at original represented depths for repeated or unordered native section owners.
    ///
    /// Owners use I64 and depths use F64 metres. Both rank-one inputs receive independent native snapshots.
    /// Zero depths are dry. Ambiguous cap membership returns a numerical error.
    pub fn evaluate_depths_tensor(
        &self,
        owners: &Tensor<R>,
        depths_m: &Tensor<R>,
    ) -> Result<PrismaticSectionQuery<R>, CrossSectionError> {
        let count = query_input::metadata(&self.context, owners, depths_m)?;
        let context = &self.context;
        query_input::checked(context, "prismatic depth query", |a| {
            let inputs = query_input::capture(context, owners, depths_m, "depth_m")?;
            let fields = if count == 0 {
                empty_fields(a, &inputs.values)?
            } else {
                let c = a.client();
                let rows = BandRows::gather(a, context, &inputs.owners)?;
                query_input::reject_cap(
                    context,
                    &inputs,
                    &c.mul(&inputs.wet, &c.gt(&inputs.values, rows.cap_depth.upper())?)?,
                    &rows.cap_depth,
                    "depth_m",
                    false,
                )?;
                let overlap = c.mul(
                    &inputs.wet,
                    &c.mul(
                        &c.gt(&inputs.values, rows.cap_depth.lower())?,
                        &c.le(&inputs.values, rows.cap_depth.upper())?,
                    )?,
                )?;
                query_input::reject_cap(
                    context,
                    &inputs,
                    &overlap,
                    &rows.cap_depth,
                    "depth_m",
                    true,
                )?;
                let depth = a.point(&inputs.values)?;
                geometry(a, context, &rows, &inputs.owners, depth, &inputs.wet, None)?
            };
            Ok(PrismaticSectionQuery {
                context: Arc::clone(context),
                owners: inputs.owners,
                requested_depths: inputs.values,
                dry: inputs.dry,
                fields,
            })
        })
    }
}
