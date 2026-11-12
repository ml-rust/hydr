use super::super::{config::PrismaticRecoveryConfig, query_input};
use super::{
    area_point_contact, area_point_query::PrismaticPerimeterAreaPointRecovery,
    batch::PrismaticPerimeterBatch,
};
use crate::channel::section::CrossSectionError;
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::CertifiedExpansion;

impl<R> PrismaticPerimeterBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Recover geometry, wetted perimeter, and radius at exact high-plus-low area points.
    ///
    /// Owners use rank-one I64 tensors. Matching F64 area components require numerically zero radii.
    /// Contact uses retained band coordinates. Radius uses the original requested area point.
    pub fn recover_area_points_tensor(
        &self,
        owners: &Tensor<R>,
        areas_m2: &CertifiedExpansion<R>,
        config: &PrismaticRecoveryConfig,
    ) -> Result<PrismaticPerimeterAreaPointRecovery<R>, CrossSectionError> {
        let geometry = self
            .geometry
            .recover_area_points_tensor(owners, areas_m2, config)?;
        let fields =
            query_input::checked(&geometry.context, "prismatic area-point perimeter", |a| {
                area_point_contact::fields(a, &self.contact, &geometry)
            })?;
        Ok(PrismaticPerimeterAreaPointRecovery { geometry, fields })
    }
}
