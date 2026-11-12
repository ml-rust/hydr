use super::super::{batch::PrismaticSectionBatch, config::PrismaticRecoveryConfig, topology};
use super::{
    compilation::{self, ContactTable},
    contact,
    query::{PrismaticPerimeterQuery, PrismaticPerimeterRecovery},
};
use crate::channel::section::{CrossSectionDefinition, CrossSectionError};
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

/// Certified surveyed geometry with separately compiled bank contact lengths.
#[derive(Clone, Debug)]
pub struct PrismaticPerimeterBatch<R: Runtime<DType = DType>> {
    pub(super) geometry: PrismaticSectionBatch<R>,
    pub(super) contact: Arc<ContactTable<R>>,
}

impl<R: Runtime<DType = DType>> PrismaticPerimeterBatch<R> {
    /// Original immutable definitions in section-index order.
    pub fn definitions(&self) -> &[CrossSectionDefinition] {
        self.geometry.definitions()
    }

    /// Band columns allocated for each section and native query.
    pub fn maximum_band_count(&self) -> usize {
        self.geometry.maximum_band_count()
    }
}

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
    /// Compile certified wetted contact from the retained original surveyed coordinates.
    ///
    /// Unsupported square, sum, square-root, or quotient ranges return typed errors.
    pub fn certify_perimeter(&self) -> Result<PrismaticPerimeterBatch<R>, CrossSectionError> {
        let topology = topology::classify(self.definitions())?;
        let contact = compilation::compile(&self.context, &topology)?;
        Ok(PrismaticPerimeterBatch {
            geometry: PrismaticSectionBatch {
                context: Arc::clone(&self.context),
            },
            contact: Arc::new(contact),
        })
    }
}

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
    /// Enclose geometry, wetted perimeter, and radius for original native depths in metres.
    ///
    /// Inputs have matching rank-one I64 owner and F64 depth shapes. Results own their native storage.
    pub fn evaluate_depths_tensor(
        &self,
        owners: &Tensor<R>,
        depths_m: &Tensor<R>,
    ) -> Result<PrismaticPerimeterQuery<R>, CrossSectionError> {
        let geometry = self.geometry.evaluate_depths_tensor(owners, depths_m)?;
        let fields = contact::evaluate(
            &geometry.context,
            &self.contact,
            geometry.owners(),
            geometry.depth_m(),
            geometry.dry_mask(),
            geometry.area_m2(),
        )?;
        Ok(PrismaticPerimeterQuery { geometry, fields })
    }

    /// Recover geometry while retaining the original conserved area as the radius numerator.
    ///
    /// Inputs have matching rank-one I64 owner and F64 area shapes. Areas use square metres.
    pub fn recover_depths_tensor(
        &self,
        owners: &Tensor<R>,
        areas_m2: &Tensor<R>,
        config: &PrismaticRecoveryConfig,
    ) -> Result<PrismaticPerimeterRecovery<R>, CrossSectionError> {
        let geometry = self
            .geometry
            .recover_depths_tensor(owners, areas_m2, config)?;
        let context = &geometry.context;
        let fields = super::super::query_input::checked(
            context,
            "prismatic conserved-area perimeter",
            |a| {
                let numerator = a.point(geometry.requested_areas_m2())?;
                contact::fields(
                    a,
                    context,
                    &self.contact,
                    geometry.owners(),
                    geometry.depth_m(),
                    geometry.dry_mask(),
                    &numerator,
                )
            },
        )?;
        Ok(PrismaticPerimeterRecovery { geometry, fields })
    }
}
