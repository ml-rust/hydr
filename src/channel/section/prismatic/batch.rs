use super::super::{CrossSectionBatch, CrossSectionDefinition, CrossSectionError};
use super::{compilation, context::Context, topology};
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Certified connected surveyed profiles queried through their configured native client.
#[derive(Clone, Debug)]
pub struct PrismaticSectionBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
}

impl<R: Runtime<DType = DType>> PrismaticSectionBatch<R> {
    /// Original immutable definitions in section-index order.
    pub fn definitions(&self) -> &[CrossSectionDefinition] {
        &self.context.source.definitions
    }

    /// Band columns allocated for each section and each native query.
    pub fn maximum_band_count(&self) -> usize {
        self.context.maximum_bands
    }
}

impl<R: Runtime<DType = DType>> CrossSectionBatch<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Certify connected monotone banks and convex pressure below each declared cap.
    ///
    /// Cap-level width uses its limit from below. This method adds no flow advancement.
    pub fn certify_prismatic(&self) -> Result<PrismaticSectionBatch<R>, CrossSectionError> {
        let topology = topology::classify(&self.context.definitions)?;
        let context = compilation::compile(Arc::clone(&self.context), topology)?;
        Ok(PrismaticSectionBatch {
            context: Arc::new(context),
        })
    }
}
