use super::super::hydraulic_geometry::GatheredBands;
use super::{
    chart::{AreaQuery, SupportedAreaQuery},
    support_assessment::SupportAssessment,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Whole-image support decisions in input order and certificates for passing rows only.
pub(crate) struct SupportSelection<R: Runtime<DType = DType>> {
    /// Mask in input order: one for proved support, zero otherwise.
    pub(crate) proved: Tensor<R>,
    /// Zero-based `I64` indices into this invocation, not section owners or diagnostic row IDs.
    pub(crate) owner: Tensor<R>,
    /// Certificates in `owner` order. `None` when no row proves support.
    pub(crate) query: Option<SupportedAreaQuery<R>>,
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
    /// Select rows whose complete area image proves physical support.
    /// Results remain provisional until the enclosing checked call completes.
    pub(crate) fn select_supported(
        self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
    ) -> HydraulicResult<SupportSelection<R>> {
        let assessment = SupportAssessment::assess(a, bands, &self)?;
        let c = a.client();
        let proved = c.mul(
            &c.ge(assessment.area.lower(), &assessment.zero_tensor)?,
            &c.ge(assessment.deficit.lower(), &assessment.zero_tensor)?,
        )?;
        let owner = c.nonzero(&proved)?.flatten()?;
        let query = if owner.numel() == 0 {
            None
        } else {
            let query = self.select(a, &owner)?;
            let bands = bands.select(a, &owner)?;
            let assessment = assessment.select(a, &owner)?;
            Some(assessment.finish(a, &bands, query)?)
        };
        Ok(SupportSelection {
            proved,
            owner,
            query,
        })
    }
}
