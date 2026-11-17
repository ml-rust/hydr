use super::super::hydraulic_geometry::GatheredBands;
use super::{
    admission,
    chart::{AreaQuery, SupportedAreaQuery},
    support_assessment::SupportAssessment,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};

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
    pub(crate) fn prove_support(
        self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
    ) -> HydraulicResult<SupportedAreaQuery<R>> {
        let assessment = SupportAssessment::assess(a, bands, &self)?;
        let c = a.client();
        admission::reject(
            a,
            &c.lt(assessment.area.lower(), &assessment.zero_tensor)?,
            "area query image is not proved nonnegative. Change the chart or tighten its certificates",
        )?;
        admission::reject(
            a,
            &c.lt(assessment.deficit.lower(), &assessment.zero_tensor)?,
            "area query image is not proved below its physical cap. Change the chart or tighten its certificates",
        )?;
        assessment.finish(a, bands, self)
    }
}
