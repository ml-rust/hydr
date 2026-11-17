use super::{CrossSectionError, context::Context};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub fn any(&self, mask: &Tensor<R>) -> Result<bool, CrossSectionError> {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }
    pub(super) fn rejected_lane(
        &self,
        mask: &Tensor<R>,
    ) -> Result<Option<usize>, CrossSectionError> {
        if !self.any(mask)? {
            return Ok(None);
        }
        let selected = self.client.argmax(mask, 0, false)?.item::<i64>()?;
        let lane = usize::try_from(selected)
            .ok()
            .filter(|lane| *lane < mask.numel())
            .ok_or_else(|| {
                CrossSectionError::numerical(
                    "rejection mask has an invalid index. Inspect runtime indexing kernels",
                )
            })?;
        if mask.narrow(0, lane, 1)?.item::<f64>()? == 0.0 {
            return Err(CrossSectionError::numerical(
                "rejection mask contradicts its reduction. Inspect runtime comparison kernels",
            ));
        }
        Ok(Some(lane))
    }
    pub(super) fn invalid(&self, value: &Tensor<R>) -> Result<Tensor<R>, CrossSectionError> {
        Ok(self.client.maximum(
            &self.client.cast(&self.client.isnan(value)?, DType::F64)?,
            &self.client.cast(&self.client.isinf(value)?, DType::F64)?,
        )?)
    }
    pub fn field_rejection(
        &self,
        values: &Tensor<R>,
        positive_support: &Tensor<R>,
    ) -> Result<Tensor<R>, CrossSectionError> {
        Ok(self.client.maximum(
            &self.invalid(values)?,
            &self
                .client
                .mul(positive_support, &self.client.le(values, &self.zero)?)?,
        )?)
    }
    pub fn stage_input(&self, stages: &Tensor<R>) -> Result<(), CrossSectionError> {
        if stages.shape() != [self.definitions.len()]
            || stages.dtype() != DType::F64
            || !stages.device().is_same(self.client.device())
        {
            return Err(CrossSectionError::input(format!(
                "stage shape={:?}, dtype={:?}: supply one F64 stage per ordered section on the configured device",
                stages.shape(),
                stages.dtype()
            )));
        }
        let invalid = self.invalid(stages)?;
        let above_cap = self.client.gt(stages, &self.cap)?;
        if !self.any(&self.client.maximum(&invalid, &above_cap)?)? {
            return Ok(());
        }
        if let Some(index) = self.rejected_lane(&invalid)? {
            let value = stages.narrow(0, index, 1)?.item::<f64>()?;
            return Err(CrossSectionError::input(format!(
                "stage={value}: supply a finite stage in the section's declared vertical datum"
            ))
            .at(index));
        }
        if let Some(index) = self.rejected_lane(&above_cap)? {
            return Err(CrossSectionError::UnsupportedStage {
                section: index,
                stage_m: stages.narrow(0, index, 1)?.item::<f64>()?,
                maximum_stage_m: self.definitions[index].maximum_stage_m,
            });
        }
        Ok(())
    }
    pub fn reject_segment(
        &self,
        mask: &Tensor<R>,
        values: &Tensor<R>,
        quantity: &str,
    ) -> Result<(), CrossSectionError> {
        if let Some(lane) = self.rejected_lane(mask)? {
            let section = *self.segment_sections.get(lane).ok_or_else(|| {
                CrossSectionError::numerical(
                    "rejected segment has no source section. Rebuild the immutable geometry",
                )
            })?;
            let value = values.narrow(0, lane, 1)?.item::<f64>()?;
            return Err(CrossSectionError::numerical(format!("{quantity}={value}, segment={lane}: rescale profile coordinates or query a representable stage")).at(section));
        }
        Ok(())
    }
    pub fn reject_section(
        &self,
        mask: &Tensor<R>,
        values: &Tensor<R>,
        quantity: &str,
    ) -> Result<(), CrossSectionError> {
        if let Some(section) = self.rejected_lane(mask)? {
            let value = values.narrow(0, section, 1)?.item::<f64>()?;
            return Err(CrossSectionError::numerical(format!(
                "{quantity}={value}: rescale profile coordinates or query a representable stage"
            ))
            .at(section));
        }
        Ok(())
    }
    pub fn segment_finite(
        &self,
        values: &Tensor<R>,
        quantity: &str,
    ) -> Result<(), CrossSectionError> {
        self.reject_segment(&self.invalid(values)?, values, quantity)
    }
    pub fn section_finite(
        &self,
        values: &Tensor<R>,
        quantity: &str,
    ) -> Result<(), CrossSectionError> {
        self.reject_section(&self.invalid(values)?, values, quantity)
    }
}
