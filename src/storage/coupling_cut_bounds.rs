//! Directed native arithmetic for physical affine inverse-volume enclosures.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::LevelPoolError;
use crate::tensor::level_pool::TensorPoolContext;

pub(super) struct CutArithmetic<'a, R: Runtime<DType = DType>> {
    pub context: &'a TensorPoolContext<R>,
}

impl<R: Runtime<DType = DType>> CutArithmetic<'_, R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    fn rounded(&self, value: &Tensor<R>, upper: bool) -> Result<Tensor<R>, LevelPoolError> {
        self.context.finite(value, "cut arithmetic operand")?;
        let bound = crate::numerics::storage::bounds::rounding(self.context, value)?;
        let c = &self.context.client;
        let result = if upper {
            c.add(value, &bound)?
        } else {
            c.sub(value, &bound)?
        };
        self.context.finite(&result, "directed cut arithmetic")?;
        Ok(result)
    }

    pub fn add(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        upper: bool,
    ) -> Result<Tensor<R>, LevelPoolError> {
        let c = &self.context.client;
        let (high, low) = c.two_sum(a, b)?;
        self.context.finite(&high, "cut TwoSum high")?;
        self.context.finite(&low, "cut TwoSum residual")?;
        let zero = c.mul_scalar(&high, 0.0)?;
        let needs_rounding = if upper {
            c.gt(&low, &zero)?
        } else {
            c.lt(&low, &zero)?
        };
        let active = c.where_cond(&needs_rounding, &high, &zero)?;
        Ok(c.where_cond(&needs_rounding, &self.rounded(&active, upper)?, &high)?)
    }

    pub fn difference(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), LevelPoolError> {
        let negative = self.context.client.neg(b)?;
        Ok((
            self.add(a, &negative, false)?,
            self.add(a, &negative, true)?,
        ))
    }

    pub fn inverse(
        &self,
        query: &Tensor<R>,
        stage0: &Tensor<R>,
        stage1: &Tensor<R>,
        volume0: &Tensor<R>,
        volume1: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), LevelPoolError> {
        let c = &self.context.client;
        let zero = c.mul_scalar(query, 0.0)?;
        let (dz_lower, dz_upper) = self.difference(query, stage0)?;
        let (dh_lower, dh_upper) = self.difference(stage1, stage0)?;
        let (dv_lower, dv_upper) = self.difference(volume1, volume0)?;
        self.context.reject(
            &c.le(&dh_lower, &zero)?,
            "cut stage difference has no positive certified lower bound",
        )?;
        self.context.reject(
            &c.lt(&c.minimum(&dz_lower, &dv_lower)?, &zero)?,
            "cut positive affine operands lack nonnegative bounds",
        )?;
        let ratio_lower = self.rounded(&c.div(&dz_lower, &dh_upper)?, false)?;
        let ratio_lower = c.maximum(&ratio_lower, &zero)?;
        let ratio_upper = self.rounded(&c.div(&dz_upper, &dh_lower)?, true)?;
        let increment_lower = self.rounded(&c.mul(&ratio_lower, &dv_lower)?, false)?;
        let increment_lower = c.maximum(&increment_lower, &zero)?;
        let increment_upper = self.rounded(&c.mul(&ratio_upper, &dv_upper)?, true)?;
        let lower = self.add(volume0, &increment_lower, false)?;
        let upper = self.add(volume0, &increment_upper, true)?;
        // Monotonic support supplies additional exact endpoint bounds.
        Ok((c.maximum(&lower, volume0)?, c.minimum(&upper, volume1)?))
    }
}
