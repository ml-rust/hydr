//! Model-bound attachment cuts and original-water donor availability.

use std::sync::Arc;

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

use super::coupling_cut_bounds::CutArithmetic;
use super::{LevelPoolBatch, LevelPoolError};
use crate::storage::tensor::TensorPoolContext;

mod bind;
#[cfg(test)]
pub(super) mod fixture;

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct LevelPoolCut {
    pub pool_index: usize,
    pub elevation_m: f64,
}

#[doc(hidden)]
#[derive(Debug)]
pub struct LevelPoolCutPlan<R: Runtime<DType = DType>> {
    context: Arc<TensorPoolContext<R>>,
    indices: Tensor<R>,
    #[cfg(test)]
    nominal_m3: Tensor<R>,
    #[cfg(test)]
    lower_m3: Tensor<R>,
    upper_m3: Tensor<R>,
}

#[doc(hidden)]
pub struct LevelPoolCutAvailability<R: Runtime<DType = DType>> {
    pub available_lower_m3: Tensor<R>,
}

impl<R: Runtime<DType = DType>> LevelPoolCutPlan<R> {
    #[cfg(test)]
    pub(crate) fn nominal_volume_m3(&self) -> &Tensor<R> {
        &self.nominal_m3
    }

    #[cfg(test)]
    pub(crate) fn lower_volume_m3(&self) -> &Tensor<R> {
        &self.lower_m3
    }

    #[doc(hidden)]
    pub fn upper_volume_m3(&self) -> &Tensor<R> {
        &self.upper_m3
    }
}

impl<R: Runtime<DType = DType>> LevelPoolBatch<R>
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
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    #[doc(hidden)]
    pub fn coupling_cut_availability(
        &self,
        plan: &LevelPoolCutPlan<R>,
    ) -> Result<LevelPoolCutAvailability<R>, LevelPoolError> {
        if !Arc::ptr_eq(&self.context, &plan.context) {
            return Err(LevelPoolError::input(
                "cut plan belongs to another model/client. Bind cuts on this pool batch",
            ));
        }
        if plan.upper_m3.numel() == 0 {
            return Ok(LevelPoolCutAvailability {
                available_lower_m3: plan.upper_m3.clone(),
            });
        }
        let c = &self.context.client;
        let original = c.index_select(&self.state.volume_m3, 0, &plan.indices)?;
        let arithmetic = CutArithmetic {
            context: &self.context,
        };
        let lower = arithmetic.add(&original, &c.neg(&plan.upper_m3)?, false)?;
        let zero = c.mul_scalar(&lower, 0.0)?;
        Ok(LevelPoolCutAvailability {
            available_lower_m3: c.maximum(&lower, &zero)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::coupling_cuts::fixture::batch;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn availability_rounds_subtraction_down_without_water_tolerance_credit() -> TestResult {
        let small = 2.0f64.powi(-54);
        let batch = batch(&[0.0, small, 2.0], &[0.0, 1.0, 2.0], 1.0)?;
        let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
            pool_index: 0,
            elevation_m: 1.0,
        }])?;
        assert_eq!(
            plan.upper_volume_m3().try_to_vec::<f64>()?[0].to_bits(),
            small.to_bits()
        );
        let available = batch
            .coupling_cut_availability(&plan)?
            .available_lower_m3
            .try_to_vec::<f64>()?[0];
        assert!(available > 0.0 && available < 1.0);
        Ok(())
    }

    #[test]
    fn zero_certified_availability_grants_no_positive_water() -> TestResult {
        let batch = batch(&[0.0, 3.0, 11.0], &[-2.0, 0.0, 4.0], 6.0)?;
        let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
            pool_index: 0,
            elevation_m: 1.5,
        }])?;
        assert_eq!(
            batch
                .coupling_cut_availability(&plan)?
                .available_lower_m3
                .try_to_vec::<f64>()?,
            vec![0.0]
        );
        Ok(())
    }
}
