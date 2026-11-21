//! Callback interface of one certified lane Simpson integration.

use super::error::{Arithmetic, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

/// Simpson callbacks of one integration call, invoked in the order `values`, `nominal`, `fourth`.
pub(crate) trait SimpsonCallbacks<R: Runtime<DType = DType>> {
    /// Function enclosures over the value-node coordinate intervals `x`.
    fn values(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        x: &Interval<R>,
    ) -> HydraulicResult<Interval<R>>;

    /// Function enclosures at the represented value-node points, for the owners of `values`.
    fn nominal(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        point: &Tensor<R>,
    ) -> HydraulicResult<Interval<R>>;

    /// Absolute fourth-derivative bounds over the panel intervals `x`.
    fn fourth(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        x: &Interval<R>,
    ) -> HydraulicResult<Tensor<R>>;
}

/// One value closure serves both enclosure and nominal queries.
pub(super) struct Closures<F, D> {
    pub(super) value: F,
    pub(super) fourth: D,
}

impl<R, F, D> SimpsonCallbacks<R> for Closures<F, D>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
    F: FnMut(&mut Arithmetic<'_, R>, &Tensor<R>, &Interval<R>) -> HydraulicResult<Interval<R>>,
    D: FnMut(&mut Arithmetic<'_, R>, &Tensor<R>, &Interval<R>) -> HydraulicResult<Tensor<R>>,
{
    fn values(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        x: &Interval<R>,
    ) -> HydraulicResult<Interval<R>> {
        (self.value)(a, owner, x)
    }

    fn nominal(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        point: &Tensor<R>,
    ) -> HydraulicResult<Interval<R>> {
        let point = a.point(point)?;
        (self.value)(a, owner, &point)
    }

    fn fourth(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        x: &Interval<R>,
    ) -> HydraulicResult<Tensor<R>> {
        (self.fourth)(a, owner, x)
    }
}
