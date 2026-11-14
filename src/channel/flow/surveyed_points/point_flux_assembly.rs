use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

pub(super) struct Assembly<R: Runtime<DType = DType>> {
    pub(super) water: Interval<R>,
    pub(super) momentum: Interval<R>,
    pub(super) seen: Tensor<R>,
}
impl<R> Assembly<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn new(a: &mut Arithmetic<'_, R>, count: usize) -> HydraulicResult<Self> {
        let zero = Tensor::zeros(&[count], a.dtype(), a.client().device())?;
        Ok(Self {
            water: a.point(&zero)?,
            momentum: a.point(&zero)?,
            seen: zero,
        })
    }
    pub(super) fn merge(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owners: &Tensor<R>,
        water: &Interval<R>,
        momentum: &Interval<R>,
    ) -> HydraulicResult<()> {
        let c = a.client();
        let seen = c.index_select(&self.seen, 0, owners)?;
        self.water = merge(a, &self.water, owners, &seen, water)?;
        self.momentum = merge(a, &self.momentum, owners, &seen, momentum)?;
        self.seen = c.index_put(
            &self.seen,
            0,
            owners,
            &Tensor::ones(owners.shape(), a.dtype(), c.device())?,
        )?;
        Ok(())
    }
}

fn merge<R>(
    a: &mut Arithmetic<'_, R>,
    destination: &Interval<R>,
    owners: &Tensor<R>,
    seen: &Tensor<R>,
    source: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let lower = c.index_select(destination.lower(), 0, owners)?;
    let upper = c.index_select(destination.upper(), 0, owners)?;
    let lower = c.where_cond(seen, &c.minimum(&lower, source.lower())?, source.lower())?;
    let upper = c.where_cond(seen, &c.maximum(&upper, source.upper())?, source.upper())?;
    Ok(a.bounds(
        &c.index_put(destination.lower(), 0, owners, &lower)?,
        &c.index_put(destination.upper(), 0, owners, &upper)?,
    )?)
}

pub(super) fn hull<R>(
    a: &mut Arithmetic<'_, R>,
    left: &Interval<R>,
    right: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.bounds(
        &a.client().minimum(left.lower(), right.lower())?,
        &a.client().maximum(left.upper(), right.upper())?,
    )?)
}
