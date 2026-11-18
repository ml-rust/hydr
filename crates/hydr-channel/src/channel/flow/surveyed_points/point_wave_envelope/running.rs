use super::super::{
    point_flux_masks::flag, wave_curve_types::CurveValue, wet_wave_types::WetWaveSide,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Running bounds. Empty rows hold +inf and -inf until a value is included.
pub(super) struct Running<R: Runtime<DType = DType>> {
    pub(super) lower: Tensor<R>,
    pub(super) upper: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Running<R> {
    pub(super) fn new(a: &Arithmetic<'_, R>, shape: &[usize]) -> HydraulicResult<Self>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let device = a.client().device();
        Ok(Self {
            lower: Tensor::full_scalar(shape, a.dtype(), f64::INFINITY, device)?,
            upper: Tensor::full_scalar(shape, a.dtype(), f64::NEG_INFINITY, device)?,
        })
    }

    /// Joins the outward physical bounds of `value` where `mask` is one and the value is available.
    pub(super) fn include(
        &mut self,
        a: &Arithmetic<'_, R>,
        mask: &Tensor<R>,
        value: &CurveValue<R>,
    ) -> HydraulicResult<()>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let c = a.client();
        let shape = mask.shape();
        let take = c.mul(mask, &positive(a, &value.available)?)?;
        let take = c.gt(&take, &zeros(a, shape)?)?;
        let empty = Self::new(a, shape)?;
        self.lower = c.minimum(
            &self.lower,
            &c.where_cond(&take, value.physical.lower(), &empty.lower)?,
        )?;
        self.upper = c.maximum(
            &self.upper,
            &c.where_cond(&take, value.physical.upper(), &empty.upper)?,
        )?;
        Ok(())
    }

    /// Joins every possible branch speed of one wet side and returns its validity mask.
    /// A possible branch needs its speed: the shock speed, or the head and the rarefaction tail.
    pub(super) fn include_side(
        &mut self,
        a: &Arithmetic<'_, R>,
        side: &WetWaveSide<R>,
    ) -> HydraulicResult<Tensor<R>>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let c = a.client();
        let shock = positive(a, &side.shock_possible)?;
        let rarefaction = positive(a, &side.rarefaction_possible)?;
        let head = c.maximum(&rarefaction, &positive(a, &side.equality_possible)?)?;
        let tail = positive(a, &side.tail.available)?;
        self.include(a, &shock, &side.shock_speed)?;
        self.include(a, &head, &side.head)?;
        self.include(a, &tail, &side.tail)?;
        let mut ok = positive(a, &side.valid)?;
        ok = c.mul(&ok, &c.maximum(&shock, &head)?)?;
        let shock_speed = positive(a, &side.shock_speed.available)?;
        ok = c.mul(&ok, &require(a, &shock, &shock_speed)?)?;
        ok = c.mul(
            &ok,
            &require(a, &head, &positive(a, &side.head.available)?)?,
        )?;
        ok = c.mul(&ok, &require(a, &rarefaction, &tail)?)?;
        Ok(ok)
    }
}

/// One where `needed` is zero or `available` is one.
pub(super) fn require<R>(
    a: &Arithmetic<'_, R>,
    needed: &Tensor<R>,
    available: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
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
    let one = ones(a, needed.shape())?;
    Ok(c.sub(&one, &c.mul(needed, &c.sub(&one, available)?)?)?)
}

/// Numeric zero-one mask of the entries above zero.
pub(super) fn positive<R>(a: &Arithmetic<'_, R>, values: &Tensor<R>) -> HydraulicResult<Tensor<R>>
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
    let zero = Tensor::zeros(values.shape(), values.dtype(), c.device())?;
    flag(a, &c.gt(values, &zero)?)
}

pub(super) fn ones<R>(a: &Arithmetic<'_, R>, shape: &[usize]) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(Tensor::ones(shape, a.dtype(), a.client().device())?)
}

pub(super) fn zeros<R>(a: &Arithmetic<'_, R>, shape: &[usize]) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(Tensor::zeros(shape, a.dtype(), a.client().device())?)
}
