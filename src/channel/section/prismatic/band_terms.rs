use crate::channel::flow::surveyed::error::Arithmetic;
use numr::dtype::DType;
use numr::ops::{
    CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, CertifiedExpansion};

/// Model coefficients of one band for each row, with matching shapes.
pub(super) struct BandTerms<R: Runtime<DType = DType>> {
    /// Knot area, in square metres.
    pub area: CertifiedExpansion<R>,
    /// Six times the knot integral, in cubic metres.
    pub six_integral: CertifiedExpansion<R>,
    /// Knot width as an exact F64 point, in metres.
    pub width: CertifiedExpansion<R>,
    /// Width slope with respect to depth.
    pub slope: CertifiedExpansion<R>,
}

impl<R> BandTerms<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Model area `A + s*(w + s*m/2)` at depth `s` above the knot.
    pub fn area_at(
        &self,
        a: &mut Arithmetic<'_, R>,
        depth: &CertifiedExpansion<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        let half = constant(a, depth, 0.5)?;
        let half_slope = a.expansion_mul(&self.slope, &half)?;
        let rise = a.expansion_mul(&half_slope, depth)?;
        let mean = a.expansion_add(&self.width, &rise)?;
        let gain = a.expansion_mul(&mean, depth)?;
        a.expansion_add(&self.area, &gain)
    }

    /// Model width `w + m*s` at depth `s` above the knot.
    pub fn width_at(
        &self,
        a: &mut Arithmetic<'_, R>,
        depth: &CertifiedExpansion<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        let rise = a.expansion_mul(&self.slope, depth)?;
        a.expansion_add(&self.width, &rise)
    }

    /// Six times the model integral, `6I + s*(6A + s*(3w + m*s))`, at depth `s` above the knot.
    pub fn six_integral_at(
        &self,
        a: &mut Arithmetic<'_, R>,
        depth: &CertifiedExpansion<R>,
    ) -> ArithmeticResult<CertifiedExpansion<R>> {
        let three = constant(a, depth, 3.0)?;
        let six = constant(a, depth, 6.0)?;
        let slope_depth = a.expansion_mul(&self.slope, depth)?;
        let three_width = a.expansion_mul(&self.width, &three)?;
        let inner = a.expansion_add(&three_width, &slope_depth)?;
        let inner_depth = a.expansion_mul(&inner, depth)?;
        let six_area = a.expansion_mul(&self.area, &six)?;
        let middle = a.expansion_add(&six_area, &inner_depth)?;
        let middle_depth = a.expansion_mul(&middle, depth)?;
        a.expansion_add(&self.six_integral, &middle_depth)
    }
}

/// Select `yes` where `mask` is one and `no` elsewhere, component by component.
pub(super) fn choose<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    yes: &CertifiedExpansion<R>,
    no: &CertifiedExpansion<R>,
) -> ArithmeticResult<CertifiedExpansion<R>>
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
    a.expansion_bounds(
        &c.where_cond(mask, yes.high(), no.high())?,
        &c.where_cond(mask, yes.low(), no.low())?,
        &c.where_cond(mask, yes.abs_error(), no.abs_error())?,
    )
}

/// Exact F64 point with the shape of `like`.
pub(super) fn constant<R>(
    a: &mut Arithmetic<'_, R>,
    like: &CertifiedExpansion<R>,
    value: f64,
) -> ArithmeticResult<CertifiedExpansion<R>>
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
    let tensor = Tensor::full_scalar(like.high().shape(), DType::F64, value, c.device())?;
    a.expansion_point(&tensor)
}

/// Stack per-band columns along dimension one, component by component.
pub(super) fn stack<R>(
    a: &mut Arithmetic<'_, R>,
    columns: &[CertifiedExpansion<R>],
) -> ArithmeticResult<CertifiedExpansion<R>>
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
    let part = |component: fn(&CertifiedExpansion<R>) -> &Tensor<R>| {
        c.stack(&columns.iter().map(component).collect::<Vec<_>>(), 1)
    };
    let high = part(CertifiedExpansion::high)?;
    let low = part(CertifiedExpansion::low)?;
    let abs_error = part(CertifiedExpansion::abs_error)?;
    a.expansion_bounds(&high, &low, &abs_error)
}
