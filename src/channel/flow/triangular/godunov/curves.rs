use super::super::{
    envelope::{lower, upper},
    faces::Face,
    wave::SideBounds,
};
use super::types::ValueBounds;
use crate::channel::flow::{ChannelError, context::Context, geometry::TriangleGeometry};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    pub(in crate::channel::flow::triangular) fn triangular_curve_bounds(
        &self,
        geometry: &TriangleGeometry<R>,
        face: &Face<R>,
        side: &SideBounds<R>,
        depth: &Tensor<R>,
    ) -> Result<ValueBounds<R>, ChannelError> {
        let nominal = self.triangular_curve_nominal(geometry, depth, &face.depth)?;
        let (lo, _) = self.triangular_curve_enclosure(geometry, depth, &side.upper_depth)?;
        let (_, hi) = self.triangular_curve_enclosure(geometry, depth, &side.lower_depth)?;
        Ok(ValueBounds {
            nominal,
            lower: lo,
            upper: hi,
        })
    }

    pub(in crate::channel::flow::triangular) fn triangular_residual_bounds(
        &self,
        geometry: &TriangleGeometry<R>,
        left: &Face<R>,
        right: &Face<R>,
        left_side: &SideBounds<R>,
        right_side: &SideBounds<R>,
        depth: &Tensor<R>,
    ) -> Result<ValueBounds<R>, ChannelError> {
        let c = &self.client;
        let l = self.triangular_curve_bounds(geometry, left, left_side, depth)?;
        let r = self.triangular_curve_bounds(geometry, right, right_side, depth)?;
        let nominal = c.add(
            &c.add(&l.nominal, &r.nominal)?,
            &c.sub(&right.velocity, &left.velocity)?,
        )?;
        let lo = lower::<R>(
            c,
            &c.add(
                &lower::<R>(c, &c.add(&l.lower, &r.lower)?)?,
                &lower::<R>(
                    c,
                    &c.sub(&right_side.lower_velocity, &left_side.upper_velocity)?,
                )?,
            )?,
        )?;
        let hi = upper::<R>(
            c,
            &c.add(
                &upper::<R>(c, &c.add(&l.upper, &r.upper)?)?,
                &upper::<R>(
                    c,
                    &c.sub(&right_side.upper_velocity, &left_side.lower_velocity)?,
                )?,
            )?,
        )?;
        Ok(ValueBounds {
            nominal,
            lower: lo,
            upper: hi,
        })
    }

    pub(super) fn triangular_residual_nominal(
        &self,
        geometry: &TriangleGeometry<R>,
        left: &Face<R>,
        right: &Face<R>,
        depth: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let left_curve = self.triangular_curve_nominal(geometry, depth, &left.depth)?;
        let right_curve = self.triangular_curve_nominal(geometry, depth, &right.depth)?;
        let residual = c.add(
            &c.add(&left_curve, &right_curve)?,
            &c.sub(&right.velocity, &left.velocity)?,
        )?;
        self.finite(&residual, "triangular nominal star residual")?;
        Ok(residual)
    }
}
