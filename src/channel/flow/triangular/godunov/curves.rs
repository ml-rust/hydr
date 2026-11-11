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

    fn triangular_curve_nominal(
        &self,
        geometry: &TriangleGeometry<R>,
        depth: &Tensor<R>,
        reference: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let root = c.sqrt(depth)?;
        let root_k = c.sqrt(reference)?;
        let difference = c.sub(depth, reference)?;
        let denominator = c.add(&root, &root_k)?;
        let root_factor = c.div(&geometry.root_gravity, &geometry.root_two)?;
        let rare = c.mul_scalar(
            &c.mul(&root_factor, &c.div(&difference, &denominator)?)?,
            4.0,
        )?;
        let h = c.maximum(depth, reference)?;
        let ratio = c.div(reference, &h)?;
        let poly = c.mul(
            &c.add_scalar(&ratio, 1.0)?,
            &c.add_scalar(&c.add(&ratio, &c.mul(&ratio, &ratio)?)?, 1.0)?,
        )?;
        let sqrt_three = self.constant(depth, 3.0)?;
        let sqrt_three = c.sqrt(&sqrt_three)?;
        let scale = c.div(&c.mul(&geometry.root_gravity, &c.sqrt(&h)?)?, &sqrt_three)?;
        let jump = c.sub(&h, reference)?;
        let shock = c.mul(&c.mul(&scale, &c.div(&jump, reference)?)?, &c.sqrt(&poly)?)?;
        let nominal = c.where_cond(&c.gt(depth, reference)?, &shock, &rare)?;
        self.finite(&nominal, "triangular barotropic curve")?;
        Ok(nominal)
    }

    fn triangular_curve_enclosure(
        &self,
        geometry: &TriangleGeometry<R>,
        depth: &Tensor<R>,
        reference: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.client;
        let zero = self.constant(depth, 0.0)?;
        if !self.all(&c.gt(reference, &zero)?)? {
            return Err(ChannelError::numerical(
                "triangular reference depth has a nonpositive bound: mask dry lanes or rescale positive depths",
            ));
        }
        let root = c.sqrt(depth)?;
        let root_k = c.sqrt(reference)?;
        let difference = c.sub(depth, reference)?;
        let dl = lower::<R>(c, &difference)?;
        let du = upper::<R>(c, &difference)?;
        let den_l = lower::<R>(
            c,
            &c.add(
                &c.maximum(&lower::<R>(c, &root)?, &zero)?,
                &c.maximum(&lower::<R>(c, &root_k)?, &zero)?,
            )?,
        )?;
        let den_u = upper::<R>(c, &c.add(&upper::<R>(c, &root)?, &upper::<R>(c, &root_k)?)?)?;
        let factor_l = lower::<R>(
            c,
            &c.div(&geometry.root_gravity_lower, &geometry.root_two_upper)?,
        )?;
        let factor_u = upper::<R>(
            c,
            &c.div(&geometry.root_gravity_upper, &geometry.root_two_lower)?,
        )?;
        let rare_l = lower::<R>(
            c,
            &c.mul_scalar(
                &lower::<R>(
                    c,
                    &c.mul(
                        &c.where_cond(&c.lt(&dl, &zero)?, &factor_u, &factor_l)?,
                        &lower::<R>(
                            c,
                            &c.div(&dl, &c.where_cond(&c.lt(&dl, &zero)?, &den_l, &den_u)?)?,
                        )?,
                    )?,
                )?,
                4.0,
            )?,
        )?;
        let rare_u = upper::<R>(
            c,
            &c.mul_scalar(
                &upper::<R>(
                    c,
                    &c.mul(
                        &c.where_cond(&c.lt(&du, &zero)?, &factor_l, &factor_u)?,
                        &upper::<R>(
                            c,
                            &c.div(&du, &c.where_cond(&c.lt(&du, &zero)?, &den_u, &den_l)?)?,
                        )?,
                    )?,
                )?,
                4.0,
            )?,
        )?;
        // Shock operands stay in their physical branch, including eagerly evaluated rarefaction lanes.
        let h = c.maximum(depth, reference)?;
        let ratio = c.div(reference, &h)?;
        let ratio_l = c.maximum(&lower::<R>(c, &ratio)?, &zero)?;
        let ratio_u = upper::<R>(c, &ratio)?;
        let poly_l = c.maximum(
            &lower::<R>(
                c,
                &c.mul(
                    &lower::<R>(c, &c.add_scalar(&ratio_l, 1.0)?)?,
                    &lower::<R>(
                        c,
                        &c.add_scalar(
                            &lower::<R>(
                                c,
                                &c.add(
                                    &ratio_l,
                                    &c.maximum(
                                        &lower::<R>(c, &c.mul(&ratio_l, &ratio_l)?)?,
                                        &zero,
                                    )?,
                                )?,
                            )?,
                            1.0,
                        )?,
                    )?,
                )?,
            )?,
            &zero,
        )?;
        let poly_u = upper::<R>(
            c,
            &c.mul(
                &upper::<R>(c, &c.add_scalar(&ratio_u, 1.0)?)?,
                &upper::<R>(
                    c,
                    &c.add_scalar(
                        &upper::<R>(
                            c,
                            &c.add(&ratio_u, &upper::<R>(c, &c.mul(&ratio_u, &ratio_u)?)?)?,
                        )?,
                        1.0,
                    )?,
                )?,
            )?,
        )?;
        let sqrt_three = self.constant(depth, 3.0)?;
        let sqrt_three = c.sqrt(&sqrt_three)?;
        let scale_l = c.maximum(
            &lower::<R>(
                c,
                &c.div(
                    &lower::<R>(
                        c,
                        &c.mul(
                            &geometry.root_gravity_lower,
                            &c.maximum(&lower::<R>(c, &c.sqrt(&h)?)?, &zero)?,
                        )?,
                    )?,
                    &upper::<R>(c, &sqrt_three)?,
                )?,
            )?,
            &zero,
        )?;
        let scale_u = upper::<R>(
            c,
            &c.div(
                &upper::<R>(
                    c,
                    &c.mul(&geometry.root_gravity_upper, &upper::<R>(c, &c.sqrt(&h)?)?)?,
                )?,
                &lower::<R>(c, &sqrt_three)?,
            )?,
        )?;
        let jump = c.sub(&h, reference)?;
        let shock_l = c.maximum(
            &lower::<R>(
                c,
                &c.mul(
                    &c.maximum(
                        &lower::<R>(
                            c,
                            &c.mul(
                                &scale_l,
                                &c.maximum(
                                    &lower::<R>(
                                        c,
                                        &c.div(
                                            &c.maximum(&lower::<R>(c, &jump)?, &zero)?,
                                            reference,
                                        )?,
                                    )?,
                                    &zero,
                                )?,
                            )?,
                        )?,
                        &zero,
                    )?,
                    &c.maximum(&lower::<R>(c, &c.sqrt(&poly_l)?)?, &zero)?,
                )?,
            )?,
            &zero,
        )?;
        let shock_u = upper::<R>(
            c,
            &c.mul(
                &upper::<R>(
                    c,
                    &c.mul(
                        &scale_u,
                        &upper::<R>(c, &c.div(&upper::<R>(c, &jump)?, reference)?)?,
                    )?,
                )?,
                &upper::<R>(c, &c.sqrt(&poly_u)?)?,
            )?,
        )?;
        let is_shock = c.gt(depth, reference)?;
        let lo = c.where_cond(&is_shock, &shock_l, &rare_l)?;
        let hi = c.where_cond(&is_shock, &shock_u, &rare_u)?;
        self.finite(&lo, "triangular lower barotropic curve")?;
        self.finite(&hi, "triangular upper barotropic curve")?;
        Ok((lo, hi))
    }
}
