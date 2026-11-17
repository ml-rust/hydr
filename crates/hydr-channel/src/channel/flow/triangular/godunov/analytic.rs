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

pub(super) struct DepthVelocity<R: Runtime<DType = DType>> {
    pub depth: Tensor<R>,
    pub lower_depth: Tensor<R>,
    pub upper_depth: Tensor<R>,
    pub velocity: Tensor<R>,
    pub lower_velocity: Tensor<R>,
    pub upper_velocity: Tensor<R>,
    pub sound: Tensor<R>,
    pub lower_sound: Tensor<R>,
    pub upper_sound: Tensor<R>,
}

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
    pub(super) fn triangular_sound(
        &self,
        geometry: &TriangleGeometry<R>,
        depth: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        Ok(self.client.div(
            &self
                .client
                .mul(&geometry.root_gravity, &self.client.sqrt(depth)?)?,
            &geometry.root_two,
        )?)
    }

    pub(super) fn triangular_analytic_state(
        &self,
        geometry: &TriangleGeometry<R>,
        sound: ValueBounds<R>,
        velocity: ValueBounds<R>,
    ) -> Result<DepthVelocity<R>, ChannelError> {
        let ValueBounds {
            nominal: sound,
            lower: sound_lower,
            upper: sound_upper,
        } = sound;
        let ValueBounds {
            nominal: velocity,
            lower: lower_velocity,
            upper: upper_velocity,
        } = velocity;
        let c = &self.client;
        let zero = self.constant(&sound, 0.0)?;
        let root = c.div(&c.mul(&sound, &geometry.root_two)?, &geometry.root_gravity)?;
        let root_lower = c.maximum(
            &lower::<R>(
                c,
                &c.div(
                    &c.maximum(
                        &lower::<R>(c, &c.mul(&sound_lower, &geometry.root_two_lower)?)?,
                        &zero,
                    )?,
                    &geometry.root_gravity_upper,
                )?,
            )?,
            &zero,
        )?;
        let root_upper = upper::<R>(
            c,
            &c.div(
                &upper::<R>(c, &c.mul(&sound_upper, &geometry.root_two_upper)?)?,
                &geometry.root_gravity_lower,
            )?,
        )?;
        let depth = c.mul(&root, &root)?;
        self.represented(
            &c.gt(&sound, &zero)?,
            &depth,
            "Godunov positive sampled depth",
        )?;
        let lower_depth = c.maximum(&lower::<R>(c, &c.mul(&root_lower, &root_lower)?)?, &zero)?;
        let upper_depth = upper::<R>(c, &c.mul(&root_upper, &root_upper)?)?;
        Ok(DepthVelocity {
            depth,
            lower_depth,
            upper_depth,
            velocity,
            lower_velocity,
            upper_velocity,
            sound,
            lower_sound: sound_lower,
            upper_sound: sound_upper,
        })
    }

    pub(super) fn triangular_sonic_state(
        &self,
        geometry: &TriangleGeometry<R>,
        face: &Face<R>,
        side: &SideBounds<R>,
        right: bool,
    ) -> Result<DepthVelocity<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(&face.area, 0.0)?;
        let sound = self.triangular_sound(geometry, &face.depth)?;
        let four = c.mul_scalar(&sound, 4.0)?;
        let native = if right {
            c.sub(&four, &face.velocity)?
        } else {
            c.add(&face.velocity, &four)?
        };
        let four_lower = lower::<R>(c, &c.mul_scalar(&side.lower_sound, 4.0)?)?;
        let four_upper = upper::<R>(c, &c.mul_scalar(&side.upper_sound, 4.0)?)?;
        let lo = if right {
            lower::<R>(c, &c.sub(&four_lower, &side.upper_velocity)?)?
        } else {
            lower::<R>(c, &c.add(&side.lower_velocity, &four_lower)?)?
        };
        let hi = if right {
            upper::<R>(c, &c.sub(&four_upper, &side.lower_velocity)?)?
        } else {
            upper::<R>(c, &c.add(&side.upper_velocity, &four_upper)?)?
        };
        let five = self.constant(&zero, 5.0)?;
        let sound = c.maximum(&c.div(&native, &five)?, &zero)?;
        let sl = c.maximum(&lower::<R>(c, &c.div(&lo, &five)?)?, &zero)?;
        let sh = c.maximum(&upper::<R>(c, &c.div(&hi, &five)?)?, &zero)?;
        let (velocity, ul, uh) = if right {
            (c.neg(&sound)?, c.neg(&sh)?, c.neg(&sl)?)
        } else {
            (sound.clone(), sl.clone(), sh.clone())
        };
        self.triangular_analytic_state(
            geometry,
            ValueBounds {
                nominal: sound,
                lower: sl,
                upper: sh,
            },
            ValueBounds {
                nominal: velocity,
                lower: ul,
                upper: uh,
            },
        )
    }

    pub(super) fn triangular_two_rare_state(
        &self,
        geometry: &TriangleGeometry<R>,
        left: &Face<R>,
        right: &Face<R>,
        ls: &SideBounds<R>,
        rs: &SideBounds<R>,
    ) -> Result<DepthVelocity<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(&left.area, 0.0)?;
        let cl = self.triangular_sound(geometry, &left.depth)?;
        let cr = self.triangular_sound(geometry, &right.depth)?;
        let mean = c.add(&c.mul_scalar(&cl, 0.5)?, &c.mul_scalar(&cr, 0.5)?)?;
        let sound = c.maximum(
            &c.sub(
                &mean,
                &c.mul_scalar(&c.sub(&right.velocity, &left.velocity)?, 0.125)?,
            )?,
            &zero,
        )?;
        let mean_lower = lower::<R>(
            c,
            &c.add(
                &lower::<R>(c, &c.mul_scalar(&ls.lower_sound, 0.5)?)?,
                &lower::<R>(c, &c.mul_scalar(&rs.lower_sound, 0.5)?)?,
            )?,
        )?;
        let mean_upper = upper::<R>(
            c,
            &c.add(
                &upper::<R>(c, &c.mul_scalar(&ls.upper_sound, 0.5)?)?,
                &upper::<R>(c, &c.mul_scalar(&rs.upper_sound, 0.5)?)?,
            )?,
        )?;
        let jump_lower = lower::<R>(c, &c.sub(&rs.lower_velocity, &ls.upper_velocity)?)?;
        let jump_upper = upper::<R>(c, &c.sub(&rs.upper_velocity, &ls.lower_velocity)?)?;
        let sl = c.maximum(
            &lower::<R>(
                c,
                &c.sub(
                    &mean_lower,
                    &upper::<R>(c, &c.mul_scalar(&jump_upper, 0.125)?)?,
                )?,
            )?,
            &zero,
        )?;
        let sh = c.maximum(
            &upper::<R>(
                c,
                &c.sub(
                    &mean_upper,
                    &lower::<R>(c, &c.mul_scalar(&jump_lower, 0.125)?)?,
                )?,
            )?,
            &zero,
        )?;
        let velocity = c.add(
            &c.add(
                &c.mul_scalar(&left.velocity, 0.5)?,
                &c.mul_scalar(&right.velocity, 0.5)?,
            )?,
            &c.mul_scalar(&c.sub(&cl, &cr)?, 2.0)?,
        )?;
        let vl = lower::<R>(
            c,
            &c.add(
                &lower::<R>(
                    c,
                    &c.add(
                        &lower::<R>(c, &c.mul_scalar(&ls.lower_velocity, 0.5)?)?,
                        &lower::<R>(c, &c.mul_scalar(&rs.lower_velocity, 0.5)?)?,
                    )?,
                )?,
                &lower::<R>(
                    c,
                    &c.mul_scalar(
                        &lower::<R>(c, &c.sub(&ls.lower_sound, &rs.upper_sound)?)?,
                        2.0,
                    )?,
                )?,
            )?,
        )?;
        let vh = upper::<R>(
            c,
            &c.add(
                &upper::<R>(
                    c,
                    &c.add(
                        &upper::<R>(c, &c.mul_scalar(&ls.upper_velocity, 0.5)?)?,
                        &upper::<R>(c, &c.mul_scalar(&rs.upper_velocity, 0.5)?)?,
                    )?,
                )?,
                &upper::<R>(
                    c,
                    &c.mul_scalar(
                        &upper::<R>(c, &c.sub(&ls.upper_sound, &rs.lower_sound)?)?,
                        2.0,
                    )?,
                )?,
            )?,
        )?;
        self.triangular_analytic_state(
            geometry,
            ValueBounds {
                nominal: sound,
                lower: sl,
                upper: sh,
            },
            ValueBounds {
                nominal: velocity,
                lower: vl,
                upper: vh,
            },
        )
    }
}
