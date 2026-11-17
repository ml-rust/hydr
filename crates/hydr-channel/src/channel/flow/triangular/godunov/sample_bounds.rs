use super::super::{
    envelope::{lower, upper},
    faces::Face,
};
use super::types::ValueBounds;
use crate::channel::flow::{ChannelError, context::Context, geometry::TriangleGeometry};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

pub(super) struct Sample<R: Runtime<DType = DType>> {
    pub face: Face<R>,
    pub area_bound: Tensor<R>,
    pub discharge_bound: Tensor<R>,
    pub lower_area: Tensor<R>,
    pub upper_area: Tensor<R>,
    pub lower_discharge: Tensor<R>,
    pub upper_discharge: Tensor<R>,
    pub lower_depth: Tensor<R>,
    pub upper_depth: Tensor<R>,
    pub lower_velocity: Tensor<R>,
    pub upper_velocity: Tensor<R>,
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
    pub(super) fn triangular_sample(
        &self,
        geometry: &TriangleGeometry<R>,
        depth: &ValueBounds<R>,
        velocity: &ValueBounds<R>,
    ) -> Result<Sample<R>, ChannelError> {
        let ValueBounds {
            nominal: depth,
            lower: dl,
            upper: dh,
        } = depth;
        let ValueBounds {
            nominal: velocity,
            lower: ul,
            upper: uh,
        } = velocity;
        let c = &self.client;
        let zero = self.constant(depth, 0.0)?;
        let kh = c.mul(&geometry.coefficient, depth)?;
        let kh_bound = self.product_bound(
            &geometry.coefficient,
            &zero,
            depth,
            &kh,
            "Godunov coefficient-depth product",
        )?;
        let area = c.mul(&kh, depth)?;
        let area_bound =
            self.product_bound(&kh, &kh_bound, depth, &area, "Godunov sampled section area")?;
        let discharge = c.mul(&area, velocity)?;
        let discharge_bound = self.product_bound(
            &area,
            &area_bound,
            velocity,
            &discharge,
            "Godunov sampled discharge",
        )?;
        let kl = c.maximum(&lower::<R>(c, &geometry.coefficient)?, &zero)?;
        let ku = upper::<R>(c, &geometry.coefficient)?;
        let lower_area = c.maximum(
            &lower::<R>(
                c,
                &c.mul(&c.maximum(&lower::<R>(c, &c.mul(&kl, dl)?)?, &zero)?, dl)?,
            )?,
            &zero,
        )?;
        let upper_area = upper::<R>(c, &c.mul(&upper::<R>(c, &c.mul(&ku, dh)?)?, dh)?)?;
        let (lower_discharge, upper_discharge) =
            self.triangular_signed_product(&lower_area, &upper_area, ul, uh)?;
        Ok(Sample {
            face: Face {
                area,
                discharge,
                depth: depth.clone(),
                velocity: velocity.clone(),
            },
            area_bound,
            discharge_bound,
            lower_area,
            upper_area,
            lower_discharge,
            upper_discharge,
            lower_depth: dl.clone(),
            upper_depth: dh.clone(),
            lower_velocity: ul.clone(),
            upper_velocity: uh.clone(),
        })
    }

    pub(super) fn triangular_original_sample(
        &self,
        face: &Face<R>,
        side: &super::super::wave::SideBounds<R>,
        possible: &Tensor<R>,
    ) -> Result<Sample<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(possible, 0.0)?;
        let mask = |x: &Tensor<R>| c.where_cond(possible, x, &zero);
        let area = mask(&face.area)?;
        let discharge = mask(&face.discharge)?;
        Ok(Sample {
            face: Face {
                area: area.clone(),
                discharge: discharge.clone(),
                depth: mask(&face.depth)?,
                velocity: mask(&face.velocity)?,
            },
            area_bound: zero.clone(),
            discharge_bound: zero.clone(),
            lower_area: area.clone(),
            upper_area: area,
            lower_discharge: discharge.clone(),
            upper_discharge: discharge,
            lower_depth: mask(&side.lower_depth)?,
            upper_depth: mask(&side.upper_depth)?,
            lower_velocity: mask(&side.lower_velocity)?,
            upper_velocity: mask(&side.upper_velocity)?,
        })
    }

    pub(super) fn triangular_derived_candidate(
        &self,
        geometry: &TriangleGeometry<R>,
        state: &super::analytic::DepthVelocity<R>,
        possible: Tensor<R>,
        selected: Tensor<R>,
    ) -> Result<super::accuracy::Candidate<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(&possible, 0.0)?;
        let mask = |x: &Tensor<R>| c.where_cond(&possible, x, &zero);
        let depth = mask(&state.depth)?;
        let velocity = mask(&state.velocity)?;
        let lower_depth = mask(&state.lower_depth)?;
        let upper_depth = mask(&state.upper_depth)?;
        let lower_velocity = mask(&state.lower_velocity)?;
        let upper_velocity = mask(&state.upper_velocity)?;
        let sample = self.triangular_sample(
            geometry,
            &ValueBounds {
                nominal: depth,
                lower: lower_depth,
                upper: upper_depth,
            },
            &ValueBounds {
                nominal: velocity,
                lower: lower_velocity,
                upper: upper_velocity,
            },
        )?;
        Ok(super::accuracy::Candidate {
            possible,
            selected,
            sample,
        })
    }

    pub(super) fn triangular_signed_product(
        &self,
        al: &Tensor<R>,
        ah: &Tensor<R>,
        bl: &Tensor<R>,
        bh: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.client;
        let p = [
            c.mul(al, bl)?,
            c.mul(al, bh)?,
            c.mul(ah, bl)?,
            c.mul(ah, bh)?,
        ];
        let lo = c.minimum(&c.minimum(&p[0], &p[1])?, &c.minimum(&p[2], &p[3])?)?;
        let hi = c.maximum(&c.maximum(&p[0], &p[1])?, &c.maximum(&p[2], &p[3])?)?;
        Ok((lower::<R>(c, &lo)?, upper::<R>(c, &hi)?))
    }
}
