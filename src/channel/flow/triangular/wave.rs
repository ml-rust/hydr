use super::super::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    boundary::assembly::{is_scalar, numerical_range},
    context::Context,
    geometry::TriangleGeometry,
    reconstruction::CflSpeeds,
};
use super::{
    envelope::{lower, upper},
    faces::{Face, narrow_face},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
    },
    runtime::Runtime,
    tensor::Tensor,
};
use solvr::optimize::roots::BatchedBisectionAlgorithms;

pub(super) struct WaveBounds<R: Runtime<DType = DType>> {
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
    pub vacuum: Tensor<R>,
}

pub(in crate::channel::flow::triangular) struct SideBounds<R: Runtime<DType = DType>> {
    pub lower_depth: Tensor<R>,
    pub upper_depth: Tensor<R>,
    pub lower_signal: Tensor<R>,
    pub upper_signal: Tensor<R>,
    pub lower_sound: Tensor<R>,
    pub upper_sound: Tensor<R>,
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
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub(in crate::channel::flow) fn triangular_speed(
        &self,
        geometry: &TriangleGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<CflSpeeds<R>, ChannelError> {
        let faces = self.triangular_faces(geometry, fields, left, right)?;
        if is_scalar(left) || is_scalar(right) {
            let frame = self.scalar_boundary_frame(
                &self.triangular_endpoint_interior(&faces)?,
                left,
                right,
            )?;
            let (start, len) = numerical_range(self.count(), left, right);
            let numerical = if len > 0 {
                let (sl, sr) = self.triangular_bounds(
                    geometry,
                    &narrow_face(&faces.left, start, len)?,
                    &narrow_face(&faces.right, start, len)?,
                )?;
                Some(
                    self.client
                        .maximum(&self.client.abs(&sl)?, &self.client.abs(&sr)?)?,
                )
            } else {
                None
            };
            return Ok(CflSpeeds {
                outer: self.assemble_boundary_faces(
                    frame.as_ref(),
                    numerical.as_ref(),
                    frame.as_ref().map(|f| &f.traces.signal_speed_bound_m_per_s),
                )?,
                internal: faces.internal_speed,
            });
        }
        let (sl, sr) = self.triangular_bounds(geometry, &faces.left, &faces.right)?;
        Ok(CflSpeeds {
            outer: self
                .client
                .maximum(&self.client.abs(&sl)?, &self.client.abs(&sr)?)?,
            internal: faces.internal_speed,
        })
    }
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
    pub(super) fn triangular_bounds(
        &self,
        geometry: &TriangleGeometry<R>,
        left: &Face<R>,
        right: &Face<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let WaveBounds {
            lower,
            upper,
            vacuum: _vacuum,
        } = self.triangular_wave_bounds(geometry, left, right)?;
        Ok((lower, upper))
    }

    pub(super) fn triangular_wave_bounds(
        &self,
        geometry: &TriangleGeometry<R>,
        left: &Face<R>,
        right: &Face<R>,
    ) -> Result<WaveBounds<R>, ChannelError> {
        let c = &self.client;
        let left_bounds = self.triangular_side_bounds(geometry, left)?;
        let right_bounds = self.triangular_side_bounds(geometry, right)?;
        let zero = self.constant(&left.area, 0.0)?;
        let lower_half_left = lower::<R>(c, &c.mul_scalar(&left_bounds.lower_sound, 0.5)?)?;
        let lower_half_right = lower::<R>(c, &c.mul_scalar(&right_bounds.lower_sound, 0.5)?)?;
        let lower_mean = lower::<R>(c, &c.add(&lower_half_left, &lower_half_right)?)?;
        let upper_jump = upper::<R>(
            c,
            &c.sub(&right_bounds.upper_velocity, &left_bounds.lower_velocity)?,
        )?;
        self.finite(&upper_jump, "triangular upper velocity jump")?;
        let upper_jump_eighth = upper::<R>(c, &c.mul_scalar(&upper_jump, 0.125)?)?;
        let rarefaction_lower = lower::<R>(c, &c.sub(&lower_mean, &upper_jump_eighth)?)?;
        // The minimum of side sounds and nonnegative rarefaction star sound bounds every state below.
        let floor = c.minimum(
            &c.minimum(&left_bounds.lower_sound, &right_bounds.lower_sound)?,
            &c.maximum(&rarefaction_lower, &zero)?,
        )?;
        let three_floor = c.maximum(&lower::<R>(c, &c.mul_scalar(&floor, 3.0)?)?, &zero)?;
        let wet = c.maximum(&c.gt(&left.area, &zero)?, &c.gt(&right.area, &zero)?)?;
        let sl = c.where_cond(
            &wet,
            &lower::<R>(
                c,
                &c.add(
                    &c.minimum(&left_bounds.lower_signal, &right_bounds.lower_signal)?,
                    &three_floor,
                )?,
            )?,
            &zero,
        )?;
        let sr = c.where_cond(
            &wet,
            &upper::<R>(
                c,
                &c.sub(
                    &c.maximum(&left_bounds.upper_signal, &right_bounds.upper_signal)?,
                    &three_floor,
                )?,
            )?,
            &zero,
        )?;
        let lower_jump = lower::<R>(
            c,
            &c.sub(&right_bounds.lower_velocity, &left_bounds.upper_velocity)?,
        )?;
        let upper_left_difference = upper::<R>(
            c,
            &c.sub(&left_bounds.upper_sound, &right_bounds.lower_sound)?,
        )?;
        let upper_right_difference = upper::<R>(
            c,
            &c.sub(&right_bounds.upper_sound, &left_bounds.lower_sound)?,
        )?;
        let upper_difference = c.maximum(
            &zero,
            &c.maximum(&upper_left_difference, &upper_right_difference)?,
        )?;
        let upper_threshold = upper::<R>(c, &c.mul_scalar(&upper_difference, 4.0)?)?;
        let both_wet = c.mul(&c.gt(&left.area, &zero)?, &c.gt(&right.area, &zero)?)?;
        let certified = c.mul(&both_wet, &c.ge(&lower_jump, &upper_threshold)?)?;
        let lower_head = lower::<R>(
            c,
            &c.sub(&left_bounds.lower_velocity, &left_bounds.upper_sound)?,
        )?;
        let upper_head = upper::<R>(
            c,
            &c.add(&right_bounds.upper_velocity, &right_bounds.upper_sound)?,
        )?;
        let sl = c.where_cond(&certified, &lower_head, &sl)?;
        let sr = c.where_cond(&certified, &upper_head, &sr)?;
        self.finite(&sl, "triangular lower signal")?;
        self.finite(&sr, "triangular upper signal")?;
        let left_front_upper = upper::<R>(
            c,
            &c.add(
                &left_bounds.upper_velocity,
                &upper::<R>(c, &c.mul_scalar(&left_bounds.upper_sound, 4.0)?)?,
            )?,
        )?;
        let right_front_lower = lower::<R>(
            c,
            &c.sub(
                &right_bounds.lower_velocity,
                &upper::<R>(c, &c.mul_scalar(&right_bounds.upper_sound, 4.0)?)?,
            )?,
        )?;
        let vacuum = c.mul(
            &both_wet,
            &c.mul(
                &c.le(&left_front_upper, &zero)?,
                &c.ge(&right_front_lower, &zero)?,
            )?,
        )?;
        Ok(WaveBounds {
            lower: sl,
            upper: sr,
            vacuum,
        })
    }

    pub(in crate::channel::flow::triangular) fn triangular_side_bounds(
        &self,
        geometry: &TriangleGeometry<R>,
        face: &Face<R>,
    ) -> Result<SideBounds<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(&face.area, 0.0)?;
        let wet = c.gt(&face.area, &zero)?;
        let root_area = upper::<R>(c, &c.sqrt(&face.area)?)?;
        let depth_upper = upper::<R>(c, &c.div(&root_area, &geometry.root_coefficient_lower)?)?;
        let root_depth_upper = upper::<R>(c, &c.sqrt(&depth_upper)?)?;
        let sound_numerator =
            upper::<R>(c, &c.mul(&geometry.root_gravity_upper, &root_depth_upper)?)?;
        let sound_upper = upper::<R>(c, &c.div(&sound_numerator, &geometry.root_two_lower)?)?;
        let four_sound = upper::<R>(c, &c.mul_scalar(&sound_upper, 4.0)?)?;
        let low_velocity = lower::<R>(c, &face.velocity)?;
        let high_velocity = upper::<R>(c, &face.velocity)?;
        let lo = c.where_cond(
            &wet,
            &lower::<R>(c, &c.sub(&low_velocity, &four_sound)?)?,
            &zero,
        )?;
        let hi = c.where_cond(
            &wet,
            &upper::<R>(c, &c.add(&high_velocity, &four_sound)?)?,
            &zero,
        )?;
        self.finite(&lo, "triangular side lower signal")?;
        self.finite(&hi, "triangular side upper signal")?;
        if !self.all(&c.maximum(&c.eq(&face.area, &zero)?, &c.gt(&four_sound, &zero)?)?)? {
            return Err(ChannelError::numerical(
                "triangular positive area has unresolved sound speed: rescale geometry",
            ));
        }
        let root_area_lower = c.maximum(&lower::<R>(c, &c.sqrt(&face.area)?)?, &zero)?;
        let depth_lower = c.maximum(
            &lower::<R>(
                c,
                &c.div(&root_area_lower, &geometry.root_coefficient_upper)?,
            )?,
            &zero,
        )?;
        let root_depth_lower = c.maximum(&lower::<R>(c, &c.sqrt(&depth_lower)?)?, &zero)?;
        let numerator_lower = c.maximum(
            &lower::<R>(c, &c.mul(&geometry.root_gravity_lower, &root_depth_lower)?)?,
            &zero,
        )?;
        let lower_sound = c.where_cond(
            &wet,
            &c.maximum(
                &lower::<R>(c, &c.div(&numerator_lower, &geometry.root_two_upper)?)?,
                &zero,
            )?,
            &zero,
        )?;
        self.finite(&lower_sound, "triangular lower sound speed")?;
        Ok(SideBounds {
            lower_depth: c.where_cond(&wet, &depth_lower, &zero)?,
            upper_depth: c.where_cond(&wet, &depth_upper, &zero)?,
            lower_signal: lo,
            upper_signal: hi,
            lower_sound,
            upper_sound: c.where_cond(&wet, &sound_upper, &zero)?,
            lower_velocity: c.where_cond(&wet, &low_velocity, &zero)?,
            upper_velocity: c.where_cond(&wet, &high_velocity, &zero)?,
        })
    }
}
