use super::super::{
    ChannelError, ChannelTensorBoundary, admission::Fields, boundary::assembly::numerical_range,
    cfl_speeds::CflSpeeds, context::Context, geometry::TriangleGeometry,
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
        let frame =
            self.direct_boundary_frame(&self.triangular_endpoint_interior(&faces)?, left, right)?;
        let fronts = self.triangular_fronts(geometry, &faces, frame.as_ref())?;
        if frame.is_some() {
            let (start, len) = numerical_range(self.count(), left, right);
            let numerical = if len > 0 {
                let (sl, sr) = self.triangular_bounds(
                    geometry,
                    &narrow_face(&faces.left, start, len)?,
                    &narrow_face(&faces.right, start, len)?,
                )?;
                Some(self.directional_speeds(&sl, &sr)?)
            } else {
                None
            };
            return Ok(CflSpeeds {
                outer: self.assemble_face_speeds(frame.as_ref(), numerical.as_ref())?,
                internal: faces.internal_speed,
                fronts,
            });
        }
        let (sl, sr) = self.triangular_bounds(geometry, &faces.left, &faces.right)?;
        Ok(CflSpeeds {
            outer: self.directional_speeds(&sl, &sr)?,
            internal: faces.internal_speed,
            fronts,
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

#[cfg(test)]
mod tests {
    use super::super::super::geometry::TensorGeometry;
    use super::super::fixture::{TestResult, context, face};

    #[test]
    fn film_signals_remain_at_the_wet_vacuum_front_scale() -> TestResult {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        let wet = face(&ctx, 1.0, 0.0)?;
        for exponent in 1..=150 {
            let film_depth = 10_f64.powi(-exponent);
            let film = face(&ctx, film_depth, 0.0)?;
            let (sl, sr) = ctx.triangular_bounds(geometry, &wet, &film)?;
            // For g=8, the resting sound floor is exactly 2*sqrt(h_right).
            let reference_speed = 8.0 - 6.0 * film_depth.sqrt();
            let sl = sl.item::<f64>()?;
            let sr = sr.item::<f64>()?;
            assert!(sl.is_finite() && sr.is_finite());
            assert!(sl <= -reference_speed && (sl + reference_speed).abs() < 1e-12);
            assert!(sr >= reference_speed && (sr - reference_speed).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn invariant_signals_enclose_compression_vacuum_and_characteristic_cancellation() -> TestResult
    {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        for (ul, ur, reference_lower, reference_upper) in [
            (3.0, -3.0, -5.0, 5.0),
            (-10.0, 10.0, -12.0, 12.0),
            (8.0, 8.0, 6.0, 10.0),
            (2.0, 2.0, 0.0, 4.0),
            (0.0, 0.0, -2.0, 2.0),
            (-3.0, 3.0, -5.0, 5.0),
        ] {
            let (sl, sr) =
                ctx.triangular_bounds(geometry, &face(&ctx, 1.0, ul)?, &face(&ctx, 1.0, ur)?)?;
            assert!(sl.item::<f64>()? <= reference_lower);
            assert!(sr.item::<f64>()? >= reference_upper);
            assert!((sl.item::<f64>()? - reference_lower).abs() < 1e-12);
            assert!((sr.item::<f64>()? - reference_upper).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn uncertain_rarefaction_boundary_retains_the_enclosure_until_certified() -> TestResult {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        let left = face(&ctx, 1.0, 0.0)?;
        // The exact transition has c_left=2, c_right=1, and velocity jump 4.
        for right_velocity in [f64::from_bits(4.0_f64.to_bits() - 1), 4.0] {
            let (sl, sr) =
                ctx.triangular_bounds(geometry, &left, &face(&ctx, 0.25, right_velocity)?)?;
            assert!(sl.item::<f64>()? < -4.9);
            assert!(sl.item::<f64>()? <= -2.0);
            assert!(sr.item::<f64>()? >= right_velocity + 1.0);
        }
        let right_velocity = 4.0 + 1e-10;
        let (sl, sr) =
            ctx.triangular_bounds(geometry, &left, &face(&ctx, 0.25, right_velocity)?)?;
        assert!(sl.item::<f64>()? <= -2.0);
        assert!((sl.item::<f64>()? + 2.0).abs() < 1e-12);
        assert!(sr.item::<f64>()? >= right_velocity + 1.0);
        assert!((sr.item::<f64>()? - (right_velocity + 1.0)).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn vacuum_front_certificate_remains_false_when_zero_is_uncertain() -> TestResult {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        // At g=8 and h=1 the two vacuum fronts cross zero at velocities -8 and +8.
        for (velocity, expected) in [(8.0 - 1e-10, 0.0), (8.0, 0.0), (8.0 + 1e-10, 1.0)] {
            let bounds = ctx.triangular_wave_bounds(
                geometry,
                &face(&ctx, 1.0, -velocity)?,
                &face(&ctx, 1.0, velocity)?,
            )?;
            assert_eq!(bounds.vacuum.item::<f64>()?, expected);
        }
        let one_dry =
            ctx.triangular_wave_bounds(geometry, &face(&ctx, 1.0, -10.0)?, &face(&ctx, 0.0, 0.0)?)?;
        assert_eq!(one_dry.vacuum.item::<f64>()?, 0.0);
        Ok(())
    }

    #[test]
    fn unselected_vacuum_advection_uses_benign_operands_before_product_checks() -> TestResult {
        let ctx = context()?;
        let TensorGeometry::Triangular(geometry) = &ctx.geometry else {
            return Err("triangular geometry required".into());
        };
        let left = face(&ctx, 1.0, -1e200)?;
        let right = face(&ctx, 1.0, 1e200)?;
        let bounds = ctx.triangular_wave_bounds(geometry, &left, &right)?;
        assert!(bounds.lower.item::<f64>()?.is_finite() && bounds.upper.item::<f64>()?.is_finite());
        assert_eq!(bounds.vacuum.item::<f64>()?, 1.0);
        let physical = ctx.triangular_physical(&left, &ctx.tensor(&[0.0])?)?;
        assert_eq!(physical.momentum.item::<f64>()?, 0.0);
        assert_eq!(physical.momentum_bound.item::<f64>()?, 0.0);
        assert!((physical.pressure.item::<f64>()? - 8.0 / 3.0).abs() < 1e-14);
        Ok(())
    }
}
