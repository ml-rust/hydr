use super::super::{
    envelope::{lower, upper},
    faces::Face,
    wave::SideBounds,
};
use super::types::RiemannInput;
use super::{
    accuracy::{Candidate, SampleFlux},
    analytic::DepthVelocity,
};
use crate::channel::flow::{ChannelError, context::Context, geometry::TriangleGeometry};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps, ShapeOps,
        UnaryOps,
    },
    runtime::Runtime,
    tensor::Tensor,
};
use solvr::optimize::roots::BatchedBisectionAlgorithms;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub(in crate::channel::flow::triangular) fn triangular_godunov(
        &self,
        geometry: &TriangleGeometry<R>,
        input: RiemannInput<'_, R>,
    ) -> Result<SampleFlux<R>, ChannelError> {
        let RiemannInput {
            left,
            right,
            left_bounds: ls,
            right_bounds: rs,
            lower_signal: sl,
            upper_signal: sr,
        } = input;
        let c = &self.client;
        let zero = self.constant(&left.area, 0.0)?;
        let one = self.constant(&left.area, 1.0)?;
        let and = |a: &Tensor<R>, b: &Tensor<R>| c.mul(a, b);
        let or = |a: &Tensor<R>, b: &Tensor<R>| c.maximum(a, b);
        let not = |a: &Tensor<R>| c.eq(a, &zero);
        let wl = c.gt(&left.area, &zero)?;
        let wr = c.gt(&right.area, &zero)?;
        let both = and(&wl, &wr)?;
        let identical = and(
            &both,
            &and(
                &c.eq(&left.area, &right.area)?,
                &c.eq(&left.discharge, &right.discharge)?,
            )?,
        )?;
        let fast_left = or(&identical, &and(&wl, &c.ge(sl, &zero)?)?)?;
        let fast_right = and(&wr, &and(&not(&fast_left)?, &c.le(sr, &zero)?)?)?;
        let unresolved = not(&or(&fast_left, &fast_right)?)?;
        let cl = self.triangular_sound(geometry, &left.depth)?;
        let cr = self.triangular_sound(geometry, &right.depth)?;
        let lh = c.sub(&left.velocity, &cl)?;
        let rh = c.add(&right.velocity, &cr)?;
        let lf = c.add(&left.velocity, &c.mul_scalar(&cl, 4.0)?)?;
        let rf = c.sub(&right.velocity, &c.mul_scalar(&cr, 4.0)?)?;
        let lhl = lower::<R>(c, &c.sub(&ls.lower_velocity, &ls.upper_sound)?)?;
        let lhh = upper::<R>(c, &c.sub(&ls.upper_velocity, &ls.lower_sound)?)?;
        let rhl = lower::<R>(c, &c.add(&rs.lower_velocity, &rs.lower_sound)?)?;
        let rhh = upper::<R>(c, &c.add(&rs.upper_velocity, &rs.upper_sound)?)?;
        let lfl = lower::<R>(
            c,
            &c.add(
                &ls.lower_velocity,
                &lower::<R>(c, &c.mul_scalar(&ls.lower_sound, 4.0)?)?,
            )?,
        )?;
        let lfh = upper::<R>(
            c,
            &c.add(
                &ls.upper_velocity,
                &upper::<R>(c, &c.mul_scalar(&ls.upper_sound, 4.0)?)?,
            )?,
        )?;
        let rfl = lower::<R>(
            c,
            &c.sub(
                &rs.lower_velocity,
                &upper::<R>(c, &c.mul_scalar(&rs.upper_sound, 4.0)?)?,
            )?,
        )?;
        let rfh = upper::<R>(
            c,
            &c.sub(
                &rs.upper_velocity,
                &lower::<R>(c, &c.mul_scalar(&rs.lower_sound, 4.0)?)?,
            )?,
        )?;
        let lf_possible = and(
            &unresolved,
            &and(&wl, &and(&c.le(&lhl, &zero)?, &c.ge(&lfh, &zero)?)?)?,
        )?;
        let rf_possible = and(
            &unresolved,
            &and(&wr, &and(&c.ge(&rhh, &zero)?, &c.le(&rfl, &zero)?)?)?,
        )?;
        let (left_fan_face, left_fan_side) =
            self.triangular_mask_reference(left, ls, &lf_possible)?;
        let (right_fan_face, right_fan_side) =
            self.triangular_mask_reference(right, rs, &rf_possible)?;
        let left_fan =
            self.triangular_sonic_state(geometry, &left_fan_face, &left_fan_side, false)?;
        let right_fan =
            self.triangular_sonic_state(geometry, &right_fan_face, &right_fan_side, true)?;
        let left_probe_active = and(&both, &lf_possible)?;
        let right_probe_active = and(&both, &rf_possible)?;
        let (left_probe_left, left_probe_ls) =
            self.triangular_mask_reference(left, ls, &left_probe_active)?;
        let (left_probe_right, left_probe_rs) =
            self.triangular_mask_reference(right, rs, &left_probe_active)?;
        let (right_probe_left, right_probe_ls) =
            self.triangular_mask_reference(left, ls, &right_probe_active)?;
        let (right_probe_right, right_probe_rs) =
            self.triangular_mask_reference(right, rs, &right_probe_active)?;
        let left_probe = c.where_cond(&left_probe_active, &left_fan.lower_depth, &one)?;
        let right_probe = c.where_cond(&right_probe_active, &right_fan.lower_depth, &one)?;
        let left_residual = self.triangular_residual_bounds(
            geometry,
            &left_probe_left,
            &left_probe_right,
            &left_probe_ls,
            &left_probe_rs,
            &left_probe,
        )?;
        let right_residual = self.triangular_residual_bounds(
            geometry,
            &right_probe_left,
            &right_probe_right,
            &right_probe_ls,
            &right_probe_rs,
            &right_probe,
        )?;
        let certified_left_fan = and(
            &both,
            &and(
                &lf_possible,
                &and(
                    &c.lt(&lhh, &zero)?,
                    &and(&c.gt(&lfl, &zero)?, &c.ge(&left_residual.lower, &zero)?)?,
                )?,
            )?,
        )?;
        let certified_right_fan = and(
            &both,
            &and(
                &rf_possible,
                &and(
                    &c.gt(&rhl, &zero)?,
                    &and(&c.lt(&rfh, &zero)?, &c.ge(&right_residual.lower, &zero)?)?,
                )?,
            )?,
        )?;
        let jump_lower = lower::<R>(c, &c.sub(&rs.lower_velocity, &ls.upper_velocity)?)?;
        let delta = c.maximum(
            &zero,
            &c.maximum(
                &upper::<R>(c, &c.sub(&ls.upper_sound, &rs.lower_sound)?)?,
                &upper::<R>(c, &c.sub(&rs.upper_sound, &ls.lower_sound)?)?,
            )?,
        )?;
        let analytic_mask = and(
            &both,
            &and(
                &unresolved,
                &c.ge(&jump_lower, &upper::<R>(c, &c.mul_scalar(&delta, 4.0)?)?)?,
            )?,
        )?;
        let bypass = or(
            &or(&certified_left_fan, &certified_right_fan)?,
            &analytic_mask,
        )?;
        let root_mask = and(&both, &and(&unresolved, &not(&bypass)?)?)?;
        let root = self.triangular_star_bracket(geometry, left, right, ls, rs, &root_mask)?;
        let (analytic_left, analytic_ls) =
            self.triangular_mask_reference(left, ls, &analytic_mask)?;
        let (analytic_right, analytic_rs) =
            self.triangular_mask_reference(right, rs, &analytic_mask)?;
        let analytic = self.triangular_two_rare_state(
            geometry,
            &analytic_left,
            &analytic_right,
            &analytic_ls,
            &analytic_rs,
        )?;
        let choose = |a: &Tensor<R>, b: &Tensor<R>| c.where_cond(&analytic_mask, a, b);
        let symmetric_star = and(
            &both,
            &and(
                &c.eq(&left.area, &right.area)?,
                &c.eq(&left.discharge, &c.neg(&right.discharge)?)?,
            )?,
        )?;
        // Equal areas and opposite conserved discharges give an exactly stationary star state.
        let star = DepthVelocity {
            depth: choose(&analytic.depth, &root.depth)?,
            lower_depth: choose(&analytic.lower_depth, &root.depth_lower)?,
            upper_depth: choose(&analytic.upper_depth, &root.depth_upper)?,
            velocity: choose(&analytic.velocity, &root.velocity)?,
            lower_velocity: c.where_cond(
                &symmetric_star,
                &zero,
                &choose(&analytic.lower_velocity, &root.velocity_lower)?,
            )?,
            upper_velocity: c.where_cond(
                &symmetric_star,
                &zero,
                &choose(&analytic.upper_velocity, &root.velocity_upper)?,
            )?,
            sound: choose(
                &analytic.sound,
                &self.triangular_sound(geometry, &root.depth)?,
            )?,
            lower_sound: choose(&analytic.lower_sound, &root.sound_lower)?,
            upper_sound: choose(&analytic.upper_sound, &root.sound_upper)?,
        };
        let vacuum_possible = or(
            &and(&unresolved, &not(&both)?)?,
            &and(&analytic_mask, &c.eq(&analytic.lower_sound, &zero)?)?,
        )?;
        let vacuum_selected = or(
            &and(&unresolved, &not(&both)?)?,
            &and(&analytic_mask, &c.eq(&analytic.sound, &zero)?)?,
        )?;
        let star_possible = or(
            &root_mask,
            &and(&analytic_mask, &c.gt(&analytic.upper_sound, &zero)?)?,
        )?;
        let star_selected = and(&or(&root_mask, &analytic_mask)?, &not(&vacuum_selected)?)?;
        let left_half_possible = and(&star_possible, &c.ge(&star.upper_velocity, &zero)?)?;
        let right_half_possible = and(&star_possible, &c.le(&star.lower_velocity, &zero)?)?;
        let left_half = and(&star_selected, &c.ge(&star.velocity, &zero)?)?;
        let right_half = and(&star_selected, &c.lt(&star.velocity, &zero)?)?;
        let lr_possible = c.le(&star.lower_depth, &ls.upper_depth)?;
        let rr_possible = c.le(&star.lower_depth, &rs.upper_depth)?;
        let lshock_possible = and(&root_mask, &c.ge(&star.upper_depth, &ls.lower_depth)?)?;
        let rshock_possible = and(&root_mask, &c.ge(&star.upper_depth, &rs.lower_depth)?)?;
        let lshock = and(&root_mask, &c.gt(&star.depth, &left.depth)?)?;
        let rshock = and(&root_mask, &c.gt(&star.depth, &right.depth)?)?;
        let ltail = c.sub(&star.velocity, &star.sound)?;
        let rtail = c.add(&star.velocity, &star.sound)?;
        let ltl = lower::<R>(c, &c.sub(&star.lower_velocity, &star.upper_sound)?)?;
        let lth = upper::<R>(c, &c.sub(&star.upper_velocity, &star.lower_sound)?)?;
        let rtl = lower::<R>(c, &c.add(&star.lower_velocity, &star.lower_sound)?)?;
        let rth = upper::<R>(c, &c.add(&star.upper_velocity, &star.upper_sound)?)?;
        let left_original_possible = or(
            &fast_left,
            &or(
                &and(&vacuum_possible, &and(&wl, &c.ge(&lhh, &zero)?)?)?,
                &and(
                    &left_half_possible,
                    &or(
                        &and(&lr_possible, &c.ge(&lhh, &zero)?)?,
                        &and(&lshock_possible, &c.ge(&root.left_shock_upper, &zero)?)?,
                    )?,
                )?,
            )?,
        )?;
        let right_original_possible = or(
            &fast_right,
            &or(
                &and(&vacuum_possible, &and(&wr, &c.le(&rhl, &zero)?)?)?,
                &and(
                    &right_half_possible,
                    &or(
                        &and(&rr_possible, &c.le(&rhl, &zero)?)?,
                        &and(&rshock_possible, &c.le(&root.right_shock_lower, &zero)?)?,
                    )?,
                )?,
            )?,
        )?;
        let left_original = or(
            &fast_left,
            &or(
                &and(&vacuum_selected, &and(&wl, &c.ge(&lh, &zero)?)?)?,
                &and(
                    &left_half,
                    &c.where_cond(&lshock, &c.ge(&root.left_shock, &zero)?, &c.ge(&lh, &zero)?)?,
                )?,
            )?,
        )?;
        let right_original = or(
            &fast_right,
            &or(
                &and(&vacuum_selected, &and(&wr, &c.le(&rh, &zero)?)?)?,
                &and(
                    &right_half,
                    &c.where_cond(
                        &rshock,
                        &c.le(&root.right_shock, &zero)?,
                        &c.le(&rh, &zero)?,
                    )?,
                )?,
            )?,
        )?;
        let left_fan_possible = or(
            &certified_left_fan,
            &and(
                &lf_possible,
                &or(
                    &vacuum_possible,
                    &and(
                        &left_half_possible,
                        &and(&lr_possible, &c.ge(&lth, &zero)?)?,
                    )?,
                )?,
            )?,
        )?;
        let right_fan_possible = or(
            &certified_right_fan,
            &and(
                &rf_possible,
                &or(
                    &vacuum_possible,
                    &and(
                        &right_half_possible,
                        &and(&rr_possible, &c.le(&rtl, &zero)?)?,
                    )?,
                )?,
            )?,
        )?;
        let left_fan_selected = or(
            &certified_left_fan,
            &or(
                &and(
                    &vacuum_selected,
                    &and(&wl, &and(&c.lt(&lh, &zero)?, &c.gt(&lf, &zero)?)?)?,
                )?,
                &and(
                    &left_half,
                    &and(
                        &not(&lshock)?,
                        &and(&c.lt(&lh, &zero)?, &c.gt(&ltail, &zero)?)?,
                    )?,
                )?,
            )?,
        )?;
        let right_fan_selected = or(
            &certified_right_fan,
            &or(
                &and(
                    &vacuum_selected,
                    &and(&wr, &and(&c.gt(&rh, &zero)?, &c.lt(&rf, &zero)?)?)?,
                )?,
                &and(
                    &right_half,
                    &and(
                        &not(&rshock)?,
                        &and(&c.gt(&rh, &zero)?, &c.lt(&rtail, &zero)?)?,
                    )?,
                )?,
            )?,
        )?;
        let star_region_possible = or(
            &and(
                &left_half_possible,
                &or(
                    &and(&lr_possible, &c.le(&ltl, &zero)?)?,
                    &and(&lshock_possible, &c.le(&root.left_shock_lower, &zero)?)?,
                )?,
            )?,
            &and(
                &right_half_possible,
                &or(
                    &and(&rr_possible, &c.ge(&rth, &zero)?)?,
                    &and(&rshock_possible, &c.ge(&root.right_shock_upper, &zero)?)?,
                )?,
            )?,
        )?;
        let star_region_selected = and(
            &star_selected,
            &not(&or(
                &or(&left_original, &right_original)?,
                &or(&left_fan_selected, &right_fan_selected)?,
            )?)?,
        )?;
        let zero_possible = and(
            &vacuum_possible,
            &and(
                &or(&not(&wl)?, &c.le(&lfl, &zero)?)?,
                &or(&not(&wr)?, &c.ge(&rfh, &zero)?)?,
            )?,
        )?;
        let zero_selected = and(
            &vacuum_selected,
            &not(&or(
                &or(&left_original, &right_original)?,
                &or(&left_fan_selected, &right_fan_selected)?,
            )?)?,
        )?;
        let zero_state = DepthVelocity {
            depth: zero.clone(),
            lower_depth: zero.clone(),
            upper_depth: zero.clone(),
            velocity: zero.clone(),
            lower_velocity: zero.clone(),
            upper_velocity: zero.clone(),
            sound: zero.clone(),
            lower_sound: zero.clone(),
            upper_sound: zero,
        };
        let candidates = vec![
            Candidate {
                sample: self.triangular_original_sample(left, ls, &left_original_possible)?,
                possible: left_original_possible,
                selected: left_original,
            },
            Candidate {
                sample: self.triangular_original_sample(right, rs, &right_original_possible)?,
                possible: right_original_possible,
                selected: right_original,
            },
            self.triangular_derived_candidate(
                geometry,
                &left_fan,
                left_fan_possible,
                left_fan_selected,
            )?,
            self.triangular_derived_candidate(
                geometry,
                &right_fan,
                right_fan_possible,
                right_fan_selected,
            )?,
            self.triangular_derived_candidate(
                geometry,
                &star,
                star_region_possible,
                star_region_selected,
            )?,
            self.triangular_derived_candidate(geometry, &zero_state, zero_possible, zero_selected)?,
        ];
        self.triangular_sample_flux(candidates, root.iterations)
    }

    fn triangular_mask_reference(
        &self,
        face: &Face<R>,
        side: &SideBounds<R>,
        active: &Tensor<R>,
    ) -> Result<(Face<R>, SideBounds<R>), ChannelError> {
        let c = &self.client;
        let zero = self.constant(active, 0.0)?;
        let one = self.constant(active, 1.0)?;
        let nonnegative = |x: &Tensor<R>| c.where_cond(active, x, &zero);
        let positive = |x: &Tensor<R>| c.where_cond(active, x, &one);
        Ok((
            Face {
                area: positive(&face.area)?,
                discharge: nonnegative(&face.discharge)?,
                depth: positive(&face.depth)?,
                velocity: nonnegative(&face.velocity)?,
            },
            SideBounds {
                lower_depth: positive(&side.lower_depth)?,
                upper_depth: positive(&side.upper_depth)?,
                lower_signal: nonnegative(&side.lower_signal)?,
                upper_signal: nonnegative(&side.upper_signal)?,
                lower_sound: positive(&side.lower_sound)?,
                upper_sound: positive(&side.upper_sound)?,
                lower_velocity: nonnegative(&side.lower_velocity)?,
                upper_velocity: nonnegative(&side.upper_velocity)?,
            },
        ))
    }
}
