use super::super::{
    envelope::{lower, upper},
    faces::Face,
    wave::SideBounds,
};
use super::types::StarBracket;
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
use solvr::optimize::{
    OptimizeError,
    roots::{BatchedBisectionAlgorithms, BatchedBisectionOptions},
};
use std::cell::RefCell;

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
    pub(in crate::channel::flow::triangular) fn triangular_star_bracket(
        &self,
        geometry: &TriangleGeometry<R>,
        left: &Face<R>,
        right: &Face<R>,
        left_side: &SideBounds<R>,
        right_side: &SideBounds<R>,
        active: &Tensor<R>,
    ) -> Result<StarBracket<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(active, 0.0)?;
        let one = self.constant(active, 1.0)?;
        if self.all(&c.eq(active, &zero)?)? {
            return Ok(StarBracket {
                depth: one.clone(),
                depth_lower: one.clone(),
                depth_upper: one.clone(),
                velocity: zero.clone(),
                velocity_lower: zero.clone(),
                velocity_upper: zero.clone(),
                sound_lower: one.clone(),
                sound_upper: one.clone(),
                left_shock: c.neg(&one)?,
                right_shock: one.clone(),
                left_shock_lower: c.neg(&one)?,
                left_shock_upper: c.neg(&one)?,
                right_shock_lower: one.clone(),
                right_shock_upper: one,
                iterations: 0,
            });
        }
        let mask_face = |f: &Face<R>| -> Result<Face<R>, ChannelError> {
            Ok(Face {
                area: c.where_cond(active, &f.area, &one)?,
                discharge: c.where_cond(active, &f.discharge, &zero)?,
                depth: c.where_cond(active, &f.depth, &one)?,
                velocity: c.where_cond(active, &f.velocity, &zero)?,
            })
        };
        let mask_side = |s: &SideBounds<R>| -> Result<SideBounds<R>, ChannelError> {
            Ok(SideBounds {
                lower_signal: c.where_cond(active, &s.lower_signal, &zero)?,
                upper_signal: c.where_cond(active, &s.upper_signal, &zero)?,
                lower_sound: c.where_cond(active, &s.lower_sound, &one)?,
                upper_sound: c.where_cond(active, &s.upper_sound, &one)?,
                lower_depth: c.where_cond(active, &s.lower_depth, &one)?,
                upper_depth: c.where_cond(active, &s.upper_depth, &one)?,
                lower_velocity: c.where_cond(active, &s.lower_velocity, &zero)?,
                upper_velocity: c.where_cond(active, &s.upper_velocity, &zero)?,
            })
        };
        let left = mask_face(left)?;
        let right = mask_face(right)?;
        let ls = mask_side(left_side)?;
        let rs = mask_side(right_side)?;
        let residual =
            |h: &Tensor<R>| self.triangular_residual_bounds(geometry, &left, &right, &ls, &rs, h);
        let initial_lower = c.where_cond(
            active,
            &c.mul_scalar(&c.minimum(&ls.lower_depth, &rs.lower_depth)?, 0.5)?,
            &one,
        )?;
        let compression = c.maximum(
            &upper::<R>(c, &c.sub(&ls.upper_velocity, &rs.lower_velocity)?)?,
            &zero,
        )?;
        let upper_sound = upper::<R>(
            c,
            &c.add(
                &c.maximum(&ls.upper_sound, &rs.upper_sound)?,
                &upper::<R>(c, &c.mul_scalar(&compression, 0.125)?)?,
            )?,
        )?;
        let upper_root = upper::<R>(
            c,
            &c.div(
                &upper::<R>(c, &c.mul(&upper_sound, &geometry.root_two_upper)?)?,
                &geometry.root_gravity_lower,
            )?,
        )?;
        let initial_upper = c.where_cond(
            active,
            &upper::<R>(c, &c.mul(&upper_root, &upper_root)?)?,
            &one,
        )?;
        self.finite(&initial_upper, "triangular initial star upper depth")?;
        self.star_require(
            active,
            &c.gt(&initial_lower, &zero)?,
            "positive lower depth is unresolved",
        )?;
        self.star_require(
            active,
            &c.le(&residual(&initial_lower)?.upper, &zero)?,
            "lower depth does not certify a negative residual",
        )?;
        self.star_require(
            active,
            &c.ge(&residual(&initial_upper)?.lower, &zero)?,
            "upper depth does not certify a positive residual",
        )?;
        let log_lower = c.cr_log(&initial_lower)?;
        let log_upper = c.cr_log(&initial_upper)?;
        let center = c.add(
            &c.mul_scalar(&log_lower, 0.5)?,
            &c.mul_scalar(&log_upper, 0.5)?,
        )?;
        let xlo = c.sub(&log_lower, &center)?;
        let xhi = c.sub(&log_upper, &center)?;
        let coordinate_scale = c
            .max(&c.maximum(&c.abs(&xlo)?, &c.abs(&xhi)?)?, &[], false)?
            .item::<f64>()?;
        let config = &self.definition.config;
        // Only dimensionless relative controls set the initial log-depth width target.
        let target = [config.riemann_flux_relative, config.root_relative_depth]
            .into_iter()
            .filter(|v| *v > 0.0)
            .fold(1e-14_f64, f64::min)
            * 0.001;
        let tolerance = target.max(4.0 * f64::EPSILON * coordinate_scale.max(1.0));
        let diagnostic = RefCell::new(None);
        let root = c.bisect_batched(
            |x| {
                let result = (|| -> Result<Tensor<R>, ChannelError> {
                    let h = c.cr_exp(&c.add(&center, x)?)?;
                    Ok(c.where_cond(
                        active,
                        &self.triangular_residual_nominal(geometry, &left, &right, &h)?,
                        &zero,
                    )?)
                })();
                match result {
                    Ok(value) => Ok(value),
                    Err(error) => {
                        let mut slot = diagnostic.try_borrow_mut().map_err(|_| {
                            numr::error::Error::InvalidArgument {
                                arg: "triangular star diagnostic",
                                reason: "callback diagnostic channel is already borrowed".into(),
                            }
                        })?;
                        if slot.is_none() {
                            *slot = Some(error);
                        }
                        Err(numr::error::Error::InvalidArgument {
                            arg: "triangular star callback",
                            reason: "physical curve evaluation rejected the candidate".into(),
                        })
                    }
                }
            },
            &xlo,
            &xhi,
            &BatchedBisectionOptions {
                atol: tolerance,
                rtol: 0.0,
                max_iter: config.root_iterations,
                max_expansions: 0,
                expansion_factor: 2.0,
            },
        );
        if let Some(error) = diagnostic.into_inner() {
            return Err(error);
        }
        let root = root.map_err(|error| {
            let context = format!("triangular Solvr star root: {error}. Increase root_iterations or rescale the states");
            match error { OptimizeError::NumrError(_) => ChannelError::Runtime { cell: None, context },
                _ => ChannelError::numerical(context) }
        })?;
        let a = c.cr_exp(&c.add(&center, &root.lower)?)?;
        let b = c.cr_exp(&c.add(&center, &root.upper)?)?;
        let ra = residual(&a)?;
        let rb = residual(&b)?;
        let signed = c.mul(&c.le(&ra.upper, &zero)?, &c.ge(&rb.lower, &zero)?)?;
        let correction = c.mul(active, &c.eq(&signed, &zero)?)?;
        let error = c.maximum(
            &c.maximum(&c.abs(&ra.lower)?, &c.abs(&ra.upper)?)?,
            &c.maximum(&c.abs(&rb.lower)?, &c.abs(&rb.upper)?)?,
        )?;
        // Phi'(H) >= 4 sqrt(g/2)/sqrt(H). One expansion remains within four times b.
        let derivative = lower::<R>(
            c,
            &c.div(
                &lower::<R>(
                    c,
                    &c.mul_scalar(
                        &lower::<R>(
                            c,
                            &c.div(&geometry.root_gravity_lower, &geometry.root_two_upper)?,
                        )?,
                        4.0,
                    )?,
                )?,
                &upper::<R>(c, &c.sqrt(&b)?)?,
            )?,
        )?;
        self.star_require(
            &correction,
            &c.gt(&derivative, &zero)?,
            "residual derivative lower bound is unresolved",
        )?;
        let delta = c.where_cond(
            &correction,
            &upper::<R>(
                c,
                &c.mul_scalar(&upper::<R>(c, &c.div(&error, &derivative)?)?, 2.0)?,
            )?,
            &zero,
        )?;
        self.star_require(
            &correction,
            &c.le(&c.div(&delta, &b)?, &self.constant(&b, 3.0)?)?,
            "one residual correction exceeds its derivative range",
        )?;
        let depth_lower = c.maximum(&lower::<R>(c, &c.sub(&a, &delta)?)?, &zero)?;
        let depth_upper = upper::<R>(c, &c.add(&b, &delta)?)?;
        self.finite(&depth_upper, "triangular star upper depth")?;
        self.star_require(
            active,
            &c.gt(&depth_lower, &zero)?,
            "corrected lower depth is unresolved",
        )?;
        self.star_require(
            active,
            &c.le(&residual(&depth_lower)?.upper, &zero)?,
            "corrected lower residual remains unsigned",
        )?;
        self.star_require(
            active,
            &c.ge(&residual(&depth_upper)?.lower, &zero)?,
            "corrected upper residual remains unsigned",
        )?;
        let depth = c.add(
            &depth_lower,
            &c.mul_scalar(&c.sub(&depth_upper, &depth_lower)?, 0.5)?,
        )?;
        let depth_tolerance = c.add_scalar(
            &c.mul_scalar(&depth_upper, config.root_relative_depth)?,
            config.root_absolute_depth,
        )?;
        self.star_require(
            active,
            &c.le(&c.sub(&depth_upper, &depth_lower)?, &depth_tolerance)?,
            "physical depth bracket exceeds root controls",
        )?;
        let ll = self.triangular_curve_bounds(geometry, &left, &ls, &depth_lower)?;
        let lh = self.triangular_curve_bounds(geometry, &left, &ls, &depth_upper)?;
        let rl = self.triangular_curve_bounds(geometry, &right, &rs, &depth_lower)?;
        let rh = self.triangular_curve_bounds(geometry, &right, &rs, &depth_upper)?;
        let velocity_lower = c.maximum(
            &lower::<R>(c, &c.sub(&ls.lower_velocity, &lh.upper)?)?,
            &lower::<R>(c, &c.add(&rs.lower_velocity, &rl.lower)?)?,
        )?;
        let velocity_upper = c.minimum(
            &upper::<R>(c, &c.sub(&ls.upper_velocity, &ll.lower)?)?,
            &upper::<R>(c, &c.add(&rs.upper_velocity, &rh.upper)?)?,
        )?;
        self.star_require(
            active,
            &c.le(&velocity_lower, &velocity_upper)?,
            "star velocity enclosures do not intersect",
        )?;
        let lc = self
            .triangular_curve_bounds(geometry, &left, &ls, &depth)?
            .nominal;
        let rc = self
            .triangular_curve_bounds(geometry, &right, &rs, &depth)?
            .nominal;
        let use_left = c.le(
            &c.add(&c.abs(&left.velocity)?, &c.abs(&lc)?)?,
            &c.add(&c.abs(&right.velocity)?, &c.abs(&rc)?)?,
        )?;
        let candidate_velocity = c.where_cond(
            &use_left,
            &c.sub(&left.velocity, &lc)?,
            &c.add(&right.velocity, &rc)?,
        )?;
        let symmetric = c.mul(
            &c.eq(&left.area, &right.area)?,
            &c.eq(&left.discharge, &c.neg(&right.discharge)?)?,
        )?;
        let velocity = c.where_cond(
            &symmetric,
            &zero,
            &c.minimum(
                &velocity_upper,
                &c.maximum(&velocity_lower, &candidate_velocity)?,
            )?,
        )?;
        let sound_lower = c.maximum(
            &lower::<R>(
                c,
                &c.div(
                    &lower::<R>(
                        c,
                        &c.mul(
                            &geometry.root_gravity_lower,
                            &c.maximum(&lower::<R>(c, &c.sqrt(&depth_lower)?)?, &zero)?,
                        )?,
                    )?,
                    &geometry.root_two_upper,
                )?,
            )?,
            &zero,
        )?;
        let sound_upper = upper::<R>(
            c,
            &c.div(
                &upper::<R>(
                    c,
                    &c.mul(
                        &geometry.root_gravity_upper,
                        &upper::<R>(c, &c.sqrt(&depth_upper)?)?,
                    )?,
                )?,
                &geometry.root_two_lower,
            )?,
        )?;
        let (left_shock_lower, left_shock_upper) =
            self.star_shock_bounds(geometry, &ls, &depth_lower, &depth_upper, false)?;
        let (right_shock_lower, right_shock_upper) =
            self.star_shock_bounds(geometry, &rs, &depth_lower, &depth_upper, true)?;
        let shock = |face: &Face<R>| -> Result<Tensor<R>, ChannelError> {
            let h = c.maximum(&depth, &face.depth)?;
            let r = c.div(&face.depth, &h)?;
            let polynomial = c.add_scalar(&c.add(&r, &c.mul(&r, &r)?)?, 1.0)?;
            let shape = c.sqrt(&c.div(&polynomial, &c.add_scalar(&r, 1.0)?)?)?;
            let root_three = c.sqrt(&self.constant(&h, 3.0)?)?;
            let scale = c.div(&c.mul(&geometry.root_gravity, &c.sqrt(&h)?)?, &root_three)?;
            Ok(c.mul(&c.mul(&scale, &c.div(&h, &face.depth)?)?, &shape)?)
        };
        let left_shock = c.sub(&left.velocity, &shock(&left)?)?;
        let right_shock = c.add(&right.velocity, &shock(&right)?)?;
        self.finite(&left_shock, "triangular native left shock speed")?;
        self.finite(&right_shock, "triangular native right shock speed")?;
        Ok(StarBracket {
            depth,
            depth_lower,
            depth_upper,
            velocity,
            velocity_lower,
            velocity_upper,
            sound_lower,
            sound_upper,
            left_shock,
            right_shock,
            left_shock_lower,
            left_shock_upper,
            right_shock_lower,
            right_shock_upper,
            iterations: root.iterations,
        })
    }

    fn star_require(
        &self,
        active: &Tensor<R>,
        condition: &Tensor<R>,
        context: &str,
    ) -> Result<(), ChannelError> {
        let c = &self.client;
        let zero = self.constant(active, 0.0)?;
        if !self.all(&c.maximum(&c.eq(active, &zero)?, condition)?)? {
            return Err(ChannelError::numerical(format!(
                "triangular star {context}: rescale states or adjust root controls"
            )));
        }
        Ok(())
    }
}
