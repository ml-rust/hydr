use super::error::arithmetic_error;
use super::source_diagnostics::FrictionGeometryState;
use crate::channel::flow::{ChannelError, context::Context, geometry::TensorGeometry};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
    TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;

pub(in crate::channel::flow) struct FrictionTrial<R: Runtime<DType = DType>> {
    pub momentum: Tensor<R>,
    pub geometry: Option<FrictionGeometryState<R>>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>,
{
    pub fn friction_trial(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        dt: f64,
    ) -> Result<FrictionTrial<R>, ChannelError> {
        if matches!(&self.geometry, TensorGeometry::Surveyed(_)) {
            return self.surveyed_friction(volume, momentum, dt);
        }
        Ok(FrictionTrial {
            momentum: self.friction(volume, momentum, dt)?,
            geometry: None,
        })
    }

    pub(in crate::channel::flow) fn surveyed_friction(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        dt: f64,
    ) -> Result<FrictionTrial<R>, ChannelError> {
        let TensorGeometry::Surveyed(geometry) = &self.geometry else {
            return Err(ChannelError::input(
                "surveyed Manning requires surveyed geometry: select the matching source method",
            ));
        };
        let c = &self.client;
        let zero = self.constant(momentum, 0.0)?;
        let identity = || FrictionTrial {
            momentum: momentum.clone(),
            geometry: Some(FrictionGeometryState {
                momentum_abs_bound: zero.clone(),
            }),
        };
        if dt == 0.0 {
            return Ok(identity());
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(ChannelError::input(format!(
                "surveyed Manning dt={dt}: supply a finite nonnegative duration"
            )));
        }
        let active = c.mul(
            &c.mul(&c.gt(&self.manning, &zero)?, &c.gt(volume, &zero)?)?,
            &c.gt(&c.abs(momentum)?, &zero)?,
        )?;
        let inactive = c.eq(&active, &zero)?;
        if self.all(&inactive)? {
            return Ok(identity());
        }
        let one = self.constant(volume, 1.0)?;
        let area = c.div(&c.where_cond(&active, volume, &zero)?, &self.length)?;
        self.finite(&area, "surveyed Manning original area")?;
        let recovery = geometry.recover_perimeter(&area)?;
        let enclosure = recovery.hydraulic_radius_m();
        let lower = c.where_cond(&active, enclosure.lower(), &one)?;
        let upper = c.where_cond(&active, enclosure.upper(), &one)?;
        let radius = c.midpoint(&lower, &upper)?;
        let magnitude = c.abs(momentum)?;
        let bound =
            self.surveyed_radius_selection_bound(&lower, &upper, &radius, &magnitude, &active)?;
        let next = self.surveyed_manning_update(volume, momentum, &radius, &active, dt)?;
        Ok(FrictionTrial {
            momentum: next,
            geometry: Some(FrictionGeometryState {
                momentum_abs_bound: bound,
            }),
        })
    }

    pub(in crate::channel::flow) fn surveyed_radius_selection_bound(
        &self,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
        selected: &Tensor<R>,
        magnitude: &Tensor<R>,
        active: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let TensorGeometry::Surveyed(geometry) = &self.geometry else {
            return Err(ChannelError::input(
                "surveyed radius selection requires surveyed geometry: select the matching source method",
            ));
        };
        let c = &self.client;
        let zero = self.constant(magnitude, 0.0)?;
        let one = self.constant(magnitude, 1.0)?;
        let lower = c.where_cond(active, lower, &one)?;
        let upper = c.where_cond(active, upper, &one)?;
        let radius = c.where_cond(active, selected, &one)?;
        let magnitude = c.where_cond(active, magnitude, &zero)?;
        self.finite(&lower, "surveyed Manning radius lower bound")?;
        self.finite(&upper, "surveyed Manning radius upper bound")?;
        if !self.all(&c.gt(&lower, &zero)?)? {
            return Err(ChannelError::numerical(
                "active surveyed Manning radius lower bound is nonpositive: refine recovery or rescale the profile",
            ));
        }
        let contained = c.mul(&c.ge(&radius, &lower)?, &c.le(&radius, &upper)?)?;
        if !self.all(&contained)? {
            return Err(ChannelError::numerical(
                "surveyed Manning selected radius leaves its enclosure: refine recovery or rescale the profile",
            ));
        }
        self.finite(&magnitude, "surveyed Manning original momentum magnitude")?;
        if !self.all(&c.ge(&magnitude, &zero)?)? {
            return Err(ChannelError::input(
                "surveyed Manning momentum magnitude is negative: supply the original absolute momentum",
            ));
        }
        let three = self.constant(&magnitude, 3.0)?;
        let absolute = self.constant(
            &magnitude,
            geometry.numerical.friction_geometry_atol_m4_per_s,
        )?;
        let relative = self.constant(&magnitude, geometry.numerical.friction_geometry_rtol)?;
        let (bound, allowance) = with_interval_arithmetic::<R, _, _, _>(
            c,
            DType::F64,
            "surveyed Manning radius selection",
            |a| {
                let selected = a.point(&radius)?;
                let lo = a.point(&lower)?;
                let hi = a.point(&upper)?;
                let left = a.sub(&selected, &lo)?;
                let right = a.sub(&hi, &selected)?;
                let displacement = c.maximum(left.upper(), right.upper())?;
                let displacement = c.maximum(&displacement, &zero)?;
                let displacement = a.point(&displacement)?;
                let relative_displacement = a.div_positive(&displacement, &lower)?;
                let sensitivity = a.div_positive(&relative_displacement, &three)?;
                let original_magnitude = a.point(&magnitude)?;
                let bound = a.mul(&original_magnitude, &sensitivity)?;
                let relative = a.point(&relative)?;
                let relative_budget = a.mul(&relative, &original_magnitude)?;
                let absolute = a.point(&absolute)?;
                let allowance = a.add(&absolute, &relative_budget)?;
                Ok((bound.upper().clone(), allowance.lower().clone()))
            },
        )
        .map_err(arithmetic_error)?;
        let bound = c.where_cond(active, &bound, &zero)?;
        if !self.all(&c.le(&bound, &allowance)?)? {
            return Err(ChannelError::numerical(
                "surveyed Manning radius-selection bound exceeds its geometry allowance: refine recovery or review friction geometry tolerances",
            ));
        }
        Ok(bound)
    }

    fn surveyed_manning_update(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        radius: &Tensor<R>,
        active: &Tensor<R>,
        dt: f64,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(momentum, 0.0)?;
        let one = self.constant(volume, 1.0)?;
        let inactive = c.eq(active, &zero)?;
        let roughness = c.where_cond(active, &self.manning, &zero)?;
        let roughness_squared = c.mul(&roughness, &roughness)?;
        let original_momentum = c.where_cond(active, momentum, &zero)?;
        let safe_volume = c.where_cond(active, volume, &one)?;
        let velocity = c.abs(&c.div(&original_momentum, &safe_volume)?)?;
        let power = c.pow_scalar(radius, 4.0 / 3.0)?;
        let numerator = c.mul_scalar(
            &c.mul_scalar(
                &c.mul(&roughness_squared, &velocity)?,
                self.definition.config.gravity,
            )?,
            dt,
        )?;
        for (value, name) in [
            (&roughness_squared, "surveyed Manning squared roughness"),
            (&velocity, "surveyed Manning original velocity"),
            (&power, "surveyed Manning radius power"),
            (&numerator, "surveyed Manning resistance"),
        ] {
            self.finite(value, name)?;
            if !self.all(&c.maximum(&inactive, &c.gt(value, &zero)?)?)? {
                return Err(ChannelError::numerical(format!(
                    "active {name} underflows: rescale roughness, water, or interval"
                )));
            }
        }
        let drag = c.div(&numerator, &power)?;
        self.finite(&drag, "surveyed Manning drag")?;
        if !self.all(&c.maximum(&inactive, &c.gt(&drag, &zero)?)?)? {
            return Err(ChannelError::numerical(
                "active surveyed Manning drag underflows: rescale roughness, water, or interval",
            ));
        }
        let denominator = c.add_scalar(&drag, 1.0)?;
        self.finite(&denominator, "surveyed Manning denominator")?;
        let next = c.where_cond(active, &c.div(momentum, &denominator)?, momentum)?;
        self.finite(&next, "surveyed Manning momentum")?;
        let magnitude = c.abs(momentum)?;
        let next_magnitude = c.abs(&next)?;
        let sign_preserved = c.eq(&c.gt(momentum, &zero)?, &c.gt(&next, &zero)?)?;
        let represented = c.mul(
            &c.mul(&c.gt(&next_magnitude, &zero)?, &sign_preserved)?,
            &c.le(&next_magnitude, &magnitude)?,
        )?;
        if !self.all(&c.maximum(&inactive, &represented)?)? {
            return Err(ChannelError::numerical(
                "surveyed Manning momentum loses sign, nonzero magnitude, or damping: reduce the interval or rescale inputs",
            ));
        }
        Ok(next)
    }
}
