use crate::channel::flow::{ChannelError, conserved_update::ExpansionClient, context::Context};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, MidpointOps, NextRepresentableOps,
    ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + CorrectlyRoundedOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + ExpansionClient<R>,
{
    pub(super) fn surveyed_manning_update(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        radius: &Tensor<R>,
        active: &Tensor<R>,
        dt: f64,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.client;
        let zero = self.constant(momentum, 0.0)?;
        let one = self.constant(volume, 1.0)?;
        let inactive = c.eq(active, &zero)?;
        let roughness = c.where_cond(active, &self.manning, &zero)?;
        let roughness_squared = c.mul(&roughness, &roughness)?;
        let original_momentum = c.where_cond(active, momentum, &zero)?;
        let safe_volume = c.where_cond(active, volume, &one)?;
        let velocity = c.abs(&c.div(&original_momentum, &safe_volume)?)?;
        let power = c.cr_pow_scalar(radius, 4.0 / 3.0)?;
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
        Ok((next, c.where_cond(active, &denominator, &one)?))
    }
}
