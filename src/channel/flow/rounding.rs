//! Bounds for detected erased channel products, excluding other rounding and solution errors.

use super::{ChannelError, context::Context};
use crate::numerics::subcatchment::bounds::rounding_with_client;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) struct Pressure<R: Runtime<DType = DType>> {
    pub value: Tensor<R>,
    pub bound: Tensor<R>,
    pub unit_bound: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    pub fn bound_add(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let sum = c.add(a, b)?;
        let padded = c.add(&sum, &rounding_with_client::<R>(c, &sum)?)?;
        let zero = self.constant(a, 0.0)?;
        let bound = c.where_cond(
            &c.eq(a, &zero)?,
            b,
            &c.where_cond(&c.eq(b, &zero)?, a, &padded)?,
        )?;
        self.finite(&bound, "detected underflow bound sum")?;
        Ok(bound)
    }

    pub fn bound_scale(
        &self,
        bound: &Tensor<R>,
        factor: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let magnitude = c.abs(factor)?;
        let product = c.mul(bound, &magnitude)?;
        let padded = c.add(&product, &rounding_with_client::<R>(c, &product)?)?;
        let zero = self.constant(bound, 0.0)?;
        let active = c.mul(&c.gt(bound, &zero)?, &c.gt(&magnitude, &zero)?)?;
        let result = c.where_cond(&active, &padded, &zero)?;
        self.finite(&result, "scaled detected underflow bound")?;
        Ok(result)
    }

    pub fn product_bound(
        &self,
        value: &Tensor<R>,
        bound: &Tensor<R>,
        factor: &Tensor<R>,
        product: &Tensor<R>,
        name: &str,
    ) -> Result<Tensor<R>, ChannelError> {
        self.finite(product, name)?;
        let c = &self.client;
        let zero = self.constant(product, 0.0)?;
        let active = c.mul(
            &c.gt(&c.abs(value)?, &zero)?,
            &c.gt(&c.abs(factor)?, &zero)?,
        )?;
        let erased = c.mul(&active, &c.eq(product, &zero)?)?;
        let charge = c.where_cond(&erased, &self.constant(product, f64::from_bits(1))?, &zero)?;
        self.bound_add(&self.bound_scale(bound, factor)?, &charge)
    }

    pub fn pressure(
        &self,
        depth: &Tensor<R>,
        width: &Tensor<R>,
    ) -> Result<Pressure<R>, ChannelError> {
        let c = &self.client;
        let half_g = 0.5 * self.definition.config.gravity;
        let scalar = self.constant(depth, half_g)?;
        let scalar_bound = self.constant(
            depth,
            if half_g == 0.0 {
                f64::from_bits(1)
            } else {
                0.0
            },
        )?;
        let half_gh = c.mul_scalar(depth, half_g)?;
        let first_bound = self.product_bound(
            &scalar,
            &scalar_bound,
            depth,
            &half_gh,
            "hydrostatic gravity-depth product",
        )?;
        let unit = c.mul(&half_gh, depth)?;
        let unit_bound = self.product_bound(
            &half_gh,
            &first_bound,
            depth,
            &unit,
            "per-width hydrostatic pressure",
        )?;
        let value = c.mul(&unit, width)?;
        let bound = self.product_bound(
            &unit,
            &unit_bound,
            width,
            &value,
            "full-width hydrostatic pressure",
        )?;
        Ok(Pressure {
            value,
            bound,
            unit_bound,
        })
    }

    pub fn integrate_momentum(
        &self,
        rate: &Tensor<R>,
        rate_bound: &Tensor<R>,
        dt: f64,
        name: &str,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let integral = self.client.mul_scalar(rate, dt)?;
        let bound =
            self.product_bound(rate, rate_bound, &self.constant(rate, dt)?, &integral, name)?;
        Ok((integral, bound))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::{
        ChannelConfig, ChannelInitialState, ChannelReach, ChannelReachDefinition,
        RectangularChannelCell,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn positive_pressure_charge_survives_width_and_duration_scaling()
    -> Result<(), Box<dyn std::error::Error>> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let reach = ChannelReach::<CpuRuntime>::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "synthetic datum".into(),
                provenance: "detected pressure underflow bound arithmetic fixture".into(),
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 1.0,
                    width_m: 1.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 2.0,
                    manning_n: 0.0,
                }],
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![1.0],
                longitudinal_momentum_m4_per_s: vec![0.0],
            },
        )?;
        let context = &reach.context;
        let delta = context.tensor(&[f64::from_bits(1)])?;
        let tiny_factor = context.tensor(&[1e-200])?;
        let zero = context.tensor(&[0.0])?;
        assert_eq!(
            context.client.mul(&delta, &tiny_factor)?.item::<f64>()?,
            0.0
        );
        let width_scaled = context.bound_scale(&delta, &tiny_factor)?;
        assert!(width_scaled.item::<f64>()? > 0.0);
        let duration_scaled = context.bound_scale(&width_scaled, &tiny_factor)?;
        assert!(duration_scaled.item::<f64>()? > 0.0);
        assert_eq!(
            context.bound_scale(&zero, &tiny_factor)?.item::<f64>()?,
            0.0
        );
        assert_eq!(context.bound_add(&zero, &zero)?.item::<f64>()?, 0.0);
        let pressure = context.pressure(&context.tensor(&[1e-170])?, &context.tensor(&[1e200])?)?;
        assert_eq!(pressure.value.item::<f64>()?, 0.0);
        assert!(pressure.bound.item::<f64>()? > 0.0);
        Ok(())
    }
}
