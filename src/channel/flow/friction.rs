use super::{ChannelError, context::Context};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    pub fn friction(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        dt: f64,
    ) -> Result<Tensor<R>, ChannelError> {
        let fields = self.fields(volume, momentum)?;
        let c = &self.client;
        let zero = self.constant(volume, 0.0)?;
        let one = self.constant(volume, 1.0)?;
        let active = c.mul(
            &c.gt(&self.manning, &zero)?,
            &c.gt(&c.abs(momentum)?, &zero)?,
        )?;
        let perimeter = c.add(&self.width, &c.mul_scalar(&fields.depth, 2.0)?)?;
        self.finite(&perimeter, "wetted perimeter")?;
        let radius = c.div(&fields.area, &perimeter)?;
        let radius = c.where_cond(&active, &radius, &one)?;
        let power = c.pow_scalar(&radius, 4.0 / 3.0)?;
        let roughness = c.where_cond(&active, &self.manning, &zero)?;
        let roughness_squared = c.mul(&roughness, &roughness)?;
        let velocity = c.where_cond(&active, &c.abs(&fields.velocity)?, &zero)?;
        let numerator = c.mul_scalar(
            &c.mul_scalar(
                &c.mul(&roughness_squared, &velocity)?,
                self.definition.config.gravity,
            )?,
            dt,
        )?;
        let inactive = c.eq(&active, &zero)?;
        let mut represented = c.maximum(&inactive, &c.gt(&power, &zero)?)?;
        represented = c.mul(
            &represented,
            &c.maximum(&inactive, &c.gt(&roughness_squared, &zero)?)?,
        )?;
        represented = c.mul(
            &represented,
            &c.maximum(&inactive, &c.gt(&numerator, &zero)?)?,
        )?;
        if !self.all(&represented)? {
            return Err(ChannelError::numerical(
                "active Manning resistance underflows: rescale roughness, water, or interval",
            ));
        }
        let drag = c.div(&numerator, &power)?;
        if !self.all(&c.maximum(&inactive, &c.gt(&drag, &zero)?)?)? {
            return Err(ChannelError::numerical(
                "active Manning drag underflows: rescale roughness, water, or interval",
            ));
        }
        let denominator = c.add_scalar(&drag, 1.0)?;
        self.finite(&denominator, "Manning denominator")?;
        let next = c.div(momentum, &denominator)?;
        self.finite(&next, "Manning momentum")?;
        let represented = c.maximum(&c.eq(momentum, &zero)?, &c.gt(&c.abs(&next)?, &zero)?)?;
        if !self.all(&represented)? {
            return Err(ChannelError::numerical(
                "Manning momentum underflows to zero: reduce the interval or rescale inputs",
            ));
        }
        Ok(next)
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

    fn reach(momentum: f64, roughness: f64) -> Result<ChannelReach<CpuRuntime>, ChannelError> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        ChannelReach::new(
            client,
            ChannelReachDefinition {
                lateral_bindings: Vec::new(),
                vertical_datum: "synthetic datum".into(),
                provenance: "exact rectangular fixed-area resistance fixture".into(),
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 1.0,
                    width_m: 2.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 2.0,
                    manning_n: roughness,
                }],
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![2.0],
                longitudinal_momentum_m4_per_s: vec![momentum],
            },
        )
    }

    #[test]
    fn fixed_area_manning_decay_uses_wetted_hydraulic_radius()
    -> Result<(), Box<dyn std::error::Error>> {
        for momentum in [-2.0, 2.0] {
            let reach = reach(momentum, 0.03)?;
            let next =
                reach
                    .context
                    .friction(&reach.accepted.volume, &reach.accepted.momentum, 0.1)?;
            let expected =
                momentum / (1.0 + 0.1 * 9.81 * 0.03_f64.powi(2) / 0.5_f64.powf(4.0 / 3.0));
            assert!((next.item::<f64>()? - expected).abs() <= 1e-12);
            assert_eq!(reach.accepted.volume.item::<f64>()?, 2.0);
        }
        Ok(())
    }

    #[test]
    fn unresolved_positive_resistance_rejects_only_moving_water()
    -> Result<(), Box<dyn std::error::Error>> {
        let moving = reach(2.0, 1e-200)?;
        assert!(matches!(
            moving
                .context
                .friction(&moving.accepted.volume, &moving.accepted.momentum, 0.1),
            Err(ChannelError::Numerical { .. })
        ));
        for roughness in [1e-200, 1e200] {
            let stationary = reach(0.0, roughness)?;
            assert_eq!(
                stationary
                    .context
                    .friction(
                        &stationary.accepted.volume,
                        &stationary.accepted.momentum,
                        0.1
                    )?
                    .item::<f64>()?,
                0.0
            );
        }
        Ok(())
    }

    #[test]
    fn inactive_resistance_avoids_overflow_in_gravity_times_duration()
    -> Result<(), Box<dyn std::error::Error>> {
        let ordinary = reach(0.0, 0.0)?;
        let mut definition = ordinary.definition().clone();
        definition.config.gravity = 1e308;
        let mut initial = ordinary.context.original_initial.clone();
        initial.volume_m3 = vec![2e-100];
        let extreme =
            ChannelReach::<CpuRuntime>::new(ordinary.context.client.clone(), definition, initial)?;
        assert_eq!(
            extreme
                .context
                .friction(&extreme.accepted.volume, &extreme.accepted.momentum, 100.0)?
                .item::<f64>()?,
            0.0
        );
        assert_eq!(extreme.accepted.volume.item::<f64>()?, 2e-100);
        Ok(())
    }
}
