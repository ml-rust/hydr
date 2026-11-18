use super::{
    ChannelError, conserved_point::ConservedPoint, context::Context, geometry::TensorGeometry,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, MidpointOps, NextRepresentableOps,
    ReduceOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TensorOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + TwoSumOps<R>,
{
    pub fn friction(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        dt: f64,
    ) -> Result<Tensor<R>, ChannelError> {
        if matches!(&self.geometry, TensorGeometry::Surveyed(_)) {
            return Ok(self.surveyed_friction(volume, momentum, dt)?.momentum);
        }
        // Analytic geometry stores `+0.0` low parts.
        let fields = self.fields(
            &ConservedPoint::from_high(volume)?,
            &ConservedPoint::from_high(momentum)?,
        )?;
        let c = &self.client;
        let zero = self.constant(volume, 0.0)?;
        let one = self.constant(volume, 1.0)?;
        let active = c.mul(
            &c.gt(&self.manning, &zero)?,
            &c.gt(&c.abs(momentum)?, &zero)?,
        )?;
        let perimeter = self.section_perimeter(&fields.depth)?;
        self.finite(&perimeter, "wetted perimeter")?;
        let radius = match &self.geometry {
            TensorGeometry::Rectangular(_) => c.div(&fields.area, &perimeter)?,
            TensorGeometry::Triangular(_) | TensorGeometry::Surveyed(_) => {
                let wet = c.gt(&fields.area, &zero)?;
                let safe_perimeter = c.where_cond(&wet, &perimeter, &one)?;
                c.div(&fields.area, &safe_perimeter)?
            }
        };
        let radius = c.where_cond(&active, &radius, &one)?;
        let power = c.cr_pow_scalar(&radius, 4.0 / 3.0)?;
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
        ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReach, ChannelReachDefinition,
        RectangularChannelCell, TriangularChannelCell,
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
                geometry: ChannelGeometry::Rectangular {
                    cells: vec![RectangularChannelCell {
                        id: "cell".into(),
                        length_m: 1.0,
                        width_m: 2.0,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: 2.0,
                        manning_n: roughness,
                    }],
                },
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
            let next = reach.context.friction(
                &reach.accepted.volume.high,
                &reach.accepted.momentum.high,
                0.1,
            )?;
            let expected =
                momentum / (1.0 + 0.1 * 9.81 * 0.03_f64.powi(2) / 0.5_f64.powf(4.0 / 3.0));
            assert!((next.item::<f64>()? - expected).abs() <= 1e-12);
            assert_eq!(reach.accepted.volume.high.item::<f64>()?, 2.0);
        }
        Ok(())
    }

    #[test]
    fn triangular_manning_decay_uses_both_bank_contacts() -> Result<(), Box<dyn std::error::Error>>
    {
        for (left_slope, right_slope) in [(1.0_f64, 1.0_f64), (0.0_f64, 2.0_f64)] {
            let area = (left_slope + right_slope) / 2.0;
            let perimeter = left_slope.hypot(1.0) + right_slope.hypot(1.0);
            let radius = area / perimeter;
            for velocity in [-1.0, 1.0] {
                let client = CpuClient::new(CpuDevice::new())
                    .with_parallelism(ParallelismConfig::new(Some(1), None));
                let reach = ChannelReach::<CpuRuntime>::new(
                    client,
                    ChannelReachDefinition {
                        lateral_bindings: Vec::new(),
                        vertical_datum: "synthetic datum".into(),
                        provenance: "triangular fixed-area Manning decay fixture".into(),
                        geometry: ChannelGeometry::PrismaticTriangular {
                            left_bank_slope_h_per_v: left_slope,
                            right_bank_slope_h_per_v: right_slope,
                            bed_elevation_m: 0.0,
                            cells: vec![TriangularChannelCell {
                                id: "cell".into(),
                                length_m: 1.0,
                                maximum_depth_m: 2.0,
                                manning_n: 0.03,
                            }],
                        },
                        config: ChannelConfig::default(),
                    },
                    ChannelInitialState {
                        time_s: 0.0,
                        volume_m3: vec![area],
                        longitudinal_momentum_m4_per_s: vec![area * velocity],
                    },
                )?;
                let next = reach.context.friction(
                    &reach.accepted.volume.high,
                    &reach.accepted.momentum.high,
                    0.1,
                )?;
                let expected = area * velocity
                    / (1.0 + 0.1 * 9.81 * 0.03_f64.powi(2) / radius.powf(4.0 / 3.0));
                assert!((next.item::<f64>()? - expected).abs() <= 1e-12);
                assert_eq!(reach.accepted.volume.high.item::<f64>()?, area);
                let dry = reach.context.tensor(&[0.0])?;
                assert_eq!(reach.context.friction(&dry, &dry, 0.1)?.item::<f64>()?, 0.0);
            }
        }
        Ok(())
    }

    #[test]
    fn unresolved_positive_resistance_rejects_only_moving_water()
    -> Result<(), Box<dyn std::error::Error>> {
        let moving = reach(2.0, 1e-200)?;
        assert!(matches!(
            moving.context.friction(
                &moving.accepted.volume.high,
                &moving.accepted.momentum.high,
                0.1
            ),
            Err(ChannelError::Numerical { .. })
        ));
        for roughness in [1e-200, 1e200] {
            let stationary = reach(0.0, roughness)?;
            assert_eq!(
                stationary
                    .context
                    .friction(
                        &stationary.accepted.volume.high,
                        &stationary.accepted.momentum.high,
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
                .friction(
                    &extreme.accepted.volume.high,
                    &extreme.accepted.momentum.high,
                    100.0
                )?
                .item::<f64>()?,
            0.0
        );
        assert_eq!(extreme.accepted.volume.high.item::<f64>()?, 2e-100);
        Ok(())
    }
}
