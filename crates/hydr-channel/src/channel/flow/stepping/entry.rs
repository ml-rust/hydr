use super::super::{
    ChannelError, ChannelForcingInterval, ChannelReach, ChannelStep, ChannelTensorForcingInterval,
    forcing::BoundChannelForcing,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
    },
};

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + TensorOps<R>
        + NextRepresentableOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Calculate one accepted interval without modifying the input reach.
    ///
    /// Rate inputs permit shortening. Integrated inputs require fresh inputs for a shorter span.
    pub fn step(
        &self,
        requested_end_s: f64,
        forcing: &ChannelForcingInterval,
    ) -> Result<ChannelStep<R>, ChannelError> {
        let bound = self
            .context
            .lateral
            .bind_host(&self.context.client, forcing)?;
        self.step_bound(requested_end_s, &bound)
    }

    /// Calculate one accepted interval using original device-resident forcing operands.
    ///
    /// Every tensor uses F64, shape `[1]`, and the configured reach device.
    pub fn step_tensor(
        &self,
        requested_end_s: f64,
        forcing: &ChannelTensorForcingInterval<R>,
    ) -> Result<ChannelStep<R>, ChannelError> {
        let bound = self.context.lateral.bind_native(
            &self.context.client,
            forcing.start_s,
            forcing.end_s,
            forcing.left.clone(),
            forcing.right.clone(),
            forcing.lateral_inputs.clone(),
        )?;
        self.step_bound(requested_end_s, &bound)
    }

    pub(in super::super) fn step_bound(
        &self,
        requested_end_s: f64,
        forcing: &BoundChannelForcing<R>,
    ) -> Result<ChannelStep<R>, ChannelError> {
        self.step_bound_policy(requested_end_s, forcing, false, None)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::flux::tests::native_flux;
    use super::super::super::surveyed_update::tests::{TestResult, cpu, moving_reach, walls};
    use super::super::super::{
        ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelTensorBoundary,
    };
    use super::*;
    use crate::channel::{
        ChannelBoundary, ChannelConfig, ChannelInitialState, ChannelReachDefinition,
        RectangularChannelCell,
    };
    use crate::forcing::WaterAmount;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn rounded_numerical_transfers_preserve_water_and_consume_tolerance()
    -> Result<(), Box<dyn std::error::Error>> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let definition = ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "tiny positive wet/dry transfer arithmetic fixture".into(),
            geometry: crate::channel::ChannelGeometry::Rectangular {
                cells: (0..2)
                    .map(|index| RectangularChannelCell {
                        id: format!("cell-{index}"),
                        length_m: 1.0,
                        width_m: 1.0,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: 1.0,
                        manning_n: 0.0,
                    })
                    .collect(),
            },
            lateral_bindings: vec![],
            config: ChannelConfig::default(),
        };
        let reach = ChannelReach::<CpuRuntime>::new(
            client,
            definition,
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![1e-100, 0.0],
                longitudinal_momentum_m4_per_s: vec![0.0; 2],
            },
        )?;
        let forcing = ChannelForcingInterval {
            start_s: 0.0,
            end_s: 1e-200,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: vec![],
        };
        let fields = reach
            .context
            .fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let flux = native_flux(&reach.context, &fields, &forcing.left, &forcing.right)?;
        let original_rate = flux.water.narrow(0, 1, 1)?.item::<f64>()?;
        assert!(original_rate > 0.0);
        assert_eq!(original_rate * forcing.end_s, 0.0);
        let original = reach.snapshot_data()?;
        let accepted = reach.step(forcing.end_s, &forcing)?.next;
        assert_eq!(accepted.state()?, reach.state()?);
        let receipt = accepted.last_step()?.ok_or("missing accepted interval")?;
        assert!(receipt.face_transfer_underflow_bound_m3[1] > 0.0);
        assert!(receipt.transfer_underflow_bound_m3 > 0.0);
        assert!(
            accepted.ledger().numerical_transfer_underflow_bound_m3
                >= receipt.transfer_underflow_bound_m3
        );
        assert_eq!(reach.snapshot_data()?, original);
        let mut tight_definition = reach.definition().clone();
        tight_definition.config.volume_absolute = f64::from_bits(1);
        tight_definition.config.volume_relative = 0.0;
        let tight = ChannelReach::<CpuRuntime>::new(
            reach.context.client.clone(),
            tight_definition,
            reach.context.original_initial.clone(),
        )?;
        let before = tight.snapshot_data()?;
        assert!(matches!(
            tight.step(forcing.end_s, &forcing),
            Err(ChannelError::Numerical { .. })
        ));
        assert_eq!(tight.snapshot_data()?, before);
        Ok(())
    }

    #[test]
    fn positive_channel_motion_cannot_disappear_in_per_width_reconstruction()
    -> Result<(), Box<dyn std::error::Error>> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let reach = ChannelReach::<CpuRuntime>::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "synthetic datum".into(),
                provenance: "wide shallow moving-water arithmetic fixture".into(),
                geometry: crate::channel::ChannelGeometry::Rectangular {
                    cells: vec![RectangularChannelCell {
                        id: "wide-cell".into(),
                        length_m: 1.0,
                        width_m: 1e200,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: 1.0,
                        manning_n: 0.0,
                    }],
                },
                lateral_bindings: vec![],
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![1.0],
                longitudinal_momentum_m4_per_s: vec![1e-200],
            },
        )?;
        let original = reach.snapshot_data()?;
        let state = reach.state()?;
        assert!(state[0].depth_m > 0.0);
        assert!(state[0].velocity_m_per_s.is_some_and(|u| u > 0.0));
        assert!(state[0].discharge_m3_per_s > 0.0);
        let forcing = ChannelForcingInterval {
            start_s: 0.0,
            end_s: 0.1,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: vec![],
        };
        assert!(matches!(
            reach.step(forcing.end_s, &forcing),
            Err(ChannelError::Numerical { .. })
        ));
        assert_eq!(reach.snapshot_data()?, original);
        Ok(())
    }

    #[test]
    fn rounded_pressure_impulses_consume_the_configured_momentum_tolerance()
    -> Result<(), Box<dyn std::error::Error>> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let reach = ChannelReach::<CpuRuntime>::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "synthetic datum".into(),
                provenance: "tiny finite pressure impulse arithmetic fixture".into(),
                geometry: crate::channel::ChannelGeometry::Rectangular {
                    cells: (0..2)
                        .map(|index| RectangularChannelCell {
                            id: format!("cell-{index}"),
                            length_m: 1.0,
                            width_m: 1.0,
                            bed_elevation_m: 0.0,
                            maximum_depth_m: 1.0,
                            manning_n: 0.0,
                        })
                        .collect(),
                },
                lateral_bindings: vec![],
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![1e-100, 0.0],
                longitudinal_momentum_m4_per_s: vec![0.0; 2],
            },
        )?;
        let forcing = ChannelForcingInterval {
            start_s: 0.0,
            end_s: 1e-150,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: vec![],
        };
        let fields = reach
            .context
            .fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let flux = native_flux(&reach.context, &fields, &forcing.left, &forcing.right)?;
        assert!(flux.water.narrow(0, 1, 1)?.item::<f64>()? * forcing.end_s > 0.0);
        let original_pressure = flux.momentum.narrow(0, 1, 1)?.item::<f64>()?;
        assert!(original_pressure > 0.0);
        assert_eq!(original_pressure * forcing.end_s, 0.0);
        let original = reach.snapshot_data()?;
        let accepted = reach.step(forcing.end_s, &forcing)?.next;
        assert!(accepted.state()?[1].volume_m3 > 0.0);
        let receipt = accepted.last_step()?.ok_or("missing accepted interval")?;
        assert!(receipt.max_momentum_underflow_bound_m4_per_s > 0.0);
        assert!(receipt.max_momentum_underflow_bound_m4_per_s <= reach.config().momentum_absolute);
        assert!(
            receipt
                .momentum_underflow_bound_m4_per_s
                .iter()
                .any(|bound| *bound > 0.0)
        );
        assert_eq!(reach.snapshot_data()?, original);
        let mut tight_definition = reach.definition().clone();
        tight_definition.config.momentum_absolute = f64::from_bits(1);
        tight_definition.config.momentum_relative = 0.0;
        let tight = ChannelReach::<CpuRuntime>::new(
            reach.context.client.clone(),
            tight_definition,
            reach.context.original_initial.clone(),
        )?;
        let before = tight.snapshot_data()?;
        assert!(matches!(
            tight.step(forcing.end_s, &forcing),
            Err(ChannelError::Numerical { .. })
        ));
        assert_eq!(tight.snapshot_data()?, before);
        Ok(())
    }

    #[test]
    fn positive_wave_speed_cannot_disappear_during_width_weighting()
    -> Result<(), Box<dyn std::error::Error>> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let reach = ChannelReach::<CpuRuntime>::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "synthetic datum".into(),
                provenance: "narrow deep still-water CFL arithmetic fixture".into(),
                geometry: crate::channel::ChannelGeometry::Rectangular {
                    cells: vec![RectangularChannelCell {
                        id: "cell".into(),
                        length_m: 1.0,
                        width_m: 1e-300,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: 1e201,
                        manning_n: 0.0,
                    }],
                },
                lateral_bindings: vec![],
                config: ChannelConfig {
                    gravity: 1e-300,
                    ..ChannelConfig::default()
                },
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![1e-100],
                longitudinal_momentum_m4_per_s: vec![0.0],
            },
        )?;
        let forcing = ChannelForcingInterval {
            start_s: 0.0,
            end_s: 0.1,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: vec![],
        };
        let fields = reach
            .context
            .fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let flux = native_flux(&reach.context, &fields, &forcing.left, &forcing.right)?;
        let speed = flux.speed.rightward.narrow(0, 0, 1)?.item::<f64>()?;
        assert!(speed > 0.0);
        assert_eq!(
            speed
                * match &reach.definition().geometry {
                    crate::channel::ChannelGeometry::Rectangular { cells } => cells[0].width_m,
                    _ => unreachable!(),
                },
            0.0
        );
        let original = reach.snapshot_data()?;
        assert!(matches!(
            reach.step(forcing.end_s, &forcing),
            Err(ChannelError::Numerical { .. })
        ));
        assert_eq!(reach.snapshot_data()?, original);
        Ok(())
    }

    #[test]
    fn step_that_exceeds_start_cfl_retries_and_reports_it() -> Result<(), Box<dyn std::error::Error>>
    {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let definition = ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "retrying one-meter wall-bounded cell fixture".into(),
            geometry: crate::channel::ChannelGeometry::Rectangular {
                cells: (0..2)
                    .map(|index| RectangularChannelCell {
                        id: format!("cell-{index}"),
                        length_m: 1.0,
                        width_m: 1.0,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: 4.0,
                        manning_n: 0.0,
                    })
                    .collect(),
            },
            lateral_bindings: vec![],
            config: ChannelConfig::default(),
        };
        let reach = ChannelReach::<CpuRuntime>::new(
            client,
            definition,
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![1.0, 1.0],
                longitudinal_momentum_m4_per_s: vec![0.0; 2],
            },
        )?;
        let forcing = ChannelForcingInterval {
            start_s: 0.0,
            end_s: 1.0,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: vec![],
        };
        let accepted = reach.step(forcing.end_s, &forcing)?.next;
        let receipt = accepted.last_step()?.ok_or("missing accepted interval")?;
        assert!(receipt.retries > 0);
        assert!(receipt.end_s < forcing.end_s);
        Ok(())
    }

    /// A fast inflow into one cell drives it into the left wall.
    #[test]
    fn candidate_speed_entry_rejection_shortens_the_interval() -> TestResult {
        // The interval stays inside the accepted-start CFL limit.
        let (rate, velocity, end) = (20.0, -50.0, 0.3);
        // The full-interval candidate holds V = 10 + rate·end and M = rate·end·velocity.
        let full = moving_reach::<CpuRuntime>(cpu(), &[16.0], &[-300.0], &[0.0])?;
        let wall = ChannelTensorBoundary::Wall;
        let fields = full
            .context
            .fields(&full.accepted.volume, &full.accepted.momentum)?;
        let error = full
            .context
            .candidate_speed(&fields, &wall, &wall)
            .err()
            .ok_or("the full-interval candidate must reject")?;
        assert!(
            matches!(&error, ChannelError::Numerical { context, .. }
                if context.contains("surveyed reach Riemann")),
            "{error}"
        );
        let rest = moving_reach::<CpuRuntime>(cpu(), &[10.0], &[0.0], &[0.0])?;
        let mut definition = rest.definition().clone();
        definition.lateral_bindings = vec![ChannelLateralBinding {
            boundary_id: "in".into(),
            cell_id: "cell-0".into(),
            direction: ChannelLateralDirection::Inflow,
            ownership_assertion: "Synthetic fast inflow".into(),
        }];
        let origin = ChannelReach::<CpuRuntime>::new(
            cpu(),
            definition,
            rest.context.original_initial.clone(),
        )?;
        let mut forcing = walls(0.0, end);
        forcing.lateral_inputs = vec![ChannelLateralInput::Inflow {
            boundary_id: "in".into(),
            amount: WaterAmount::Rate {
                rate_m3_per_s: rate,
            },
            longitudinal_velocity_m_per_s: velocity,
        }];
        let next = origin.step(end, &forcing)?.next;
        let receipt = next.last_step()?.ok_or("missing accepted interval")?;
        assert!(receipt.retries > 0);
        assert!(next.time() > 0.0 && next.time() < end);
        assert!(
            receipt.accepted_start_cfl_limit_s > end,
            "start CFL={}, end={}, retries={}",
            receipt.accepted_start_cfl_limit_s,
            next.time(),
            receipt.retries
        );
        Ok(())
    }
}
