use super::super::{
    ChannelError, ChannelReach, ChannelTensorBoundary, forcing::BoundChannelForcing,
};
use super::{ChannelCouplingForcing, ChannelCouplingTrial, ChannelReservoirOperands};
use numr::dtype::DType;
use numr::ops::{CompareOps, MidpointOps, ScalarOps, TwoSumOps};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: numr::ops::TensorOps<R>
        + CompareOps<R>
        + numr::ops::CorrectlyRoundedOps<R>
        + MidpointOps<R>
        + numr::ops::NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + BatchedBisectionAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    #[doc(hidden)]
    pub fn trial_coupled(
        &self,
        requested_end_s: f64,
        forcing: &ChannelCouplingForcing<R>,
        left: Option<ChannelReservoirOperands<'_, R>>,
        right: Option<ChannelReservoirOperands<'_, R>>,
    ) -> Result<ChannelCouplingTrial<R>, ChannelError> {
        let endpoint = |external: &Option<ChannelTensorBoundary<R>>,
                        reservoir: Option<ChannelReservoirOperands<'_, R>>|
         -> Result<ChannelTensorBoundary<R>, ChannelError> {
            match (external, reservoir) {
                (Some(external), None) => Ok(external.clone()),
                (None, Some(reservoir)) => Ok(ChannelTensorBoundary::QuiescentReservoir {
                    elevation_m: reservoir.elevation_m.clone(),
                    entrance_loss_coefficient: reservoir.entrance_loss_coefficient.clone(),
                }
                .into_owned(&self.context.client)?),
                _ => Err(ChannelError::input(
                    "endpoint ownership: supply exactly one external policy or connected reservoir operand",
                )),
            }
        };
        let left = endpoint(&forcing.left, left)?;
        let right = endpoint(&forcing.right, right)?;
        self.context.lateral.check_endpoints(
            &self.context.client,
            forcing.start_s,
            forcing.end_s,
            Some(&left),
            Some(&right),
        )?;
        let bound = BoundChannelForcing {
            definition: Arc::clone(&forcing.lateral.definition),
            start_s: forcing.start_s,
            end_s: forcing.end_s,
            left,
            right,
            lateral: forcing.lateral.clone(),
        };
        match self.step_bound_policy(requested_end_s, &bound, true, forcing.reserved.as_deref()) {
            Ok(step) => Ok(ChannelCouplingTrial::Accepted(Box::new(step.next))),
            Err(ChannelError::RecomputeInterval {
                suggested_end_s,
                reason,
                ..
            }) if !forcing.has_interval_volumes() => Ok(ChannelCouplingTrial::Shorter {
                suggested_end_s,
                reason,
            }),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{interval_reach, zero_source};
    use super::*;
    use crate::channel::{
        ChannelBoundary, ChannelConfig, ChannelGeometry, ChannelInitialState,
        ChannelReachDefinition, RectangularChannelCell,
    };
    use numr::ops::BinaryOps;
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use numr::tensor::Tensor;

    #[test]
    fn reservoir_buffer_reuse_preserves_accepted_forcing() -> Result<(), Box<dyn std::error::Error>>
    {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let reach = ChannelReach::<CpuRuntime>::new(
            client.clone(),
            ChannelReachDefinition {
                vertical_datum: "synthetic datum".into(),
                provenance: "reservoir operand ownership fixture".into(),
                geometry: ChannelGeometry::Rectangular {
                    cells: vec![RectangularChannelCell {
                        id: "cell".into(),
                        length_m: 10.0,
                        width_m: 1.0,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: 2.0,
                        manning_n: 0.0,
                    }],
                },
                lateral_bindings: vec![],
                config: ChannelConfig {
                    gravity: 1.0,
                    ..ChannelConfig::default()
                },
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![10.0],
                longitudinal_momentum_m4_per_s: vec![0.0],
            },
        )?;
        let elevation = Tensor::from_slice(&[1.0], &[1], client.device())?;
        let loss = Tensor::from_slice(&[0.0], &[1], client.device())?;
        let forcing =
            reach.bind_coupling_host(0.0, 0.01, Some(&ChannelBoundary::Wall), None, &[])?;
        let accepted = match reach.trial_coupled(
            0.01,
            &forcing,
            None,
            Some(ChannelReservoirOperands {
                elevation_m: &elevation,
                entrance_loss_coefficient: &loss,
            }),
        )? {
            ChannelCouplingTrial::Accepted(accepted) => *accepted,
            ChannelCouplingTrial::Shorter { .. } => {
                return Err("unexpected shorter interval".into());
            }
        };
        let snapshot = accepted.snapshot_data()?;
        client.copy_into(
            &elevation,
            &Tensor::from_slice(&[1.25], &[1], client.device())?,
        )?;
        client.copy_into(&loss, &Tensor::from_slice(&[2.0], &[1], client.device())?)?;
        assert_eq!(accepted.snapshot_data()?, snapshot);
        assert_eq!(elevation.item::<f64>()?, 1.25);
        assert_eq!(loss.item::<f64>()?, 2.0);
        Ok(())
    }

    #[test]
    fn rate_common_interval_returns_shorter_without_advancing_origin()
    -> Result<(), Box<dyn std::error::Error>> {
        let reach = interval_reach()?;
        let original = reach.snapshot_data()?;
        let inputs = [zero_source(crate::forcing::WaterAmount::Rate {
            rate_m3_per_s: 0.0,
        })];
        let forcing = reach.bind_coupling_host(
            2.0,
            2.25,
            Some(&ChannelBoundary::Wall),
            Some(&ChannelBoundary::Wall),
            &inputs,
        )?;
        match reach.trial_coupled(2.25, &forcing, None, None)? {
            ChannelCouplingTrial::Shorter {
                suggested_end_s,
                reason,
            } => {
                assert_eq!(suggested_end_s.to_bits(), 2.125_f64.to_bits());
                assert!(!reason.is_empty());
            }
            ChannelCouplingTrial::Accepted(_) => {
                return Err("unexpected accepted common interval".into());
            }
        }
        let ordinary = reach
            .step(
                2.25,
                &super::super::super::ChannelForcingInterval {
                    start_s: 2.0,
                    end_s: 2.25,
                    left: ChannelBoundary::Wall,
                    right: ChannelBoundary::Wall,
                    lateral_inputs: inputs.to_vec(),
                },
            )?
            .next;
        assert_eq!(ordinary.time().to_bits(), 2.125_f64.to_bits());
        let coupled = match reach.trial_coupled(2.125, &forcing, None, None)? {
            ChannelCouplingTrial::Accepted(accepted) => *accepted,
            ChannelCouplingTrial::Shorter { .. } => {
                return Err("unexpected repeated shortening".into());
            }
        };
        assert_eq!(coupled.state()?, ordinary.state()?);
        assert_eq!(reach.snapshot_data()?, original);
        Ok(())
    }

    #[test]
    fn zero_integrated_input_requires_fresh_exact_span_binding()
    -> Result<(), Box<dyn std::error::Error>> {
        let reach = interval_reach()?;
        let original = reach.snapshot_data()?;
        let inputs = [zero_source(crate::forcing::WaterAmount::IntervalVolume {
            volume_m3: -0.0,
        })];
        let forcing = reach.bind_coupling_host(
            2.0,
            2.25,
            Some(&ChannelBoundary::Wall),
            Some(&ChannelBoundary::Wall),
            &inputs,
        )?;
        assert!(forcing.has_interval_volumes());
        let ordinary = super::super::super::ChannelForcingInterval {
            start_s: 2.0,
            end_s: 2.25,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: inputs.to_vec(),
        };
        for result in [
            reach.trial_coupled(2.25, &forcing, None, None).map(|_| ()),
            reach.step(2.25, &ordinary).map(|_| ()),
        ] {
            match result {
                Err(ChannelError::RecomputeInterval {
                    start_s,
                    requested_end_s,
                    suggested_end_s,
                    ..
                }) => {
                    assert_eq!(start_s.to_bits(), 2.0_f64.to_bits());
                    assert_eq!(requested_end_s.to_bits(), 2.25_f64.to_bits());
                    assert_eq!(suggested_end_s.to_bits(), 2.125_f64.to_bits());
                }
                _ => return Err("expected integrated interval recomputation".into()),
            }
        }
        assert!(matches!(
            reach.trial_coupled(2.125, &forcing, None, None),
            Err(ChannelError::InvalidInput { .. })
        ));
        let projected = forcing.project_external(&reach.context.client)?;
        assert_eq!(projected.start_s.to_bits(), 2.0_f64.to_bits());
        assert_eq!(projected.end_s.to_bits(), 2.25_f64.to_bits());
        assert!(projected.lateral_inputs[0].same_bits(&inputs[0]));
        let fresh = reach.bind_coupling_host(
            2.0,
            2.125,
            Some(&ChannelBoundary::Wall),
            Some(&ChannelBoundary::Wall),
            &inputs,
        )?;
        let accepted = match reach.trial_coupled(2.125, &fresh, None, None)? {
            ChannelCouplingTrial::Accepted(accepted) => *accepted,
            ChannelCouplingTrial::Shorter { .. } => {
                return Err("unexpected shortening of fresh interval".into());
            }
        };
        assert_eq!(accepted.time().to_bits(), 2.125_f64.to_bits());
        assert!(
            reach
                .coupling_accounting()
                .lateral_interval_volume_m3
                .is_none()
        );
        assert_eq!(
            accepted
                .coupling_accounting()
                .lateral_interval_volume_m3
                .ok_or("missing original interval volume")?
                .try_to_vec::<f64>()?,
            vec![0.0]
        );
        assert_eq!(reach.snapshot_data()?, original);
        Ok(())
    }

    #[test]
    fn endpoint_ownership_requires_one_policy_per_endpoint()
    -> Result<(), Box<dyn std::error::Error>> {
        let reach = interval_reach()?;
        let original = reach.snapshot_data()?;
        let inputs = [zero_source(crate::forcing::WaterAmount::Rate {
            rate_m3_per_s: 0.0,
        })];
        let missing =
            reach.bind_coupling_host(2.0, 2.125, Some(&ChannelBoundary::Wall), None, &inputs)?;
        assert!(matches!(
            reach.trial_coupled(2.125, &missing, None, None),
            Err(ChannelError::InvalidInput { .. })
        ));
        let external = reach.bind_coupling_host(
            2.0,
            2.125,
            Some(&ChannelBoundary::Wall),
            Some(&ChannelBoundary::Wall),
            &inputs,
        )?;
        let client = &reach.context.client;
        let elevation = Tensor::from_slice(&[1.0], &[1], client.device())?;
        let loss = Tensor::from_slice(&[0.0], &[1], client.device())?;
        assert!(matches!(
            reach.trial_coupled(
                2.125,
                &external,
                None,
                Some(ChannelReservoirOperands {
                    elevation_m: &elevation,
                    entrance_loss_coefficient: &loss
                }),
            ),
            Err(ChannelError::InvalidInput { .. })
        ));
        assert_eq!(reach.snapshot_data()?, original);
        Ok(())
    }
}
