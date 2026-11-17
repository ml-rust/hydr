//! Surveyed two-part state updates through transport, lateral exchange, and friction.

use super::{
    ChannelError,
    conserved_point::ConservedPoint,
    conserved_update::ExpansionClient,
    context::Context,
    engine::Accepted,
    sources::{PreparedLateral, SourceApplication},
    surveyed_transport::friction::SurveyedFriction,
    transport::Transport,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + ExpansionClient<R>,
{
    /// Apply surveyed transfers and lateral exchange to both state parts as expansion updates.
    pub fn surveyed_application(
        &self,
        accepted: &Accepted<R>,
        transport: &Transport<R>,
        prepared: &PreparedLateral<R>,
    ) -> Result<SourceApplication<R>, ChannelError> {
        let c = &self.client;
        if !prepared.has_nonzero_values {
            return self.lateral.apply_transport_expanded(
                c,
                &accepted.volume,
                &accepted.momentum,
                &transport.water_delta,
                &transport.momentum_delta,
            );
        }
        self.lateral.apply_lateral_expanded(
            c,
            &accepted.volume,
            &accepted.momentum,
            &transport.water_delta,
            &transport.momentum_delta,
            prepared,
        )
    }

    /// Apply Manning friction to both momentum parts through one expansion division.
    pub fn surveyed_friction_point(
        &self,
        volume: &Tensor<R>,
        momentum: &ConservedPoint<R>,
        dt: f64,
    ) -> Result<SurveyedFriction<R>, ChannelError> {
        let drag = self.surveyed_drag(volume, &momentum.high, dt)?;
        let Some(divisor) = drag.divisor else {
            let zero = self.constant(&momentum.high, 0.0)?;
            return Ok(SurveyedFriction {
                point: momentum.clone(),
                impulse: ConservedPoint {
                    high: zero.clone(),
                    low: zero.clone(),
                },
                radius: zero,
                geometry: drag.trial.geometry,
            });
        };
        Ok(SurveyedFriction {
            geometry: drag.trial.geometry,
            ..self.surveyed_friction_division(momentum, &divisor)?
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::{
        ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelGeometry,
        ChannelInitialState, ChannelReach, ChannelReachDefinition, ChannelSnapshotData,
        ChannelSurveyedConfig, SurveyedChannelCell,
        accounting::Impulses,
        cfl_speeds::FaceSpeeds,
        conserved_update::Parts,
        flux::{Flux, SurveyedRates},
        surveyed_residuals::ExpandedImpulses,
    };
    use super::*;
    use crate::channel::{CrossSectionDefinition, CrossSectionPoint};
    use numr::ops::{HypotOps, MidpointOps};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use solvr::integrate::CertifiedSimpsonAlgorithms;
    use solvr::optimize::CertifiedMappedBisectionAlgorithms;
    use solvr::optimize::roots::{
        BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
    };

    pub(in crate::channel::flow) type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    pub(in crate::channel::flow) trait Client<R: Runtime<DType = DType>>:
        ExpansionClient<R>
        + HypotOps<R>
        + MidpointOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>
    {
    }

    impl<R, C> Client<R> for C
    where
        R: Runtime<DType = DType>,
        C: ExpansionClient<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedMappedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>
            + BatchedBisectionAlgorithms<R>,
    {
    }

    pub(in crate::channel::flow) fn cpu() -> CpuClient {
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
    }

    /// Width-2 rectangular survey. Cells of length 5 hold 35 m³ at the cap stage.
    fn reach<R>(client: R::Client, volumes: &[f64]) -> TestResult<ChannelReach<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        reach_with_roughness(client, volumes, &vec![0.0; volumes.len()])
    }

    /// Same survey with one Manning roughness per cell.
    pub(in crate::channel::flow) fn reach_with_roughness<R>(
        client: R::Client,
        volumes: &[f64],
        roughness: &[f64],
    ) -> TestResult<ChannelReach<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        moving_reach(client, volumes, &vec![0.0; volumes.len()], roughness)
    }

    /// Same survey with one momentum and one Manning roughness per cell.
    pub(in crate::channel::flow) fn moving_reach<R>(
        client: R::Client,
        volumes: &[f64],
        momenta: &[f64],
        roughness: &[f64],
    ) -> TestResult<ChannelReach<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        Ok(ChannelReach::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "Synthetic SI datum".into(),
                provenance: "Surveyed two-part transport".into(),
                geometry: ChannelGeometry::PrismaticSurveyed {
                    profile: CrossSectionDefinition {
                        id: "section".into(),
                        vertical_datum: "Synthetic SI datum".into(),
                        points: [(-1.0, 3.5), (-1.0, 0.0), (1.0, 0.0), (1.0, 3.5)]
                            .into_iter()
                            .map(|(station_m, elevation_m)| CrossSectionPoint {
                                station_m,
                                elevation_m,
                            })
                            .collect(),
                        maximum_stage_m: 3.5,
                        profile_provenance: "Two-metre rectangular survey".into(),
                    },
                    cells: roughness
                        .iter()
                        .enumerate()
                        .map(|(index, manning_n)| SurveyedChannelCell {
                            id: format!("cell-{index}"),
                            length_m: 5.0,
                            manning_n: *manning_n,
                        })
                        .collect(),
                    numerical: ChannelSurveyedConfig::default(),
                },
                lateral_bindings: Vec::new(),
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: volumes.to_vec(),
                longitudinal_momentum_m4_per_s: momenta.to_vec(),
            },
        )?)
    }

    pub(in crate::channel::flow) fn walls(start_s: f64, end_s: f64) -> ChannelForcingInterval {
        ChannelForcingInterval {
            start_s,
            end_s,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: Vec::new(),
        }
    }

    /// Exact host sum of every component, as a nonoverlapping expansion.
    pub(in crate::channel::flow) fn exact_sum(values: impl IntoIterator<Item = f64>) -> Vec<f64> {
        let mut expansion: Vec<f64> = Vec::new();
        for mut value in values {
            let mut next = Vec::with_capacity(expansion.len() + 1);
            for &term in &expansion {
                let sum = value + term;
                let virtual_term = sum - value;
                let error = (value - (sum - virtual_term)) + (term - virtual_term);
                if error != 0.0 {
                    next.push(error);
                }
                value = sum;
            }
            next.push(value);
            expansion = next;
        }
        expansion
    }

    fn storage_parts(data: &ChannelSnapshotData) -> Vec<f64> {
        data.volume_m3
            .iter()
            .chain(&data.volume_low_m3)
            .copied()
            .collect()
    }

    fn sub_ulp_impulse<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client, &[35.0])?;
        let context = &reach.context;
        let c = &context.client;
        let impulse = 4791380824461455.0 * 2f64.powi(-104);
        let faces = |values: &[f64]| context.tensor(values);
        let zero = faces(&[0.0, 0.0])?;
        let rate = |high: &Tensor<R>| Parts {
            high: high.clone(),
            low: zero.clone(),
            bound: zero.clone(),
        };
        let water = faces(&[0.0, impulse])?;
        let flux = Flux {
            boundary_traces: None,
            surveyed: Some(SurveyedRates {
                water: rate(&water),
                momentum: rate(&zero),
            }),
            prescribed_water: None,
            water,
            water_bound: zero.clone(),
            riemann: None,
            momentum: zero.clone(),
            left_pressure: zero.clone(),
            right_pressure: zero.clone(),
            left_reconstructed_pressure: zero.clone(),
            right_reconstructed_pressure: zero.clone(),
            momentum_bound: zero.clone(),
            left_pressure_bound: zero.clone(),
            right_pressure_bound: zero.clone(),
            left_reconstructed_pressure_bound: zero.clone(),
            right_reconstructed_pressure_bound: zero.clone(),
            speed: FaceSpeeds {
                rightward: zero.clone(),
                leftward: zero.clone(),
            },
            internal_speed: None,
            dry_fronts: None,
        };
        let transport = context.integrate_transport(&flux, 1.0)?;
        let forcing = context.lateral.bind_host(c, &walls(0.0, 1.0))?;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let prepared = context.lateral.prepare(
            c,
            &forcing.lateral,
            &reach.accepted.lateral,
            &reach.accepted.volume.high,
            &fields.velocity,
            1.0,
        )?;
        let application = context.surveyed_application(&reach.accepted, &transport, &prepared)?;
        assert_eq!(application.volume.item::<f64>()?.to_bits(), 35f64.to_bits());
        assert_eq!(
            application.volume_low.item::<f64>()?.to_bits(),
            (-impulse).to_bits()
        );
        let volume = ConservedPoint {
            high: application.volume.clone(),
            low: application.volume_low.clone(),
        };
        let momentum = ConservedPoint {
            high: application.momentum.clone(),
            low: application.momentum_low.clone(),
        };
        let lateral = context
            .lateral
            .finalize_state(&prepared, application.water_assembly_charge_m3)?;
        let friction = context.constant(&context.length, 0.0)?;
        let balance = context.balance(
            &reach.accepted,
            &volume,
            &momentum,
            Impulses {
                water: &transport.face_volume,
                momentum: &transport.face_momentum,
                left_pressure: &transport.left_pressure,
                right_pressure: &transport.right_pressure,
                friction: &friction,
                momentum_underflow_bound: &transport.momentum_underflow_bound,
                transfer_underflow_bound: &transport.transfer_underflow_bound,
                lateral: &prepared,
                application: &application,
                next_lateral: &lateral,
                expanded: Some(ExpandedImpulses {
                    face_volume_low: &transport.face_volume_low,
                    face_momentum_low: &transport.face_momentum_low,
                    friction_low: &friction,
                    selection_bound: &application.water_selection_bound,
                }),
            },
        )?;
        // The product of two represented values is exact, so both recorded radii are zero.
        let face_bound = transport.transfer_underflow_bound.try_to_vec::<f64>()?;
        assert_eq!(face_bound, vec![0.0, 0.0]);
        assert_eq!(balance.ledger.numerical_transfer_underflow_bound_m3, 0.0);
        assert_eq!(application.water_selection_bound.item::<f64>()?, 0.0);
        assert_eq!(balance.ledger.storage_selection_abs_bound_m3, 0.0);
        assert_eq!(balance.ledger.boundary_out_m3, impulse);
        assert_eq!(balance.ledger.storage_m3, 35.0 - impulse);
        assert_eq!(balance.max_local, 0.0);
        Ok(())
    }

    fn conservation<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let origin = reach::<R>(client, &[30.0, 20.0, 10.0])?;
        let initial = exact_sum(storage_parts(&origin.snapshot_data()?));
        let mut current = origin.clone();
        let mut nonzero_low = false;
        for _ in 0..3 {
            let start = current.time();
            current = current
                .step(start + 0.01, &walls(start, start + 0.01))?
                .next;
            let data = current.snapshot_data()?;
            nonzero_low |= data
                .volume_low_m3
                .iter()
                .chain(&data.longitudinal_momentum_low_m4_per_s)
                .any(|low| *low != 0.0);
            let receipt = current.last_step()?.ok_or("Missing surveyed receipt")?;
            assert_eq!(receipt.storage_selection_bound_m3.len(), 3);
            assert_eq!(receipt.face_volume_low_m3.len(), 4);
            assert_eq!(receipt.face_momentum_low_m4_per_s.len(), 4);
        }
        assert!(nonzero_low, "the fixture must exercise nonzero low parts");
        let data = current.snapshot_data()?;
        let ledger = &data.ledger;
        let change = exact_sum(
            storage_parts(&data)
                .into_iter()
                .chain(initial.iter().map(|part| -part)),
        );
        let change: f64 = change.iter().sum();
        let boundary = ledger.boundary_in_m3 - ledger.boundary_out_m3;
        let bound =
            ledger.storage_selection_abs_bound_m3 + ledger.numerical_transfer_underflow_bound_m3;
        let slack = f64::EPSILON * change.abs().max(boundary.abs());
        assert!(
            (change - boundary).abs() <= bound + slack,
            "change={change:e}, boundary={boundary:e}, bound={bound:e}"
        );
        Ok(())
    }

    fn snapshot_round_trip<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let origin = reach::<R>(client, &[30.0, 20.0, 10.0])?;
        let mut current = origin.clone();
        for _ in 0..2 {
            let start = current.time();
            current = current
                .step(start + 0.01, &walls(start, start + 0.01))?
                .next;
        }
        let saved = current.snapshot_data()?;
        assert!(
            saved
                .volume_low_m3
                .iter()
                .chain(&saved.longitudinal_momentum_low_m4_per_s)
                .any(|low| *low != 0.0)
        );
        let trusted = origin.restore(&current.snapshot())?;
        let cold = origin.import_snapshot(&saved)?;
        assert_eq!(trusted.snapshot_data()?, saved);
        assert_eq!(cold.snapshot_data()?, saved);
        let start = current.time();
        let expected = current
            .step(start + 0.01, &walls(start, start + 0.01))?
            .next
            .snapshot_data()?;
        for restored in [trusted, cold] {
            let replayed = restored
                .step(start + 0.01, &walls(start, start + 0.01))?
                .next
                .snapshot_data()?;
            assert!(replayed.same_bits(&expected));
        }
        let mut altered = saved.clone();
        let low = altered
            .last_interval
            .as_mut()
            .ok_or("Missing stored interval")?
            .diagnostics
            .face_volume_low_m3
            .iter_mut()
            .find(|low| **low != 0.0)
            .ok_or("Missing nonzero face low part")?;
        *low = f64::from_bits(low.to_bits() + 1);
        assert!(origin.import_snapshot(&altered).is_err());
        Ok(())
    }

    #[test]
    fn cpu_cap_cell_keeps_a_sub_ulp_impulse_and_records_its_bounds() -> TestResult {
        sub_ulp_impulse::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_surveyed_steps_conserve_both_parts_within_recorded_bounds() -> TestResult {
        conservation::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_snapshot_with_nonzero_lows_round_trips_and_replays() -> TestResult {
        snapshot_round_trip::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_cap_cell_keeps_a_sub_ulp_impulse_and_records_its_bounds() -> TestResult {
            sub_ulp_impulse::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_surveyed_steps_conserve_both_parts_within_recorded_bounds() -> TestResult {
            conservation::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_snapshot_with_nonzero_lows_round_trips_and_replays() -> TestResult {
            snapshot_round_trip::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }
    }
}
