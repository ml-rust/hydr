use super::context_types::Context;
use super::state::{Accepted, ParentCounters};
use super::{ChannelPoolError, ChannelPoolInitialState, ChannelPoolSetup};
use crate::channel::ChannelReach;
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{AndersonAlgorithms, BatchedBisectionAlgorithms};
use std::sync::Arc;

/// Accepted channel and pool states sharing one runtime and common interval history.
#[derive(Clone)]
pub struct ChannelPoolBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) accepted: Accepted<R>,
}
impl<R: Runtime<DType = DType>> std::fmt::Debug for ChannelPoolBatch<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelPoolBatch")
            .field("setup", &self.context.setup)
            .field("time", &self.accepted.pools.time())
            .field("accepted_steps", &self.accepted.pools.accepted_steps())
            .finish()
    }
}
impl<R: Runtime<DType = DType>> ChannelPoolBatch<R> {
    /// Immutable definitions, connection evidence, and numerical controls.
    pub fn setup(&self) -> &ChannelPoolSetup {
        &self.context.setup
    }
    /// Common accepted time, in seconds.
    pub fn time(&self) -> f64 {
        self.accepted.pools.time()
    }
    /// Number of complete accepted common intervals.
    pub fn accepted_steps(&self) -> usize {
        self.accepted.pools.accepted_steps()
    }
    /// Read-only native reach selected by its declared identity.
    pub fn reach(&self, id: &str) -> Option<&ChannelReach<R>> {
        self.context
            .topology
            .reach_by_id
            .get(id)
            .and_then(|&i| self.accepted.reaches.get(i))
    }
    /// Read-only finite receiving and donating storage.
    pub fn pools(&self) -> &LevelPoolBatch<R> {
        &self.accepted.pools
    }
}
impl<R: Runtime<DType = DType>> ChannelPoolBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>
        + BatchedBisectionAlgorithms<R>
        + AndersonAlgorithms<R>,
{
    /// Construct all children on the supplied client with explicit initial storage and momentum.
    pub fn new(
        client: R::Client,
        setup: ChannelPoolSetup,
        initial: ChannelPoolInitialState,
    ) -> Result<Self, ChannelPoolError>
    where
        R::Client:
            numr::ops::HypotOps<R> + numr::ops::TensorOps<R> + numr::ops::NextRepresentableOps<R>,
    {
        setup.config.check()?;
        if !initial.time_s.is_finite()
            || initial.time_s < 0.0
            || initial.reaches.len() != setup.reaches.len()
            || initial.pools.len() != setup.pools.len()
            || initial
                .reaches
                .iter()
                .any(|r| r.time_s.to_bits() != initial.time_s.to_bits())
        {
            return Err(ChannelPoolError::input(
                "initial child counts or epoch differ: supply complete states at one finite nonnegative exact time",
            ));
        }
        let reaches = setup
            .reaches
            .iter()
            .zip(&initial.reaches)
            .map(|(definition, state)| {
                ChannelReach::<R>::new(client.clone(), definition.definition.clone(), state.clone())
                    .map_err(|source| ChannelPoolError::Channel {
                        reach_id: definition.id.clone(),
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pools = LevelPoolBatch::<R>::new(
            client.clone(),
            setup.pools.clone(),
            initial.pools.clone(),
            initial.time_s,
            setup.config.storage.clone(),
        )?;
        let context = Arc::new(Context::new(client, setup, initial, &reaches, &pools)?);
        let counters = ParentCounters::zeros(
            &context.client,
            context.setup.connections.len(),
            context.topology.external_endpoints.len(),
            context.setup.pools.len(),
        )?;
        let result = Self {
            context,
            accepted: Accepted {
                reaches,
                pools,
                counters,
                last: None,
            },
        };
        result
            .context
            .check_epochs(&result.accepted.reaches, &result.accepted.pools)?;
        result.context.water_balance(&result.accepted)?;
        Ok(result)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use crate::channel::{
        ChannelBoundary, ChannelConfig, ChannelGeometry, ChannelInitialState,
        ChannelReachDefinition, ChannelSurveyedConfig, CrossSectionDefinition, CrossSectionPoint,
        SurveyedChannelCell,
    };
    use crate::coupling::{
        ChannelPoolBatch, ChannelPoolConfig, ChannelPoolConnection, ChannelPoolEndpoint,
        ChannelPoolForcingInterval, ChannelPoolInitialState, ChannelPoolReach,
        ChannelPoolReachForcing, ChannelPoolSetup,
    };
    use crate::storage::{
        LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolState,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    pub(in super::super) type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;
    pub(in super::super) type Batch = ChannelPoolBatch<CpuRuntime>;

    const DATUM: &str = "Synthetic surveyed channel/pool datum";
    /// Absolute mass allowance of the coordinator, in cubic metres.
    pub(in super::super) const MASS_ABSOLUTE_M3: f64 = 1e-9;
    /// Relative mass allowance of the coordinator.
    pub(in super::super) const MASS_RELATIVE: f64 = 1e-12;
    /// The frozen `three-bands` survey: knots at depths 1 and 2, cap 3, bed at 0.
    const THREE_BANDS: [(f64, f64); 8] = [
        (-4.0, 3.0),
        (-2.5, 2.0),
        (-2.0, 1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (2.5, 2.0),
        (4.0, 3.0),
    ];

    pub(in super::super) fn client() -> CpuClient {
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), Some(64)))
    }

    /// One frictionless three-band reach of `count` cells of `length_m`, at gravity 1.
    fn definition(count: usize, length_m: f64) -> ChannelReachDefinition {
        ChannelReachDefinition {
            vertical_datum: DATUM.into(),
            provenance: "Explicit surveyed finite-pool reach".into(),
            geometry: ChannelGeometry::PrismaticSurveyed {
                profile: CrossSectionDefinition {
                    id: "three-bands".into(),
                    vertical_datum: DATUM.into(),
                    points: THREE_BANDS
                        .into_iter()
                        .map(|(station_m, elevation_m)| CrossSectionPoint {
                            station_m,
                            elevation_m,
                        })
                        .collect(),
                    maximum_stage_m: 3.0,
                    profile_provenance: "Frozen three-band acceptance survey".into(),
                },
                cells: (0..count)
                    .map(|i| SurveyedChannelCell {
                        id: format!("cell-{i}"),
                        length_m,
                        manning_n: 0.0,
                    })
                    .collect(),
                numerical: ChannelSurveyedConfig::default(),
            },
            lateral_bindings: Vec::new(),
            config: ChannelConfig {
                gravity: 1.0,
                max_step: 1.0,
                ..ChannelConfig::default()
            },
        }
    }

    /// One three-band reach whose `endpoint` connects to one constant-area pool.
    ///
    /// Each cell holds `cell_volume_m3` at rest. The pool has plan area `pool_area_m2`,
    /// its bed at 0, and its stage at `pool_stage_m`.
    pub(in super::super) fn single(
        count: usize,
        length_m: f64,
        cell_volume_m3: f64,
        endpoint: ChannelPoolEndpoint,
        pool_area_m2: f64,
        pool_stage_m: f64,
        loss: f64,
    ) -> TestResult<(ChannelPoolSetup, ChannelPoolInitialState)> {
        let curve = LevelPoolCurve::new(
            vec![
                LevelPoolKnot {
                    volume_m3: 0.0,
                    stage_m: 0.0,
                },
                LevelPoolKnot {
                    volume_m3: pool_area_m2 * 4.0,
                    stage_m: 4.0,
                },
            ],
            DATUM.into(),
            "Explicit constant-area analytical basin".into(),
        )?;
        let setup = ChannelPoolSetup {
            reaches: vec![ChannelPoolReach {
                id: "reach".into(),
                definition: definition(count, length_m),
            }],
            pools: vec![LevelPoolElement {
                id: "pool".into(),
                curve,
                connected_pool_assertion: "Connected approximately level synthetic basin".into(),
                storage_support_identity: "finite-basin-pool".into(),
            }],
            connections: vec![ChannelPoolConnection {
                id: "connection".into(),
                reach_id: "reach".into(),
                endpoint,
                pool_id: "pool".into(),
                entrance_loss_coefficient: loss,
                physical_connection_assertion:
                    "Unobstructed named channel endpoint at its actual bed".into(),
                quiescent_reservoir_assertion:
                    "Negligible basin velocity with complete exit mixing".into(),
            }],
            vertical_datum: DATUM.into(),
            nonoverlap_assertion: "Basin and reach occupy separate physical storage".into(),
            config: ChannelPoolConfig {
                storage: LevelPoolConfig::default(),
                head_absolute_m: 1e-12,
                head_relative: 1e-12,
                transfer_absolute_m3: 1e-12,
                transfer_relative: 1e-12,
                mass_absolute_m3: MASS_ABSOLUTE_M3,
                mass_relative: MASS_RELATIVE,
                minimum_step_s: 1e-10,
                maximum_step_s: 1.0,
                iteration_limit: 64,
                retry_limit: 16,
            },
        };
        let initial = ChannelPoolInitialState {
            time_s: 0.0,
            reaches: vec![ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![cell_volume_m3; count],
                longitudinal_momentum_m4_per_s: vec![0.0; count],
            }],
            pools: vec![LevelPoolState {
                volume_m3: pool_area_m2 * pool_stage_m,
            }],
        };
        Ok((setup, initial))
    }

    /// The exterior state at rest at `depth_m`.
    pub(in super::super) fn exterior(depth_m: f64) -> ChannelBoundary {
        ChannelBoundary::Exterior {
            depth_m,
            velocity_m_per_s: 0.0,
        }
    }

    /// Places `far` at each unconnected end.
    pub(in super::super) fn forcing(
        setup: &ChannelPoolSetup,
        start_s: f64,
        end_s: f64,
        far: &ChannelBoundary,
    ) -> ChannelPoolForcingInterval {
        let connected = |endpoint| setup.connections.iter().any(|c| c.endpoint == endpoint);
        let external = |endpoint| (!connected(endpoint)).then(|| far.clone());
        ChannelPoolForcingInterval {
            start_s,
            end_s,
            reaches: vec![ChannelPoolReachForcing {
                reach_id: "reach".into(),
                left: external(ChannelPoolEndpoint::Left),
                right: external(ChannelPoolEndpoint::Right),
                lateral_inputs: Vec::new(),
            }],
        }
    }

    fn side(endpoint: ChannelPoolEndpoint) -> usize {
        match endpoint {
            ChannelPoolEndpoint::Left => 0,
            ChannelPoolEndpoint::Right => 1,
        }
    }

    /// `|actual - expected| <= 1e-10 + 1e-10 |expected|`.
    fn near(actual: f64, expected: f64, field: &str) {
        assert!(
            (actual - expected).abs() <= 1e-10 + 1e-10 * expected.abs(),
            "{field}: actual={actual:e}, independent reference={expected:e}"
        );
    }

    /// A three-band lake at rest at depth 1 m stays at rest against a pool at stage 1 m.
    ///
    /// A `Stage` of 1 m holds the unconnected end. Its trace pressure has the same bits as
    /// the connected trace pressure, so the cell momentum stays exactly zero. An `Exterior`
    /// face carries a two-part Riemann pressure instead. Its rounding differs from the
    /// single-part trace pressure. The resulting transfer lies below the pool volume's
    /// resolution, and the pool rejects it.
    ///
    /// Frozen criteria, each of 16 intervals of 0.125 s, at both ends, with entrance
    /// losses `k = 0` and `k = 1`:
    /// - cell depth within 1e-12 m of 1 m
    /// - |Q| and |momentum| at most 1e-11
    /// - composite storage change at most 1e-11 m³
    /// - connection transfer at most 1e-11 m³
    #[test]
    fn surveyed_lake_at_rest_stays_at_rest_against_a_pool() -> TestResult {
        // The three-band section holds 3 m² at depth 1 m.
        let volume = 128.0 * 3.0;
        let held = ChannelBoundary::Stage { elevation_m: 1.0 };
        for endpoint in [ChannelPoolEndpoint::Left, ChannelPoolEndpoint::Right] {
            for loss in [0.0, 1.0] {
                let (setup, initial) = single(1, 128.0, volume, endpoint, 1.0, 1.0, loss)?;
                let origin = Batch::new(client(), setup.clone(), initial)?;
                let before = origin.snapshot_data()?;
                let storage = origin.ledger()?.storage_m3;
                let mut current = origin.clone();
                for index in 0..16 {
                    let end = (index + 1) as f64 / 8.0;
                    current = current
                        .trial_to(end, &forcing(&setup, current.time(), end, &held))?
                        .into_next();
                    assert_eq!(current.time(), end);
                    let state = current.reach("reach").ok_or("Missing reach")?.state()?;
                    assert!((state[0].depth_m - 1.0).abs() <= 1e-12, "{state:?}");
                    assert!(state[0].discharge_m3_per_s.abs() <= 1e-11);
                    assert!(state[0].longitudinal_momentum_m4_per_s.abs() <= 1e-11);
                    assert!((current.ledger()?.storage_m3 - storage).abs() <= 1e-11);
                    let step = current.last_step()?.ok_or("Missing rest receipt")?;
                    assert!(step.transfers[0].signed_volume_m3.abs() <= 1e-11);
                }
                assert_eq!(origin.snapshot_data()?, before);
            }
        }
        Ok(())
    }

    /// The frozen `three-bands-leaving` reference with pool area 1 m².
    ///
    /// One 128 m cell at rest at depth 1.5 m (5.125 m²), gravity 1, one 0.125 s
    /// interval, pool stage 0.875 m, `k = 0`. The opposite end holds the interior state at
    /// rest. Values are the nearest binary64 of the 150-digit references in
    /// `tests/channel_pools/surveyed_reference.rs`.
    ///
    /// Frozen criteria, at both ends:
    /// - trace depth, velocity, discharge and momentum flux, transfer, midpoint head, pool
    ///   volume, reach volume, and reach momentum within 1e-10 plus 1e-10 relative
    /// - connection volume equal to the reach end face volume bit for bit
    /// - pool incoming and outgoing volumes equal to the connection volume bit for bit
    #[test]
    fn surveyed_three_band_outflow_matches_the_frozen_reference() -> TestResult {
        const DEPTH: f64 = 0.9739784214397981;
        const VELOCITY: f64 = 0.5467305697765322;
        const DISCHARGE: f64 = 1.583654743036769;
        const TRANSFER: f64 = 0.1979568428795961;
        const HEAD: f64 = 0.9739784214397981;
        const POOL_VOLUME: f64 = 1.0729568428795961;
        const CHANNEL_VOLUME: f64 = 655.8020431571204;
        const RIGHT_MOMENTUM: f64 = 0.15396465464856154;
        const MOMENTUM_FLUX: f64 = 2.1224494294781744;
        for endpoint in [ChannelPoolEndpoint::Left, ChannelPoolEndpoint::Right] {
            let (setup, initial) = single(1, 128.0, 128.0 * 5.125, endpoint, 1.0, 0.875, 0.0)?;
            let origin = Batch::new(client(), setup.clone(), initial)?;
            let next = origin
                .trial_to(0.125, &forcing(&setup, 0.0, 0.125, &exterior(1.5)))?
                .into_next();
            assert_eq!(next.time().to_bits(), 0.125_f64.to_bits());
            let receipt = next.last_step()?.ok_or("Missing finite-pool receipt")?;
            let child = next.reach("reach").ok_or("Missing reach")?;
            let child_receipt = child.last_step()?.ok_or("Missing channel receipt")?;
            let trace = child_receipt.boundary_traces[side(endpoint)]
                .as_ref()
                .ok_or("Missing reservoir trace")?;
            assert!(trace.reservoir.is_some());
            near(trace.depth_m, DEPTH, "depth");
            near(trace.outward_velocity_m_per_s, VELOCITY, "velocity");
            near(trace.outward_discharge_m3_per_s, DISCHARGE, "discharge");
            near(
                trace.momentum_flux_m4_per_s2,
                MOMENTUM_FLUX,
                "momentum flux",
            );
            let transfer = &receipt.transfers[0];
            near(transfer.signed_volume_m3, TRANSFER, "transfer");
            near(transfer.midpoint_pool_elevation_m, HEAD, "head");
            near(
                next.pools().states()?[0].volume_m3,
                POOL_VOLUME,
                "pool volume",
            );
            let state = child.state()?;
            near(state[0].volume_m3, CHANNEL_VOLUME, "reach volume");
            let normal = if endpoint == ChannelPoolEndpoint::Left {
                -1.0
            } else {
                1.0
            };
            near(
                state[0].longitudinal_momentum_m4_per_s,
                normal * RIGHT_MOMENTUM,
                "reach momentum",
            );
            assert_eq!(
                transfer.signed_volume_m3.to_bits(),
                (normal * child_receipt.face_volume_m3[side(endpoint)]).to_bits()
            );
            let pool_receipt = next.pools().last_step()?.ok_or("Missing pool receipt")?;
            let paired = &pool_receipt.receipts[0];
            assert_eq!(
                paired.incoming_volume_m3.to_bits(),
                transfer.signed_volume_m3.max(0.0).to_bits()
            );
            assert_eq!(
                paired.outgoing_volume_m3.to_bits(),
                (-transfer.signed_volume_m3).max(0.0).to_bits()
            );
        }
        Ok(())
    }
}
