use crate::channel::{
    ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
    RectangularChannelCell,
};
use crate::coupling::{
    ChannelPoolBatch, ChannelPoolConfig, ChannelPoolConnection, ChannelPoolEndpoint,
    ChannelPoolInitialState, ChannelPoolReach, ChannelPoolSetup,
};
use crate::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolState,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

#[test]
fn attachment_support_uses_certified_cut_volume() -> Result<(), Box<dyn std::error::Error>> {
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(1), Some(64)));
    let bed = 4e-17;
    let datum = "Exact-input attachment datum";
    let setup = ChannelPoolSetup {
        reaches: vec![ChannelPoolReach {
            id: "reach".into(),
            definition: ChannelReachDefinition {
                vertical_datum: datum.into(),
                provenance: "Explicit wet channel".into(),
                geometry: ChannelGeometry::Rectangular {
                    cells: vec![RectangularChannelCell {
                        id: "cell".into(),
                        length_m: 1.0,
                        width_m: 1.0,
                        bed_elevation_m: bed,
                        maximum_depth_m: 2.0,
                        manning_n: 0.0,
                    }],
                },
                lateral_bindings: Vec::new(),
                config: ChannelConfig::default(),
            },
        }],
        pools: vec![LevelPoolElement {
            id: "pool".into(),
            curve: LevelPoolCurve::new(
                vec![
                    LevelPoolKnot {
                        volume_m3: 0.0,
                        stage_m: -0.3,
                    },
                    LevelPoolKnot {
                        volume_m3: 10.0,
                        stage_m: 2.7,
                    },
                ],
                datum.into(),
                "Exact binary64 affine curve".into(),
            )?,
            connected_pool_assertion: "One connected level pool".into(),
            storage_support_identity: "Separate finite pool".into(),
        }],
        connections: vec![ChannelPoolConnection {
            id: "connection".into(),
            reach_id: "reach".into(),
            endpoint: ChannelPoolEndpoint::Right,
            pool_id: "pool".into(),
            entrance_loss_coefficient: 0.0,
            physical_connection_assertion: "Channel endpoint at its declared bed".into(),
            quiescent_reservoir_assertion: "Quiescent pool with complete exit mixing".into(),
        }],
        vertical_datum: datum.into(),
        nonoverlap_assertion: "Separate channel and pool storage".into(),
        config: ChannelPoolConfig {
            storage: LevelPoolConfig::default(),
            head_absolute_m: 1e-12,
            head_relative: 1e-12,
            transfer_absolute_m3: 1e-12,
            transfer_relative: 1e-12,
            mass_absolute_m3: 1e-9,
            mass_relative: 1e-12,
            minimum_step_s: 1e-10,
            maximum_step_s: 1.0,
            iteration_limit: 64,
            retry_limit: 16,
        },
    };
    let batch = ChannelPoolBatch::<CpuRuntime>::new(
        client,
        setup,
        ChannelPoolInitialState {
            time_s: 0.0,
            reaches: vec![ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![1.0],
                longitudinal_momentum_m4_per_s: vec![0.0],
            }],
            pools: vec![LevelPoolState { volume_m3: 1.0 }],
        },
    )?;
    // Exact binary64 inputs give stage 2.7755575615628914e-17, below the attachment.
    assert!(batch.pools().stages_tensor()?.try_to_vec::<f64>()?[0] > bed);
    let wet = batch
        .context
        .attachment_wet(batch.pools())?
        .try_to_vec::<f64>()?;
    assert_eq!(wet, vec![0.0]);
    Ok(())
}
