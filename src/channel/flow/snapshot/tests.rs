use crate::channel::{
    ChannelBoundary, ChannelBoundaryInterval, ChannelConfig, ChannelError, ChannelInitialState,
    ChannelReach, ChannelReachDefinition, RectangularChannelCell,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use std::sync::Arc;

#[test]
fn restoration_retains_the_destination_configured_client() -> Result<(), ChannelError> {
    let client = CpuClient::new(CpuDevice::new());
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic datum".into(),
        provenance: "exact rectangular client fixture".into(),
        cells: vec![RectangularChannelCell {
            id: "cell".into(),
            length_m: 1.0,
            width_m: 2.0,
            bed_elevation_m: 0.0,
            maximum_depth_m: 4.0,
            manning_n: 0.0,
        }],
        config: ChannelConfig::default(),
    };
    let initial = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![2.0],
        longitudinal_momentum_m4_per_s: vec![0.0],
    };
    let destination = ChannelReach::<CpuRuntime>::new(
        client
            .clone()
            .with_parallelism(ParallelismConfig::new(Some(1), Some(64))),
        definition.clone(),
        initial.clone(),
    )?;
    let source = ChannelReach::<CpuRuntime>::new(
        client.with_parallelism(ParallelismConfig::new(Some(1), Some(128))),
        definition,
        initial,
    )?
    .step(
        0.01,
        &ChannelBoundaryInterval {
            start_s: 0.0,
            end_s: 1.0,
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
        },
    )?
    .next;
    let context = Arc::clone(&destination.context);
    let trusted = destination.restore(&source.snapshot())?;
    let cold = destination.import_snapshot(&source.snapshot_data()?)?;
    for restored in [&trusted, &cold] {
        assert!(Arc::ptr_eq(&restored.context, &context));
        assert_eq!(restored.context.client.parallelism().chunk_size, Some(64));
        assert_eq!(restored.snapshot_data()?, source.snapshot_data()?);
    }
    Ok(())
}

#[test]
fn cold_restore_uploads_with_a_separately_constructed_cpu_client() -> Result<(), ChannelError> {
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic datum".into(),
        provenance: "exact rectangular device fixture".into(),
        cells: vec![RectangularChannelCell {
            id: "cell".into(),
            length_m: 1.0,
            width_m: 2.0,
            bed_elevation_m: 0.0,
            maximum_depth_m: 4.0,
            manning_n: 0.0,
        }],
        config: ChannelConfig::default(),
    };
    let initial = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![2.0],
        longitudinal_momentum_m4_per_s: vec![0.0],
    };
    let source = ChannelReach::<CpuRuntime>::new(
        CpuClient::new(CpuDevice::new()),
        definition.clone(),
        initial.clone(),
    )?;
    let destination =
        ChannelReach::<CpuRuntime>::new(CpuClient::new(CpuDevice::new()), definition, initial)?;
    let context = Arc::clone(&destination.context);
    let restored = destination.import_snapshot(&source.snapshot_data()?)?;
    assert!(Arc::ptr_eq(&restored.context, &context));
    assert_eq!(restored.snapshot_data()?, source.snapshot_data()?);
    Ok(())
}
