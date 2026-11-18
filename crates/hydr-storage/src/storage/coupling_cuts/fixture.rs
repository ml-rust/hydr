use crate::storage::{
    LevelPoolBatch, LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolError,
    LevelPoolKnot, LevelPoolState,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub(in crate::storage) fn batch(
    volumes: &[f64],
    stages: &[f64],
    current: f64,
) -> Result<LevelPoolBatch<CpuRuntime>, LevelPoolError> {
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(1), Some(64)));
    let curve = LevelPoolCurve::new(
        volumes
            .iter()
            .zip(stages)
            .map(|(&volume_m3, &stage_m)| LevelPoolKnot { volume_m3, stage_m })
            .collect(),
        "analytical datum".into(),
        "exact affine storage segments".into(),
    )?;
    LevelPoolBatch::new(
        client,
        vec![LevelPoolElement {
            id: "pool".into(),
            curve,
            connected_pool_assertion: "one connected level surface".into(),
            storage_support_identity: "finite original storage".into(),
        }],
        vec![LevelPoolState { volume_m3: current }],
        0.0,
        LevelPoolConfig::default(),
    )
}
