use crate::storage::{
    LevelPoolBatch, LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolError,
    LevelPoolKnot, LevelPoolSourcePolicy, LevelPoolState,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub(super) fn batch(source_enabled: bool) -> Result<LevelPoolBatch<CpuRuntime>, LevelPoolError> {
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(1), Some(64)));
    let elements = [0.0, -2.0]
        .into_iter()
        .enumerate()
        .map(|(index, origin)| {
            Ok(LevelPoolElement {
                id: format!("pool-{index}"),
                curve: LevelPoolCurve::new(
                    vec![
                        LevelPoolKnot {
                            volume_m3: 0.0,
                            stage_m: origin,
                        },
                        LevelPoolKnot {
                            volume_m3: 4.0,
                            stage_m: origin + 4.0,
                        },
                    ],
                    "analytical datum".into(),
                    "constant area one square metre".into(),
                )?,
                connected_pool_assertion: "one level water surface per element".into(),
                storage_support_identity: format!("disjoint-support-{index}"),
            })
        })
        .collect::<Result<Vec<_>, LevelPoolError>>()?;
    let states = vec![
        LevelPoolState { volume_m3: 2.0 },
        LevelPoolState { volume_m3: 0.0 },
    ];
    if source_enabled {
        LevelPoolBatch::new_with_sources(
            client,
            elements,
            states,
            0.0,
            LevelPoolConfig::default(),
            vec![
                LevelPoolSourcePolicy {
                    dry_basal_footprint_assertion: "first segment represents exposed unit area"
                        .into(),
                    bottom_loss_provenance: "explicit analytical potential demand".into(),
                };
                2
            ],
        )
    } else {
        LevelPoolBatch::new(client, elements, states, 0.0, LevelPoolConfig::default())
    }
}
