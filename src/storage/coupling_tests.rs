use super::*;
use crate::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolInput, LevelPoolKnot,
    LevelPoolSourcePolicy, LevelPoolSources, LevelPoolState,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use std::sync::Arc;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn batch(source_enabled: bool) -> Result<LevelPoolBatch<CpuRuntime>, LevelPoolError> {
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

#[test]
fn tensor_trial_matches_host_trial_and_preserves_configured_context() -> TestResult {
    let batch = batch(false)?;
    let before = batch.snapshot_data()?;
    let context = Arc::clone(&batch.context);
    let incoming = batch.context.tensor(&[0.5, 1.0])?;
    let outgoing = batch.context.tensor(&[0.25, 0.0])?;
    let host = batch
        .trial_to(
            1.0,
            &[
                LevelPoolInput {
                    incoming_volume_m3: 0.5,
                    outgoing_volume_m3: 0.25,
                },
                LevelPoolInput {
                    incoming_volume_m3: 1.0,
                    outgoing_volume_m3: 0.0,
                },
            ],
        )?
        .into_next();
    let device = batch
        .trial_to_tensors(1.0, &incoming, &outgoing)?
        .into_next();
    assert_eq!(device.snapshot_data()?, host.snapshot_data()?);
    assert_eq!(batch.snapshot_data()?, before);
    assert!(Arc::ptr_eq(&device.context, &context));
    assert_eq!(device.context.client.parallelism().chunk_size, Some(64));
    assert_eq!(device.accepted_steps(), 1);
    assert_eq!(
        device.stages_tensor()?.try_to_vec::<f64>()?,
        vec![2.25, -1.0]
    );
    Ok(())
}

#[test]
fn tensor_access_keeps_dry_origin_distinct_from_physical_stage() -> TestResult {
    let batch = batch(false)?;
    assert_eq!(batch.volumes_tensor().try_to_vec::<f64>()?, vec![2.0, 0.0]);
    assert_eq!(batch.stages_tensor()?.try_to_vec::<f64>()?, vec![2.0, -2.0]);
    assert_eq!(batch.outputs()?[0].stage_m, Some(2.0));
    assert_eq!(batch.outputs()?[1].stage_m, None);
    Ok(())
}

#[test]
fn tensor_metadata_errors_preserve_complete_accepted_state() -> TestResult {
    let batch = batch(false)?;
    let before = batch.snapshot_data()?;
    let zero = batch.context.tensor(&[0.0, 0.0])?;
    let device = batch.context.client.device();
    let invalid = [
        Tensor::<CpuRuntime>::zeros(&[], DType::F64, device)?,
        Tensor::<CpuRuntime>::zeros(&[1], DType::F64, device)?,
        Tensor::<CpuRuntime>::zeros(&[1, 2], DType::F64, device)?,
        Tensor::<CpuRuntime>::zeros(&[3], DType::F64, device)?,
        Tensor::<CpuRuntime>::zeros(&[2], DType::F32, device)?,
    ];
    for value in invalid {
        assert!(matches!(
            batch.trial_to_tensors(1.0, &value, &zero),
            Err(LevelPoolError::InvalidInput { .. })
        ));
        assert!(matches!(
            batch.trial_to_tensors(1.0, &zero, &value),
            Err(LevelPoolError::InvalidInput { .. })
        ));
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn tensor_physical_errors_preserve_complete_accepted_state() -> TestResult {
    let batch = batch(false)?;
    let before = batch.snapshot_data()?;
    let zero = batch.context.tensor(&[0.0, 0.0])?;
    for values in [[-0.1, 0.0], [f64::NAN, 0.0], [f64::INFINITY, 0.0]] {
        let input = batch.context.tensor(&values)?;
        assert!(batch.trial_to_tensors(1.0, &input, &zero).is_err());
        assert!(batch.trial_to_tensors(1.0, &zero, &input).is_err());
        assert_eq!(batch.snapshot_data()?, before);
    }
    let unavailable = batch.context.tensor(&[3.0, 0.0])?;
    assert!(batch.trial_to_tensors(1.0, &zero, &unavailable).is_err());
    for end in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            batch.trial_to_tensors(end, &zero, &zero),
            Err(LevelPoolError::InvalidInput { .. })
        ));
    }
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn closed_tensor_trial_rejects_source_enabled_state_without_changing_public_source_trials()
-> TestResult {
    let batch = batch(true)?;
    let before = batch.snapshot_data()?;
    let zero = batch.context.tensor(&[0.0, 0.0])?;
    assert!(matches!(
        batch.trial_to_tensors(1.0, &zero, &zero),
        Err(LevelPoolError::InvalidInput { .. })
    ));
    assert_eq!(batch.snapshot_data()?, before);
    let inputs = [LevelPoolInput {
        incoming_volume_m3: 0.0,
        outgoing_volume_m3: 0.0,
    }; 2];
    let exchange_only = batch.trial_to(1.0, &inputs)?.into_next();
    assert_eq!(exchange_only.states()?, batch.states()?);
    let source_trial = batch
        .trial_with_sources_to(
            1.0,
            &inputs,
            &[LevelPoolSources {
                rain_rate_m_per_s: 0.125,
                potential_evaporation_rate_m_per_s: 0.0,
                bottom_loss_rate_m3_per_s: 0.0,
            }; 2],
        )?
        .into_next();
    assert_eq!(
        source_trial.states()?,
        vec![
            LevelPoolState { volume_m3: 2.125 },
            LevelPoolState { volume_m3: 0.125 }
        ]
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}
