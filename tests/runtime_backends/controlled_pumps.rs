use super::control_fixture::{Relay, WIDTHS, device, pumps, samples};
use super::{Result, runtime};
use hydr::structures::*;

fn relay() -> Relay {
    Relay {
        low: ControlCommand::PumpStoppedIsolated,
        high: ControlCommand::PumpRunning { speed_ratio: 0.5 },
        times: vec![1.0, 2.0, 3.0],
        end: 4.0,
        maximum_gap: 4.0,
        lateness: 4.0,
    }
}

fn responses(width: usize) -> Vec<PumpResponseDefinition> {
    (0..width)
        .map(|lane| PumpResponseDefinition {
            structure_id: device(lane),
            model: PumpResponseModel::IdealInstantaneous,
            response_provenance: "Explicit analytical instantaneous operation".into(),
            ideal_response_applicability: "No startup delay, acceleration, or coast-down".into(),
        })
        .collect()
}

macro_rules! controlled_pump_execution {
    ($runtime:ty, $client:expr) => {{
        let client = $client;
        let relay = relay();
        let running = StructureOperatingPoint::PumpRunning { speed_ratio: 0.5 };
        let stopped = StructureOperatingPoint::PumpStoppedIsolated;
        let mut finals = Vec::new();
        for width in WIDTHS {
            let build = || -> Result<ControlledPumpBatch<$runtime>> {
                let (definitions, initial, schedule, timing) = relay.parts(width);
                let structures = StructureBatch::<$runtime>::new(client.clone(), pumps(width)?)?;
                Ok(ControlledPumpBatch::<$runtime>::new(
                    structures.with_operating_controls(definitions, initial, schedule, timing)?,
                    responses(width),
                )?)
            };
            let step = |batch: &ControlledPumpBatch<$runtime>,
                        time: f64,
                        stage: Option<f64>,
                        operating_point: StructureOperatingPoint|
             -> Result<ControlledPumpBatch<$runtime>> {
                let packets = stage.map_or_else(Vec::new, |stage| samples(width, time, stage));
                let before = batch.snapshot_data()?;
                drop(batch.trial_to(time, &packets)?);
                assert_eq!(batch.snapshot_data()?, before);
                let next = batch.trial_to(time, &packets)?.into_next();
                let outputs = next.outputs()?;
                assert_eq!(outputs.len(), width);
                for output in outputs {
                    assert_eq!(output.operating_point, operating_point);
                    assert_eq!(output.model, PumpResponseModel::IdealInstantaneous);
                    assert_eq!(output.time_s.to_bits(), next.time().to_bits());
                }
                assert_eq!(next.operations()?.len(), width);
                Ok(next)
            };
            let resume = |mut batch: ControlledPumpBatch<$runtime>| -> Result<_> {
                batch = step(&batch, 2.0, Some(1.5), running)?;
                batch = step(&batch, 3.0, Some(1.0), stopped)?;
                batch = step(&batch, 4.0, None, stopped)?;
                assert!(batch.next_event().is_none());
                for state in batch.controls().states()? {
                    assert_eq!(state.transition_count, 2);
                }
                Ok(batch)
            };
            let initial = build()?;
            for output in initial.outputs()? {
                assert_eq!(output.operating_point, stopped);
                assert_eq!(output.accepted_steps, 0);
            }
            let mut batch = step(&initial, 1.0, Some(2.0), running)?;
            let trusted = batch.snapshot();
            let owned = trusted.data()?;
            batch = resume(batch)?;
            let expected = batch.snapshot_data()?;
            batch.restore(&trusted)?;
            assert_eq!(batch.snapshot_data()?, owned);
            assert_eq!(resume(batch)?.snapshot_data()?, expected);
            let mut cold = build()?;
            cold.restore_data(&owned)?;
            assert_eq!(cold.snapshot_data()?, owned);
            assert_eq!(resume(cold)?.snapshot_data()?, expected);
            finals.push(expected);
        }
        finals
    }};
}

#[test]
fn selected_cpu_client_executes_controlled_pump_operations_and_complete_restart() -> Result {
    controlled_pump_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_controlled_pump_operations_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = controlled_pump_execution!(runtime::Runtime, runtime::client());
    let cuda = controlled_pump_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    super::same_bits(&cpu, &cuda)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_controlled_pump_f64_without_fallback() -> Result {
    use numr::runtime::wgpu::WgpuRuntime;
    let (definitions, initial, schedule, timing) = relay().parts(1);
    let built = StructureBatch::<WgpuRuntime>::new(super::wgpu_client()?, pumps(1)?).and_then(
        |structures| {
            ControlledPumpBatch::<WgpuRuntime>::new(
                structures.with_operating_controls(definitions, initial, schedule, timing)?,
                responses(1),
            )
        },
    );
    let error = super::f64_rejection(built, "controlled pump")?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    Ok(())
}
