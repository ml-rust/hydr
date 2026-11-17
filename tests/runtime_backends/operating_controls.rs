use super::control_fixture::{Relay, WIDTHS, pumps, samples};
use super::{Result, runtime};
use hydr::structures::*;

fn relay() -> Relay {
    Relay {
        low: ControlCommand::PumpStoppedIsolated,
        high: ControlCommand::PumpRunning { speed_ratio: 1.0 },
        times: vec![1.0, 2.0, 3.0],
        end: 4.0,
        maximum_gap: 4.0,
        lateness: 4.0,
    }
}

macro_rules! operating_control_execution {
    ($runtime:ty, $client:expr) => {{
        let client = $client;
        let relay = relay();
        let mut finals = Vec::new();
        for width in WIDTHS {
            let build = || -> Result<OperatingControlBatch<$runtime>> {
                let (definitions, initial, schedule, timing) = relay.parts(width);
                Ok(
                    StructureBatch::<$runtime>::new(client.clone(), pumps(width)?)?
                        .with_operating_controls(definitions, initial, schedule, timing)?,
                )
            };
            let step = |batch: &OperatingControlBatch<$runtime>,
                        time: f64,
                        stage: Option<f64>|
             -> Result<OperatingControlBatch<$runtime>> {
                let packets = stage.map_or_else(Vec::new, |stage| samples(width, time, stage));
                let before = batch.snapshot_data()?;
                drop(batch.trial_to(time, &packets)?);
                assert_eq!(batch.snapshot_data()?, before);
                Ok(batch.trial_to(time, &packets)?.into_next())
            };
            let check = |batch: &OperatingControlBatch<$runtime>,
                         latch: ControlLatch,
                         command: ControlCommand,
                         transitions: usize|
             -> Result {
                let commands = batch.commands()?;
                assert_eq!(commands.len(), width);
                for output in commands {
                    assert_eq!(output.latch, latch);
                    assert_eq!(output.command, command);
                }
                for state in batch.states()? {
                    assert_eq!(state.transition_count, transitions);
                }
                Ok(())
            };
            let mut batch = step(&build()?, 1.0, Some(2.0))?;
            check(&batch, ControlLatch::High, relay.high, 1)?;
            let trusted = batch.snapshot();
            let owned = trusted.data()?;
            let resume = |mut batch: OperatingControlBatch<$runtime>| -> Result<_> {
                batch = step(&batch, 2.0, Some(1.5))?;
                check(&batch, ControlLatch::High, relay.high, 1)?;
                batch = step(&batch, 3.0, Some(1.0))?;
                check(&batch, ControlLatch::Low, relay.low, 2)?;
                batch = step(&batch, 4.0, None)?;
                assert!(batch.next_event().is_none());
                Ok(batch)
            };
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
fn selected_cpu_client_executes_operating_control_relay_and_complete_restart() -> Result {
    operating_control_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_operating_control_relay_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = operating_control_execution!(runtime::Runtime, runtime::client());
    let cuda = operating_control_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    super::same_bits(&cpu, &cuda)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_operating_control_f64_without_fallback() -> Result {
    use numr::runtime::wgpu::WgpuRuntime;
    let (definitions, initial, schedule, timing) = relay().parts(1);
    let built = StructureBatch::<WgpuRuntime>::new(super::wgpu_client()?, pumps(1)?).and_then(
        |structures| structures.with_operating_controls(definitions, initial, schedule, timing),
    );
    let error = super::f64_rejection(built, "operating control")?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    Ok(())
}
