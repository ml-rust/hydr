use super::control_fixture::{Relay, WIDTHS, device, gates, samples};
use super::{Result, runtime};
use hydr::structures::*;

fn relay() -> Relay {
    Relay {
        low: ControlCommand::GateTarget { opening_m: 0.0 },
        high: ControlCommand::GateTarget { opening_m: 1.0 },
        times: vec![1.0, 2.0, 3.0],
        end: 6.0,
        maximum_gap: 3.0,
        lateness: 0.0,
    }
}

fn actuator_parts(
    width: usize,
    relay: &Relay,
) -> (
    Vec<GateActuatorDefinition>,
    Vec<GateActuatorInitialState>,
    GateActuatorSchedule,
) {
    (
        (0..width)
            .map(|lane| GateActuatorDefinition {
                actuator_id: format!("actuator-{lane}"),
                structure_id: device(lane),
                opening_rate_m_per_s: 0.5,
                closing_rate_m_per_s: 0.25,
                accuracy: GateActuatorAccuracy {
                    maximum_absolute_opening_error_m: 1e-12,
                    maximum_relative_opening_error: 1e-12,
                    maximum_arrival_window_s: 1e-10,
                },
                rate_provenance: "Explicit analytical directional rates".into(),
                ideal_response_applicability: "Ideal rate response without inertia or backlash"
                    .into(),
            })
            .collect(),
        (0..width)
            .map(|lane| GateActuatorInitialState {
                actuator_id: format!("actuator-{lane}"),
                opening_m: 0.0,
                target_opening_m: 0.0,
            })
            .collect(),
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: relay.times.clone(),
            end_s: relay.end,
        },
    )
}

/// Endpoints, decision stages, and exact openings of the switching and reversal history.
const HISTORY: [(f64, Option<f64>, f64); 9] = [
    (0.5, None, 0.0),
    (1.0, Some(2.0), 0.0),
    (1.5, None, 0.25),
    (2.0, Some(1.0), 0.5),
    (2.5, None, 0.375),
    (3.0, Some(2.0), 0.25),
    (4.0, None, 0.75),
    (4.5, None, 1.0),
    (6.0, None, 1.0),
];

macro_rules! controlled_gate_execution {
    ($runtime:ty, $client:expr) => {{
        let client = $client;
        let relay = relay();
        let mut finals = Vec::new();
        for width in WIDTHS {
            let build = || -> Result<ControlledGateBatch<$runtime>> {
                let structures = StructureBatch::<$runtime>::new(client.clone(), gates(width)?)?;
                let (definitions, initial, schedule, timing) = relay.parts(width);
                let (actuators, positions, calendar) = actuator_parts(width, &relay);
                Ok(ControlledGateBatch::<$runtime>::new(
                    structures.with_operating_controls(definitions, initial, schedule, timing)?,
                    structures.with_gate_actuators(actuators, positions, calendar)?,
                )?)
            };
            let accept = |mut batch: ControlledGateBatch<$runtime>,
                          end: f64,
                          stage: Option<f64>,
                          exact: f64|
             -> Result<ControlledGateBatch<$runtime>> {
                while let Some(event) = batch.next_event()? {
                    if event.time_s >= end {
                        break;
                    }
                    assert!(event.decision_index.is_none(), "{event:?}");
                    batch = batch.trial_to(event.time_s, &[])?.into_next();
                }
                let packets = stage.map_or_else(Vec::new, |stage| samples(width, end, stage));
                let before = batch.snapshot_data()?;
                drop(batch.trial_to(end, &packets)?);
                assert_eq!(batch.snapshot_data()?, before);
                batch = batch.trial_to(end, &packets)?.into_next();
                let outputs = batch.actuators().outputs()?;
                assert_eq!(outputs.len(), width);
                for output in outputs {
                    let position = output.position;
                    assert!(position.lower_opening_m <= exact, "{output:?}");
                    assert!(exact <= position.upper_opening_m, "{output:?}");
                    assert!(position.lower_opening_m <= position.opening_m);
                    assert!(position.opening_m <= position.upper_opening_m);
                }
                assert_eq!(
                    batch.controls().time().to_bits(),
                    batch.actuators().time().to_bits()
                );
                Ok(batch)
            };
            let resume = |mut batch: ControlledGateBatch<$runtime>| -> Result<_> {
                for (end, stage, exact) in &HISTORY[4..] {
                    batch = accept(batch, *end, *stage, *exact)?;
                }
                for (control, actor) in batch
                    .controls()
                    .states()?
                    .iter()
                    .zip(batch.actuators().states()?)
                {
                    assert_eq!(control.transition_count, 3);
                    assert_eq!(actor.target_change_count, 3);
                    assert_eq!(actor.arrival_count, 1);
                }
                Ok(batch)
            };
            let mut batch = build()?;
            for (end, stage, exact) in &HISTORY[..4] {
                batch = accept(batch, *end, *stage, *exact)?;
            }
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
fn selected_cpu_client_executes_controlled_gate_motion_and_complete_restart() -> Result {
    controlled_gate_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_controlled_gate_motion_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = controlled_gate_execution!(runtime::Runtime, runtime::client());
    let cuda = controlled_gate_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    super::same_bits(&cpu, &cuda)
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_controlled_gate_f64_without_fallback() -> Result {
    use numr::runtime::wgpu::WgpuRuntime;
    let relay = relay();
    let (definitions, initial, schedule, timing) = relay.parts(1);
    let (actuators, positions, calendar) = actuator_parts(1, &relay);
    let built = StructureBatch::<WgpuRuntime>::new(super::wgpu_client()?, gates(1)?).and_then(
        |structures| {
            ControlledGateBatch::<WgpuRuntime>::new(
                structures.with_operating_controls(definitions, initial, schedule, timing)?,
                structures.with_gate_actuators(actuators, positions, calendar)?,
            )
        },
    );
    let error = super::f64_rejection(built, "controlled gate")?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    Ok(())
}
