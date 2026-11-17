use super::{Result, runtime};
use hydr::structures::*;

fn gates(width: usize) -> Result<Vec<StructureElement>> {
    (0..width)
        .map(|index| {
            Ok(StructureElement {
                id: format!("gate-{index}"),
                endpoint_a: "reservoir".into(),
                endpoint_b: "river".into(),
                datum: "Analytical runtime datum".into(),
                direction: DirectionPolicy::Bidirectional,
                law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.5,
                    GateApplicability {
                        coefficient_provenance: "Analytical gate fixture".into(),
                        hydrostatic_quiescent_pool_assertion: "Quiescent analytical endpoints"
                            .into(),
                        free_submerged_aperture_applicability:
                            "Declared analytical aperture envelope".into(),
                        unobstructed_aperture_assertion: "Both endpoints expose the aperture"
                            .into(),
                        symmetric_reversal: true,
                    },
                )?),
            })
        })
        .collect()
}

fn definitions(width: usize, rates: (f64, f64)) -> Vec<GateActuatorDefinition> {
    (0..width)
        .map(|index| GateActuatorDefinition {
            actuator_id: format!("actuator-{index}"),
            structure_id: format!("gate-{index}"),
            opening_rate_m_per_s: rates.0,
            closing_rate_m_per_s: rates.1,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-10,
            },
            rate_provenance: "Explicit analytical directional rates".into(),
            ideal_response_applicability: "Ideal finite rate response without inertia or backlash"
                .into(),
        })
        .collect()
}

fn updates(width: usize, time: f64, target: f64) -> Vec<GateTargetUpdate> {
    (0..width)
        .map(|index| GateTargetUpdate {
            actuator_id: format!("actuator-{index}"),
            structure_id: format!("gate-{index}"),
            target_opening_m: target,
            issued_at_s: time,
            available_at_s: time,
        })
        .collect()
}

fn check_positions(outputs: &[GateActuatorOutput], width: usize, exact: f64) {
    assert_eq!(outputs.len(), width);
    for output in outputs {
        let position = output.position;
        assert_eq!(position.opening_m.to_bits(), exact.to_bits());
        assert!(position.lower_opening_m <= exact);
        assert!(exact <= position.upper_opening_m);
        assert!(position.absolute_error_m.is_finite());
        assert!((0.0..=1e-12).contains(&position.absolute_error_m));
        if let (Some(lower), Some(upper)) = (output.arrival_lower_s, output.arrival_upper_s) {
            assert!(lower <= upper);
            assert!(upper - lower <= 1e-10);
        }
    }
}

macro_rules! actuator_execution {
    ($runtime:ty, $client:expr) => {{
        let client = $client;
        let build = |width: usize,
                     opening: f64,
                     target: f64,
                     rates: (f64, f64),
                     times: &[f64],
                     end: f64|
         -> Result<GateActuatorBatch<$runtime>> {
            Ok(
                StructureBatch::<$runtime>::new(client.clone(), gates(width)?)?
                    .with_gate_actuators(
                        definitions(width, rates),
                        (0..width)
                            .map(|index| GateActuatorInitialState {
                                actuator_id: format!("actuator-{index}"),
                                opening_m: opening,
                                target_opening_m: target,
                            })
                            .collect(),
                        GateActuatorSchedule {
                            initial_time_s: 0.0,
                            target_times_s: times.to_vec(),
                            end_s: end,
                        },
                    )?,
            )
        };
        let walk = |mut batch: GateActuatorBatch<$runtime>,
                    end: f64,
                    packets: &[GateTargetUpdate]|
         -> Result<GateActuatorBatch<$runtime>> {
            while let Some(event) = batch.next_event()? {
                assert!(event.time_s > batch.time());
                if event.time_s >= end {
                    break;
                }
                assert!(event.target_index.is_none());
                let before = batch.snapshot_data()?;
                drop(batch.trial_to(event.time_s, &[])?);
                assert_eq!(batch.snapshot_data()?, before);
                batch = batch.trial_to(event.time_s, &[])?.into_next();
            }
            let before = batch.snapshot_data()?;
            if !packets.is_empty() {
                let mut future = packets.to_vec();
                for packet in &mut future {
                    packet.available_at_s = end + 1.0;
                }
                let error = batch
                    .trial_to(end, &future)
                    .err()
                    .ok_or("Future-availability actuator packets were accepted")?;
                assert_eq!(
                    error.input_kind(),
                    Some(StructureInputErrorKind::UnsupportedRegime)
                );
                assert_eq!(batch.snapshot_data()?, before);
            }
            drop(batch.trial_to(end, packets)?);
            assert_eq!(batch.snapshot_data()?, before);
            Ok(batch.trial_to(end, packets)?.into_next())
        };

        for width in [1, 33] {
            let mut source = build(width, 0.0, 1.0, (0.25, 0.5), &[5.0, 6.0], 9.0)?;
            for (end, target, exact, arrivals) in [
                (1.0, None, 0.25, 0),
                (4.0, None, 1.0, 1),
                (5.0, Some(0.0), 1.0, 1),
                (6.0, Some(0.0), 0.5, 1),
                (7.0, None, 0.0, 2),
                (8.0, None, 0.0, 2),
                (9.0, None, 0.0, 2),
            ] {
                let trusted = source.snapshot();
                let data = source.snapshot_data()?;
                assert_eq!(trusted.time().to_bits(), source.time().to_bits());
                assert_eq!(trusted.data()?, data);
                let mut restored = build(width, 0.0, 1.0, (0.25, 0.5), &[5.0, 6.0], 9.0)?;
                let mut imported = restored.clone();
                restored.restore(&trusted)?;
                imported.restore_data(&data)?;
                assert_eq!(restored.snapshot_data()?, data);
                assert_eq!(imported.snapshot_data()?, data);
                assert_eq!(restored.next_event()?, source.next_event()?);
                assert_eq!(imported.next_event()?, source.next_event()?);
                let packets = target
                    .map(|target| updates(width, end, target))
                    .unwrap_or_default();
                source = walk(source, end, &packets)?;
                restored = walk(restored, end, &packets)?;
                imported = walk(imported, end, &packets)?;
                check_positions(&source.outputs()?, width, exact);
                for state in source.states()? {
                    assert_eq!(state.arrival_count, arrivals);
                    assert_eq!(state.target_change_count, usize::from(end >= 5.0));
                    assert_eq!(state.episode_count, if end >= 5.0 { 2 } else { 1 });
                }
                assert_eq!(restored.snapshot_data()?, source.snapshot_data()?);
                assert_eq!(imported.snapshot_data()?, source.snapshot_data()?);
                assert_eq!(restored.next_event()?, source.next_event()?);
                assert_eq!(imported.next_event()?, source.next_event()?);
            }

            let end = f64::from_bits(50.0f64.to_bits() + 1);
            let predecessor = f64::from_bits(50.0f64.to_bits() - 1);
            let source = build(width, 1.0, 0.0, (0.0, 0.02), &[], end)?;
            let positive = walk(source, predecessor, &[])?;
            let positive_outputs = positive.outputs()?;
            check_positions(&positive_outputs, width, f64::from_bits(0x3ca17ae147ae147b));
            for output in positive_outputs {
                assert_ne!(output.support, GateActuatorSupport::ExactClosed);
            }
            let trusted = positive.snapshot();
            let owned = positive.snapshot_data()?;
            let closed = walk(positive, 50.0, &[])?;
            let closed_outputs = closed.outputs()?;
            check_positions(&closed_outputs, width, 0.0);
            for output in closed_outputs {
                assert_eq!(output.support, GateActuatorSupport::ExactClosed);
                assert_eq!(output.position.absolute_error_m.to_bits(), 0);
            }
            for state in closed.states()? {
                assert_eq!(state.arrival_count, 1);
                assert_eq!(
                    state
                        .last_completion
                        .ok_or("Missing closure receipt")?
                        .effective_upper_s,
                    50.0
                );
            }
            let mut restored = build(width, 1.0, 0.0, (0.0, 0.02), &[], end)?;
            let mut imported = restored.clone();
            restored.restore(&trusted)?;
            imported.restore_data(&owned)?;
            let expected = closed.snapshot_data()?;
            assert_eq!(walk(restored, 50.0, &[])?.snapshot_data()?, expected);
            assert_eq!(walk(imported, 50.0, &[])?.snapshot_data()?, expected);
            let held = walk(closed, end, &[])?;
            check_positions(&held.outputs()?, width, 0.0);
            for state in held.states()? {
                assert_eq!(state.arrival_count, 1);
            }
        }
    }};
}

#[test]
fn selected_cpu_client_executes_gate_motion_reversal_and_complete_restart() -> Result {
    actuator_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_gate_motion_reversal_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    actuator_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_gate_actuator_f64_without_fallback() -> Result {
    use numr::runtime::wgpu::WgpuRuntime;
    let built = StructureBatch::<WgpuRuntime>::new(super::wgpu_client()?, gates(1)?).and_then(
        |structures| {
            structures.with_gate_actuators(
                definitions(1, (0.25, 0.5)),
                vec![GateActuatorInitialState {
                    actuator_id: "actuator-0".into(),
                    opening_m: 0.0,
                    target_opening_m: 1.0,
                }],
                GateActuatorSchedule {
                    initial_time_s: 0.0,
                    target_times_s: vec![],
                    end_s: 1.0,
                },
            )
        },
    );
    let error = super::f64_rejection(built, "gate actuator")?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    Ok(())
}
