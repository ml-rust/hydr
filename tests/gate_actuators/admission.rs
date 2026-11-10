use super::fixture::*;
use hydr::structures::*;

fn schedule() -> GateActuatorSchedule {
    GateActuatorSchedule {
        initial_time_s: 0.0,
        target_times_s: vec![1.0],
        end_s: 2.0,
    }
}

#[test]
fn actual_ordered_gate_subset_keeps_full_context_and_original_error_index() -> Result {
    let context = mixed_context()?;
    let batch = context.with_gate_actuators(
        vec![definition("actuator", "gate-2", 0.25, 0.5)],
        vec![initial("actuator", 0.0, 1.0)],
        schedule(),
    )?;
    assert_eq!(batch.elements(), context.elements());
    assert_eq!(batch.outputs()?[0].element, 2);
    let mut packet = update(1.0, 2.0);
    packet.structure_id = "gate-2".into();
    let error = rejected(batch.trial_to(1.0, &[packet]))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::OperatingPoint)
    );
    assert_eq!(error.element(), Some(2));
    Ok(())
}

#[test]
fn duplicate_non_gate_reordered_and_unknown_authority_reject_at_construction() -> Result {
    let context = mixed_context()?;
    for structure in ["weir", "missing"] {
        kind(
            context.with_gate_actuators(
                vec![definition("a", structure, 1.0, 1.0)],
                vec![initial("a", 0.0, 1.0)],
                schedule(),
            ),
            StructureInputErrorKind::Identity,
        )?;
    }
    for definitions in [
        vec![
            definition("a", "gate-0", 1.0, 1.0),
            definition("a", "gate-2", 1.0, 1.0),
        ],
        vec![
            definition("a", "gate-0", 1.0, 1.0),
            definition("b", "gate-0", 1.0, 1.0),
        ],
        vec![
            definition("a", "gate-2", 1.0, 1.0),
            definition("b", "gate-0", 1.0, 1.0),
        ],
    ] {
        kind(
            context.with_gate_actuators(
                definitions,
                vec![initial("a", 0.0, 1.0), initial("b", 0.0, 1.0)],
                schedule(),
            ),
            StructureInputErrorKind::Identity,
        )?;
    }
    kind(
        context.with_gate_actuators(vec![], vec![], schedule()),
        StructureInputErrorKind::Identity,
    )?;
    Ok(())
}

#[test]
fn malformed_original_rates_policies_and_schedule_reject_without_defaults() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![gate("gate", 1.0)?],
    )?;
    for variant in 0..9 {
        let mut definition = definition("actuator", "gate", 0.25, 0.5);
        match variant {
            0 => definition.opening_rate_m_per_s = -1.0,
            1 => definition.closing_rate_m_per_s = f64::NAN,
            2 => definition.accuracy.maximum_absolute_opening_error_m = -1.0,
            3 => definition.accuracy.maximum_relative_opening_error = f64::INFINITY,
            4 => definition.accuracy.maximum_arrival_window_s = -1.0,
            5 => definition.rate_provenance.clear(),
            6 => definition.ideal_response_applicability.clear(),
            7 => definition.opening_rate_m_per_s = f64::INFINITY,
            8 => definition.accuracy.maximum_arrival_window_s = f64::NAN,
            _ => return Err("Unknown definition mutation".into()),
        }
        kind(
            context.with_gate_actuators(
                vec![definition],
                vec![initial("actuator", 0.0, 1.0)],
                schedule(),
            ),
            StructureInputErrorKind::Definition,
        )?;
    }
    for times in [
        vec![0.0],
        vec![1.0, 1.0],
        vec![1.5, 1.0],
        vec![3.0],
        vec![f64::NAN],
    ] {
        let mut schedule = schedule();
        schedule.target_times_s = times;
        kind(
            context.with_gate_actuators(
                vec![definition("actuator", "gate", 0.25, 0.5)],
                vec![initial("actuator", 0.0, 1.0)],
                schedule,
            ),
            StructureInputErrorKind::Definition,
        )?;
    }
    Ok(())
}

#[test]
fn target_packets_require_complete_order_identity_causality_and_actual_range() -> Result {
    let batch = build(0.0, 1.0, (0.25, 0.5), &[1.0], 2.0)?;
    let before = batch.snapshot_data()?;
    kind(
        batch.trial_to(1.0, &[]),
        StructureInputErrorKind::OperatingPoint,
    )?;
    kind(
        batch.trial_to(0.5, &[update(0.5, 0.0)]),
        StructureInputErrorKind::OperatingPoint,
    )?;
    kind(
        batch.trial_to(1.0, &[update(1.0, 0.0), update(1.0, 0.0)]),
        StructureInputErrorKind::OperatingPoint,
    )?;
    for variant in 0..9 {
        let mut packet = update(1.0, 0.0);
        let expected = match variant {
            0 => {
                packet.actuator_id = "other".into();
                StructureInputErrorKind::Identity
            }
            1 => {
                packet.structure_id = "other".into();
                StructureInputErrorKind::Identity
            }
            2 => {
                packet.target_opening_m = -1.0;
                StructureInputErrorKind::OperatingPoint
            }
            3 => {
                packet.target_opening_m = 2.0;
                StructureInputErrorKind::OperatingPoint
            }
            4 => {
                packet.issued_at_s = 0.5;
                StructureInputErrorKind::UnsupportedRegime
            }
            5 => {
                packet.available_at_s = 2.0;
                StructureInputErrorKind::UnsupportedRegime
            }
            6 => {
                packet.target_opening_m = f64::NAN;
                StructureInputErrorKind::NonfiniteInput
            }
            7 => {
                packet.issued_at_s = f64::INFINITY;
                StructureInputErrorKind::NonfiniteInput
            }
            8 => {
                packet.available_at_s = f64::NAN;
                StructureInputErrorKind::NonfiniteInput
            }
            _ => return Err("Unknown packet mutation".into()),
        };
        kind(batch.trial_to(1.0, &[packet]), expected)?;
        assert_eq!(batch.snapshot_data()?, before);
    }
    let mut valid = update(1.0, 0.0);
    valid.available_at_s = -10.0;
    assert!(batch.trial_to(1.0, &[valid]).is_ok());
    Ok(())
}

#[test]
fn trial_endpoints_preserve_strict_chronology_and_finite_support() -> Result {
    let batch = build(0.0, 1.0, (0.25, 0.5), &[1.0], 2.0)?;
    for end in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        kind(
            batch.trial_to(end, &[]),
            StructureInputErrorKind::Definition,
        )?;
    }
    for end in [1.5, 3.0] {
        kind(
            batch.trial_to(end, &[]),
            StructureInputErrorKind::UnsupportedRegime,
        )?;
    }
    let terminal = walk(walk(batch, 1.0, &[update(1.0, 1.0)])?, 2.0, &[])?;
    assert!(terminal.next_event()?.is_none());
    kind(
        terminal.trial_to(3.0, &[]),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    Ok(())
}
