use super::{fixture::*, reference_case};
use hydr::structures::*;

fn passive(id: &str) -> Result<StructureElement> {
    let mut element = reference_case::gate(id)?;
    element.law =
        StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            1.0,
            0.0,
            1.0,
            WeirApplicability {
                coefficient_provenance: "Synthetic uncontrolled sentinel".into(),
                free_nappe_and_approach_assertion: "Declared analytical crest".into(),
                submergence_correction_assertion: "Declared analytical correction".into(),
                symmetric_reversal: true,
            },
        )?);
    Ok(element)
}

fn mixed(
    id: &str,
    width: usize,
    required: usize,
    alternate_initial: bool,
) -> Result<(OperatingControlBatch<crate::runtime::Runtime>, Vec<usize>)> {
    let reference = case(id)?;
    let mut elements = Vec::new();
    let mut definitions = Vec::new();
    let mut initial = Vec::new();
    let mut lanes = Vec::new();
    for lane in 0..width {
        let identity = reference_case::device(lane);
        if lane != required && lane % 3 == 0 {
            elements.push(passive(&identity)?);
            continue;
        }
        let gate = lane % 2 == 1;
        elements.push(if gate {
            reference_case::gate(&identity)?
        } else {
            reference_case::pump(&identity)?
        });
        let mut definition = reference_case::definition(reference, lane, &identity);
        if gate {
            definition.low_command = ControlCommand::GateTarget { opening_m: 0.25 };
            definition.high_command = ControlCommand::GateTarget { opening_m: 0.75 };
        }
        let mut state = reference_case::initial(reference, lane);
        if alternate_initial && lane % 2 == 1 {
            state.latch = ControlLatch::High;
        }
        definitions.push(definition);
        initial.push(state);
        lanes.push(lane);
    }
    Ok((
        StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), elements)?
            .with_operating_controls(
                definitions,
                initial,
                reference_case::schedule(reference),
                reference_case::timing(reference),
            )?,
        lanes,
    ))
}

#[test]
fn mixed_actual_laws_preserve_ordered_subset_indices_and_simultaneous_commands() -> Result {
    for width in reference_case::LAYOUTS {
        let (batch, lanes) = mixed("inclusive_thresholds_and_band_hold", width, width / 2, true)?;
        assert_eq!(batch.elements().len(), width);
        assert_eq!(batch.commands()?.len(), lanes.len());
        let before = batch.snapshot_data()?;
        let samples = lanes
            .iter()
            .map(|lane| observe(1.0, if lane % 2 == 0 { 2.0 } else { 1.0 }, *lane))
            .collect::<Vec<_>>();
        let trial = batch.trial_to(1.0, &samples)?;
        let diagnostics = trial.diagnostics()?;
        for (subset, lane) in lanes.iter().copied().enumerate() {
            let receipt = &diagnostics.decisions[subset];
            let left = &diagnostics.left_commands[subset];
            let right = &diagnostics.right_commands[subset];
            assert_eq!(receipt.element, lane);
            assert_eq!(left.element, lane);
            assert_eq!(right.element, lane);
            assert_eq!(right.structure_id, batch.elements()[lane].id);
            assert_eq!(receipt.before, before.states[subset]);
            assert!(receipt.transitioned);
            assert_ne!(left.latch, right.latch);
            assert_eq!(
                right.latch,
                if lane % 2 == 0 {
                    ControlLatch::High
                } else {
                    ControlLatch::Low
                }
            );
            assert_eq!(
                right.command,
                if lane % 2 == 0 {
                    ControlCommand::PumpRunning { speed_ratio: 1.0 }
                } else {
                    ControlCommand::GateTarget { opening_m: 0.25 }
                }
            );
        }
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn mixed_subset_sample_errors_retain_original_full_indices_and_atomicity() -> Result {
    for width in reference_case::LAYOUTS {
        let slot = width / 2;
        let (batch, lanes) = mixed("inclusive_thresholds_and_band_hold", width, slot, false)?;
        let samples = lanes
            .iter()
            .map(|lane| observe(1.0, 2.0, *lane))
            .collect::<Vec<_>>();
        let subset = lanes
            .iter()
            .position(|lane| *lane == slot)
            .ok_or("Missing controlled lane")?;
        for nonfinite in [false, true] {
            let mut altered = samples.clone();
            if nonfinite {
                altered[subset].stage_m = f64::NAN;
            } else {
                altered[subset].support_id.push_str(" altered");
            }
            let before = batch.snapshot_data()?;
            let error = rejected(batch.trial_to(1.0, &altered))?;
            reference_case::check_error(
                &error,
                if nonfinite {
                    "NonfiniteInput"
                } else {
                    "Identity"
                },
                Some(slot),
            )?;
            assert_eq!(batch.snapshot_data()?, before);
            assert!(batch.trial_to(1.0, &samples).is_ok());
        }
    }
    Ok(())
}

#[test]
fn isolated_timer_lateness_maps_to_its_original_mixed_element() -> Result {
    for width in reference_case::LAYOUTS {
        let slot = width / 2;
        let (base, lanes) = mixed("late_sample_preserves_accepted_pending", width, slot, false)?;
        let mut definitions = base.definitions().to_vec();
        for definition in &mut definitions {
            if definition.structure_id != reference_case::device(slot) {
                definition.high_qualification_delay_s = 10.0;
            }
        }
        let batch = StructureBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            base.elements().to_vec(),
        )?
        .with_operating_controls(
            definitions,
            base.initial_states().to_vec(),
            base.schedule().clone(),
            base.timing().clone(),
        )?;
        let samples = lanes
            .iter()
            .map(|lane| observe(1.0, 2.0, *lane))
            .collect::<Vec<_>>();
        let accepted = batch.trial_to(1.0, &samples)?.into_next();
        let before = accepted.snapshot_data()?;
        let samples = lanes
            .iter()
            .map(|lane| observe(2.5, 2.0, *lane))
            .collect::<Vec<_>>();
        let error = rejected(accepted.trial_to(2.5, &samples))?;
        reference_case::check_error(&error, "Numerical", Some(slot))?;
        assert_eq!(accepted.snapshot_data()?, before);
    }
    Ok(())
}
