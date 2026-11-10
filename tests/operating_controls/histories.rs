use super::{check, fixture::*, reference_case, reference_inputs::cases};

#[test]
fn frozen_public_histories_preserve_state_timer_and_command_tables_across_layouts() -> Result {
    assert_eq!(cases().count(), 23);
    for case in cases() {
        for width in reference_case::LAYOUTS {
            let mut batch = match reference_case::build(case, width) {
                Ok(value) => {
                    assert_eq!(case.binding, "success", "{}", case.id);
                    value
                }
                Err(error) => {
                    reference_case::check_error(&error, case.binding, case.binding_element)?;
                    continue;
                }
            };
            if case.family == "cadence" {
                continue;
            }
            assert_eq!(batch.time().to_bits(), case.start.to_bits());
            assert_eq!(batch.accepted_steps(), 0);
            for (index, expected) in case.decisions.iter().enumerate() {
                let before = batch.snapshot_data()?;
                let samples = (0..width)
                    .map(|lane| reference_case::sample(expected.sample, lane))
                    .collect::<Vec<_>>();
                match batch.trial_to(expected.sample.observed, &samples) {
                    Err(error) => {
                        reference_case::check_error(&error, expected.expected, Some(0))?;
                        assert_eq!(batch.snapshot_data()?, before);
                        assert_eq!(
                            batch
                                .next_event()
                                .ok_or("Missing rejected decision")?
                                .decision_index,
                            Some(index)
                        );
                        for (lane, state) in batch.states()?.iter().enumerate() {
                            check::state(case, expected, state, lane)?;
                        }
                    }
                    Ok(trial) => {
                        assert_eq!(expected.expected, "success", "{}", case.id);
                        let diagnostics = trial.diagnostics()?;
                        let commands = trial.commands()?;
                        let states = trial.states()?;
                        assert_eq!(diagnostics.start_s.to_bits(), before.time_s.to_bits());
                        assert_eq!(
                            diagnostics.end_s.to_bits(),
                            expected.sample.observed.to_bits()
                        );
                        assert_eq!(diagnostics.decision_index, Some(index));
                        assert_eq!(diagnostics.decisions.len(), width);
                        assert_eq!(states.len(), width);
                        assert_eq!(commands, diagnostics.right_commands);
                        for (lane, state) in states.iter().enumerate() {
                            check::decision(case, expected, &diagnostics.decisions[lane], lane)?;
                            check::command(
                                case,
                                &diagnostics.left_commands[lane],
                                before.time_s,
                                lane,
                            )?;
                            check::command(
                                case,
                                &diagnostics.right_commands[lane],
                                expected.sample.observed,
                                lane,
                            )?;
                            assert_eq!(
                                diagnostics.left_commands[lane].latch,
                                before.states[lane].latch
                            );
                            assert_eq!(diagnostics.right_commands[lane].latch, state.latch);
                            assert_eq!(diagnostics.decisions[lane].before, before.states[lane]);
                            assert_eq!(diagnostics.decisions[lane].after, *state);
                        }
                        assert_eq!(batch.snapshot_data()?, before);
                        batch = trial.into_next();
                        assert_eq!(batch.time().to_bits(), expected.accepted_time.to_bits());
                        assert_eq!(batch.accepted_steps(), before.accepted_steps + 1);
                        assert_eq!(batch.last_step()?, Some(diagnostics));
                    }
                }
            }
        }
    }
    Ok(())
}
