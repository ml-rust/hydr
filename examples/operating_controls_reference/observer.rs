use super::{
    check, history,
    reference_case::*,
    reference_inputs::cases,
    row::{self, Row},
};
use std::io::{BufWriter, Write};

pub fn run() -> Result {
    let mut writer = BufWriter::new(std::io::stdout().lock());
    writeln!(writer, "{}", row::HEADER)?;
    let mut errors = Vec::new();
    for case in cases() {
        for width in LAYOUTS {
            let batch = match build(case, width) {
                Ok(batch) => {
                    if case.binding != "success" {
                        errors.push(format!("{}: binding unexpectedly succeeds", case.id));
                    }
                    batch
                }
                Err(error) => {
                    if let Err(error) = check_error(&error, case.binding, case.binding_element) {
                        errors.push(error.to_string());
                    }
                    for lane in 0..width {
                        let antecedent = case.antecedent.map(|input| sample(input, lane));
                        row::write(
                            &mut writer,
                            Row {
                                case,
                                width,
                                lane,
                                phase: "bind",
                                decision_index: None,
                                status: category(&error),
                                element: error.element(),
                                start: case.start,
                                end: case.end,
                                accepted_time: None,
                                accepted_steps: None,
                                left: None,
                                right: None,
                                state: None,
                                receipt: None,
                                sample: antecedent.as_ref(),
                            },
                        )?;
                    }
                    continue;
                }
            };
            if case.family == "cadence" {
                for lane in 0..width {
                    let antecedent = case.antecedent.map(|input| sample(input, lane));
                    row::write(
                        &mut writer,
                        Row {
                            case,
                            width,
                            lane,
                            phase: "bind",
                            decision_index: None,
                            status: "success".into(),
                            element: None,
                            start: case.start,
                            end: case.end,
                            accepted_time: None,
                            accepted_steps: None,
                            left: None,
                            right: None,
                            state: None,
                            receipt: None,
                            sample: antecedent.as_ref(),
                        },
                    )?;
                }
                continue;
            }
            let commands = batch.commands()?;
            let states = batch.states()?;
            if commands.len() != width || states.len() != width {
                return Err("Initial control projection shape differs".into());
            }
            if batch.time().to_bits() != case.start.to_bits() || batch.accepted_steps() != 0 {
                errors.push(format!("{}: initial time or step count differs", case.id));
            }
            for (lane, (command, state)) in commands.iter().zip(&states).enumerate() {
                if let Err(error) = check::command(case, command, case.start, lane) {
                    errors.push(error.to_string());
                }
                if state.controller_id != controller(lane)
                    || state.latch != case.initial_latch
                    || state.pending != case.initial_pending
                    || state.last_sample != case.antecedent.map(|input| sample(input, lane))
                    || state.transition_count != 0
                    || state.last_transition_at_s != case.initial_transition
                    || state.last_transition.is_some()
                {
                    errors.push(format!(
                        "{} lane={lane}: initial declaration differs",
                        case.id
                    ));
                }
                row::write(
                    &mut writer,
                    Row {
                        case,
                        width,
                        lane,
                        phase: "initial",
                        decision_index: None,
                        status: "success".into(),
                        element: Some(command.element),
                        start: case.start,
                        end: case.start,
                        accepted_time: Some(batch.time()),
                        accepted_steps: Some(batch.accepted_steps()),
                        left: Some(command),
                        right: Some(command),
                        state: Some(state),
                        receipt: None,
                        sample: None,
                    },
                )?;
            }
            errors.extend(history::run(&mut writer, case, width, batch)?);
        }
    }
    writer.flush()?;
    if !errors.is_empty() {
        return Err(errors.join("\n").into());
    }
    Ok(())
}
