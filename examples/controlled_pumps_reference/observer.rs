use super::{
    context::*,
    history, inputs, point,
    row::{self, Frame},
};
use hydr::structures::*;
use std::io::{BufWriter, Write};

pub fn run() -> Result {
    let mut writer = BufWriter::new(std::io::stdout().lock());
    writeln!(writer, "{}", row::HEADER)?;
    let mut errors = Vec::new();
    for case in inputs::cases() {
        for width in LAYOUTS {
            errors.extend(history::run(&mut writer, &case, width)?);
        }
    }
    for admission in inputs::admissions() {
        for width in LAYOUTS {
            let case = admission_case(admission);
            let (structures, pump) = build(&case, width)?;
            let registry = structures.with_operating_sources(vec![
                OperatingSourceBinding::ControlledPump {
                    source_id: SOURCE.into(),
                    pump: &pump,
                },
            ])?;
            let frame = Frame {
                case: &case,
                family: "admission",
                width,
                record: 0,
            };
            errors.extend(point::observe(
                &mut writer,
                &frame,
                &pump,
                &registry,
                point::Query {
                    index: 0,
                    expected: admission.expected,
                    donor: admission.donor,
                    receiver: admission.receiver,
                    mode: "Direct",
                    equal: true,
                    phase: "admission",
                },
            )?);
        }
    }
    writer.flush()?;
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n").into())
    }
}
