use super::{context::*, history, inputs, row};
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
    writer.flush()?;
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n").into())
    }
}
