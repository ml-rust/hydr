//! Public source/routing observations for independent temporal reference comparisons.

#[path = "subcatchment_reference/cases.rs"]
mod cases;
#[path = "subcatchment_reference/observe.rs"]
mod observe;
#[path = "subcatchment_reference/records.rs"]
mod records;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result {
    let mut arguments = std::env::args().skip(1);
    let mode = arguments
        .next()
        .ok_or("usage: subcatchment_reference <all|smooth|histories> <fresh-output.csv>")?;
    let output = PathBuf::from(arguments.next().ok_or("missing fresh output CSV path")?);
    if arguments.next().is_some() {
        return Err("excess arguments: supply a mode and fresh CSV path".into());
    }
    let cases = cases::catalog(&mode)?;
    let soil_path = output.with_extension("soil.csv");
    if output.exists() || soil_path.exists() {
        return Err("output or soil receipt path exists: supply fresh evidence paths".into());
    }
    let mut csv = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)?,
    );
    let mut soil = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&soil_path)?,
    );
    writeln!(csv, "{}", records::HEADER)?;
    writeln!(soil, "{}", records::SOIL_HEADER)?;
    for case in cases {
        for &dt in case.timesteps {
            observe::run(&case, dt, &mut csv, &mut soil)?;
        }
    }
    csv.flush()?;
    soil.flush()?;
    println!(
        "Wrote full endpoint observations to {} and constructed soil calls to {}",
        output.display(),
        soil_path.display()
    );
    Ok(())
}
