use super::case::{Case, Result, simulate};
use super::{fields, metrics};
use std::path::Path;

pub(crate) fn run() -> Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments
        .first()
        .is_some_and(|value| value == "suite" || value == "dry-suite")
    {
        if !(2..=3).contains(&arguments.len()) {
            return Err(
                "usage: reference_comparison suite|dry-suite REFERENCE_DIR [FRESH_HYDR_DIR]".into(),
            );
        }
        let cases: &[&str] = if arguments[0] == "dry-suite" {
            &["dry-ritter"]
        } else {
            &["wet-quarter", "wet-tenth"]
        };
        return suite(
            Path::new(&arguments[1]),
            arguments.get(2).map(Path::new),
            cases,
        );
    }
    if !(3..=4).contains(&arguments.len()) {
        return Err("usage: reference_comparison CASE CELLS REFERENCE.csv [NEW_HYDR.csv]".into());
    }
    let case = Case::parse(&arguments[0])?;
    let count: usize = arguments[1].parse()?;
    if ![100, 200, 400].contains(&count) {
        return Err(format!("cells={count}: choose 100, 200, or 400").into());
    }
    evaluate(
        case,
        count,
        Path::new(&arguments[2]),
        arguments.get(3).map(Path::new),
    )?;
    Ok(())
}

fn evaluate(
    case: Case,
    count: usize,
    reference_path: &Path,
    output: Option<&Path>,
) -> Result<metrics::Metrics> {
    if output == Some(reference_path) {
        return Err(format!(
            "output path={}: choose a distinct hydr CSV path",
            reference_path.display()
        )
        .into());
    }
    let reference = fields::read(reference_path, case, count)?;
    let solver = simulate(case, count)?;
    if let Some(path) = output {
        fields::write(path, case, solver.state()?.cells())?;
    }
    metrics::compare(
        case,
        solver.state()?.cells(),
        &reference,
        solver.ledger().residual,
    )
}

fn suite(directory: &Path, output: Option<&Path>, cases: &[&str]) -> Result<()> {
    if let Some(output) = output {
        std::fs::create_dir_all(output)?;
    }
    for &name in cases {
        let case = Case::parse(name)?;
        let mut previous = None;
        for cells in [100, 200, 400] {
            let reference = directory.join(format!("{name}_{cells}.csv"));
            let destination = output.map(|path| path.join(format!("hydr_{name}_{cells}.csv")));
            let measured = evaluate(case, cells, &reference, destination.as_deref())?;
            if let Some(metrics::Metrics { depth_l1, qx_l1 }) = previous
                && (measured.depth_l1 >= depth_l1 || measured.qx_l1 >= qx_l1)
            {
                return Err(format!("refinement case={name} cells={cells} depth_l1={} qx_l1={}: require both below previous {depth_l1} and {qx_l1}", measured.depth_l1, measured.qx_l1).into());
            }
            previous = Some(measured);
        }
    }
    Ok(())
}
