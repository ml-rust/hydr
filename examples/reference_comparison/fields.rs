use super::case::{Case, Result, centre};
use hydr::model::CellState;
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;

pub(super) fn read(path: &Path, case: Case, cells: usize) -> Result<Vec<CellState>> {
    let file = std::fs::File::open(path)?;
    parse(std::io::BufReader::new(file), case, cells)
}

fn parse(reader: impl BufRead, case: Case, cells: usize) -> Result<Vec<CellState>> {
    let mut lines = reader.lines();
    let mut metadata = BTreeMap::new();
    let header = loop {
        let line = lines
            .next()
            .transpose()?
            .ok_or("CSV is empty: supply metadata and x,h,qx fields")?;
        if let Some(comment) = line.strip_prefix("# ") {
            let (key, value) = comment
                .split_once('=')
                .ok_or_else(|| format!("CSV metadata={line:?}: require # key=value"))?;
            if metadata.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(format!("CSV metadata key={key}: remove duplicate declaration").into());
            }
        } else {
            break line;
        }
    };
    check_metadata(&metadata, case, cells)?;
    if header != "x,h,qx" {
        return Err(format!("CSV header={header:?}: require x,h,qx").into());
    }
    let mut result = Vec::with_capacity(cells);
    for (index, line) in lines.enumerate() {
        let line = line?;
        let line_number = metadata.len() + index + 2;
        let values: Vec<_> = line.split(',').collect();
        if values.len() != 3 || index >= cells {
            return Err(format!(
                "CSV line {line_number}={line:?}: require exactly {cells} rows with three numeric columns"
            )
            .into());
        }
        let x: f64 = values[0].parse()?;
        let h: f64 = values[1].parse()?;
        let qx: f64 = values[2].parse()?;
        if !x.is_finite() || !h.is_finite() || !qx.is_finite() || h <= 0.0 {
            return Err(format!(
                "CSV line {line_number}={line:?}: require finite fields and strictly positive depth"
            )
            .into());
        }
        let expected = centre(index, cells);
        if (x - expected).abs() > 1e-10 {
            return Err(format!(
                "CSV line {line_number} x={x}: require matched cell centre {expected}"
            )
            .into());
        }
        result.push(CellState { h, qx, qy: 0.0 });
    }
    if result.len() != cells {
        return Err(format!(
            "CSV rows={}: require {cells} matched cell rows",
            result.len()
        )
        .into());
    }
    Ok(result)
}

fn check_metadata(metadata: &BTreeMap<String, String>, case: Case, cells: usize) -> Result<()> {
    let required = [
        ("case", case.name.to_owned()),
        ("cells", cells.to_string()),
        ("time", "4".into()),
        ("gravity", "9.81".into()),
        ("left_depth", "1".into()),
        ("right_depth", case.downstream.to_string()),
        ("xmin", "-100".into()),
        ("xmax", "100".into()),
        ("max_step", "0.01".into()),
        ("method", "pyclaw-python-hll-order1".into()),
    ];
    if metadata.len() != required.len() {
        return Err(format!(
            "CSV metadata keys={:?}: require exactly ten declared case and method fields",
            metadata.keys()
        )
        .into());
    }
    for (key, expected) in required {
        let actual = metadata.get(key).ok_or_else(|| {
            format!("CSV metadata lacks {key}: supply the declared reference case")
        })?;
        let matches = if ["case", "method", "cells"].contains(&key) {
            actual == &expected
        } else {
            let actual: f64 = actual.parse()?;
            let expected: f64 = expected.parse()?;
            actual.is_finite() && actual == expected
        };
        if !matches {
            return Err(format!("CSV metadata {key}={actual}: require {expected}").into());
        }
    }
    Ok(())
}

pub(super) fn write(path: &Path, case: Case, cells: &[CellState]) -> Result<()> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            format!(
                "CSV output {}: {error}. Choose a new output path",
                path.display()
            )
        })?;
    let mut output = std::io::BufWriter::new(file);
    for (key, value) in [
        ("case", case.name.to_owned()),
        ("cells", cells.len().to_string()),
        ("time", "4".into()),
        ("gravity", "9.81".into()),
        ("left_depth", "1".into()),
        ("right_depth", case.downstream.to_string()),
        ("xmin", "-100".into()),
        ("xmax", "100".into()),
        ("max_step", "0.01".into()),
        ("method", "hydr-first-order-hll".into()),
    ] {
        writeln!(output, "# {key}={value}")?;
    }
    writeln!(output, "x,h,qx")?;
    for (index, cell) in cells.iter().enumerate() {
        writeln!(
            output,
            "{:.17e},{:.17e},{:.17e}",
            centre(index, cells.len()),
            cell.h,
            cell.qx
        )?;
    }
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse, write};
    use crate::comparison::case::Case;
    use hydr::model::CellState;

    const CASE: Case = Case {
        name: "wet-quarter",
        downstream: 0.25,
    };

    fn input(rows: &str) -> String {
        format!(
            "# case=wet-quarter\n# cells=1\n# time=4\n# gravity=9.81\n# left_depth=1\n# right_depth=0.25\n# xmin=-100\n# xmax=100\n# max_step=0.01\n# method=pyclaw-python-hll-order1\nx,h,qx\n{rows}"
        )
    }

    #[test]
    fn rejects_nonfinite_dry_and_misaligned_reference_fields() {
        for rows in ["0,NaN,0\n", "0,0,0\n", "1,1,0\n"] {
            assert!(
                parse(input(rows).as_bytes(), CASE, 1).is_err(),
                "rows={rows:?}"
            );
        }
    }

    #[test]
    fn rejects_missing_extra_and_incorrect_columns() {
        for rows in ["", "0,1,0\n0,1,0\n", "0,1\n"] {
            assert!(
                parse(input(rows).as_bytes(), CASE, 1).is_err(),
                "rows={rows:?}"
            );
        }
    }

    #[test]
    fn rejects_missing_duplicate_or_mismatched_case_metadata() {
        let valid = input("0,1,0\n");
        assert!(parse(valid.as_bytes(), CASE, 1).is_ok());
        for invalid in [
            "x,h,qx\n0,1,0\n".to_owned(),
            valid.replace("# time=4", "# time=0"),
            valid.replace("# case=wet-quarter", "# case=wet-tenth"),
            valid.replace("# time=4", "# time=4\n# time=4"),
            valid.replace("x,h,qx", "h,x,qx"),
        ] {
            assert!(
                parse(invalid.as_bytes(), CASE, 1).is_err(),
                "input={invalid:?}"
            );
        }
    }

    #[test]
    fn existing_output_and_hardlink_keep_reference_bytes() -> Result<(), Box<dyn std::error::Error>>
    {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "hydr-reference-fields-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&directory)?;
        let reference = directory.join("reference.csv");
        let alias = directory.join("alias.csv");
        let original = input("0,1,0\n");
        std::fs::write(&reference, original.as_bytes())?;
        std::fs::hard_link(&reference, &alias)?;
        let cells = [CellState {
            h: 0.5,
            qx: 0.1,
            qy: 0.0,
        }];
        for path in [&reference, &alias] {
            assert!(write(path, CASE, &cells).is_err());
            assert_eq!(std::fs::read(&reference)?, original.as_bytes());
        }
        std::fs::remove_file(alias)?;
        std::fs::remove_file(reference)?;
        std::fs::remove_dir(directory)?;
        Ok(())
    }
}
