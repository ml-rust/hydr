use super::fixtures::{CASES, Case};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::io::Write as _;

const IDS: [&str; 7] = [
    "trapezoid_wet",
    "trapezoid_wet_reverse",
    "legacy_multiband",
    "original_rounded_wet_fan",
    "both_knots_dry_fan",
    "both_knots_dry_fan_reverse",
    "generated_vacuum",
];
const COUNTS: [usize; 3] = [100, 200, 400];
type TestResult = Result<(), Box<dyn std::error::Error>>;

fn finite(bits: u64, id: &str, field: &str) -> f64 {
    let value = f64::from_bits(bits);
    assert!(
        value.is_finite(),
        "{id}: {field} is not finite: {bits:016x}"
    );
    value
}

fn check_metadata(case: &Case) {
    assert!(!case.points.is_empty(), "{}: profile is empty", case.id);
    for point in case.points {
        for bits in point {
            finite(*bits, case.id, "profile coordinate");
        }
    }
    finite(case.cap, case.id, "maximum stage");
    assert!(finite(case.gravity, case.id, "gravity") > 0.0);
    for state in case.initial {
        assert!(finite(state[0], case.id, "initial area") >= 0.0);
        finite(state[1], case.id, "initial discharge");
    }
    for state in case.exterior {
        assert!(finite(state[0], case.id, "exterior depth") >= 0.0);
        finite(state[1], case.id, "exterior velocity");
    }
    for scale in case.scales {
        assert!(finite(scale, case.id, "normalization scale") > 0.0);
    }

    let (classification, directions): (&str, &[i32]) = match case.id {
        "both_knots_dry_fan" => ("right_dry", &[-1]),
        "both_knots_dry_fan_reverse" => ("left_dry", &[1]),
        "generated_vacuum" => ("vacuum", &[-1, 1]),
        _ => ("wet_wet", &[]),
    };
    assert_eq!(case.classification, classification, "{}", case.id);
    assert_eq!(case.fronts.len(), directions.len(), "{}", case.id);
    let mut previous_hi = -100.0;
    for (front, direction) in case.fronts.iter().zip(directions) {
        assert_eq!(front.direction, *direction, "{}", case.id);
        let threshold = finite(front.threshold, case.id, "front threshold");
        assert!(
            0.0 < threshold && threshold < f64::from_bits(case.scales[0]),
            "{}",
            case.id
        );
        let lo = finite(front.crossing[0], case.id, "front crossing lower endpoint");
        let hi = finite(front.crossing[1], case.id, "front crossing upper endpoint");
        assert!(previous_hi <= lo && lo <= hi && hi <= 100.0, "{}", case.id);
        previous_hi = hi;
    }
}

fn check_intervals(case: &Case) {
    for (count, cells) in COUNTS.into_iter().zip(case.grids) {
        assert_eq!(cells.len(), count, "{}: grid {count}", case.id);
        for (index, cell) in cells.iter().enumerate() {
            for component in 0..2 {
                let lo = finite(cell[2 * component], case.id, "cell lower endpoint");
                let hi = finite(cell[2 * component + 1], case.id, "cell upper endpoint");
                assert!(
                    lo <= hi,
                    "{}: grid {count}, cell {index}, component {component}",
                    case.id
                );
                if component == 0 {
                    assert!(
                        lo >= 0.0,
                        "{}: grid {count}, cell {index}: negative area",
                        case.id
                    );
                }
                let allowance = 0.03 * f64::from_bits(case.scales[component]) / 100.0;
                assert!(
                    hi - lo < allowance,
                    "{}: grid {count}, cell {index}, component {component}: interval width exceeds allowance",
                    case.id
                );
            }
        }
    }
}

fn check_original_fan(case: &Case) {
    assert_eq!(case.cap, 0x4008000000000000);
    assert_eq!(case.gravity, 0x3ff0000000000000);
    assert_eq!(case.scales[0], 0x402c000000000000);
    assert_eq!(case.initial[0][0], 0x402c000000000000);
    assert_eq!(case.initial[1][1], 0x400451bec8c475f1);
    assert_eq!(case.exterior[1][1], 0x400451bec8c475f1);
    assert_eq!(
        case.points,
        &[
            [0xc010000000000000, 0x4008000000000000],
            [0xc004000000000000, 0x4000000000000000],
            [0xc000000000000000, 0x3ff0000000000000],
            [0xbff0000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x3ff0000000000000],
            [0x4004000000000000, 0x4000000000000000],
            [0x4010000000000000, 0x4008000000000000],
        ]
    );
}

fn write_bits<const N: usize>(output: &mut String, bits: &[u64; N]) -> std::fmt::Result {
    output.push('[');
    for (index, value) in bits.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "\"0x{value:016x}\"")?;
    }
    output.push(']');
    Ok(())
}

fn write_rows<const N: usize>(output: &mut String, rows: &[[u64; N]]) -> std::fmt::Result {
    output.push('[');
    for (index, row) in rows.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write_bits(output, row)?;
    }
    output.push(']');
    Ok(())
}

fn canonical_json(cases: &[&Case]) -> Result<String, std::fmt::Error> {
    let mut output = String::new();
    output.push_str("{\"format\":\"hydr-surveyed-transport-bits-v1\",\"encoding\":\"IEEE-754 binary64 hexadecimal bits\",\"cases\":[");
    for (index, case) in cases.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(
            output,
            "{{\"id\":\"{}\",\"classification\":\"{}\",\"points\":",
            case.id, case.classification
        )?;
        write_rows(&mut output, case.points)?;
        write!(
            output,
            ",\"cap\":\"0x{:016x}\",\"gravity\":\"0x{:016x}\",\"initial\":",
            case.cap, case.gravity
        )?;
        write_rows(&mut output, &case.initial)?;
        output.push_str(",\"exterior\":");
        write_rows(&mut output, &case.exterior)?;
        output.push_str(",\"scales\":");
        write_bits(&mut output, &case.scales)?;
        output.push_str(",\"grids\":[");
        for (grid_index, (count, cells)) in COUNTS.into_iter().zip(case.grids).enumerate() {
            if grid_index != 0 {
                output.push(',');
            }
            write!(output, "{{\"count\":{count},\"cells\":")?;
            write_rows(&mut output, cells)?;
            output.push('}');
        }
        output.push_str("],\"fronts\":[");
        for (front_index, front) in case.fronts.iter().enumerate() {
            if front_index != 0 {
                output.push(',');
            }
            write!(
                output,
                "{{\"direction\":{},\"threshold\":\"0x{:016x}\",\"crossing\":",
                front.direction, front.threshold
            )?;
            write_bits(&mut output, &front.crossing)?;
            output.push('}');
        }
        output.push_str("]}");
    }
    output.push_str("]}\n");
    Ok(output)
}

#[test]
fn frozen_transport_metadata_and_intervals_are_consistent() -> TestResult {
    assert_eq!(CASES.len(), IDS.len());
    let actual: BTreeSet<_> = CASES.iter().map(|case| case.id).collect();
    let expected: BTreeSet<_> = IDS.into_iter().collect();
    assert_eq!(actual.len(), CASES.len(), "Duplicate transport fixture IDs");
    assert_eq!(actual, expected);
    let mut canonical_cases = Vec::with_capacity(IDS.len());
    for id in IDS {
        let case = CASES.iter().find(|case| case.id == id).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Missing fixture {id}"),
            )
        })?;
        check_metadata(case);
        check_intervals(case);
        if id == "original_rounded_wet_fan" {
            check_original_fan(case);
        }
        canonical_cases.push(case);
    }

    let bytes = canonical_json(&canonical_cases)?.into_bytes();
    let digest = bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    // Independently checked against the frozen two-second reference data.
    assert_eq!(bytes.len(), 427_344);
    assert_eq!(digest, 0x3f33439128850330, "Transport fixture bits changed");
    if let Some(path) = std::env::var_os("HYDR_TRANSPORT_FIXTURE_EXPORT") {
        // Refuse existing paths, including frozen authority files and symlinks.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(&bytes)?;
        println!(
            "Transport fixture export: {path:?}, bytes={}, fnv1a64={digest:016x}",
            bytes.len()
        );
    }
    Ok(())
}
