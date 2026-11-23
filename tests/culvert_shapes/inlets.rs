//! Arc barrel inlet control: HDS-5 codes 30 to 47 against the coefficient fixture, shape
//! matching, rejected codes, and the frozen headwater rows and limit anchors.

use super::coefficient_rows::CODES as COEFFICIENTS;
use super::inlet_rows::{
    ANCHORS, CODES, INLET_SHA256, KU, RELATIVE_TOLERANCE, ROWS, Regime, SLOPES,
};
use super::lanes::{Outcome, inlet_barrel as barrel, inlet_row, inlet_shape as shape};
use super::runtime::{Runtime, client};
use hydr::structures::{
    CulvertBarrel, CulvertInlet, CulvertInletBatch, CulvertInletForm, CulvertInletHeadwaterRow,
    CulvertInletRow, InletRegime, StructureError, StructureInputErrorKind,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

type TestResult = Outcome<()>;

/// Headwater of every frozen row from one batch.
fn headwater() -> Outcome<Vec<CulvertInletHeadwaterRow>> {
    let rows = ROWS.iter().map(inlet_row).collect::<Outcome<Vec<_>>>()?;
    let flows: Vec<f64> = ROWS
        .iter()
        .map(|row| f64::from_bits(row.discharge))
        .collect();
    let batch = CulvertInletBatch::<Runtime>::new(client(), rows)?;
    let flows = Tensor::<Runtime>::from_slice(&flows, &[flows.len()], client().device())?;
    Ok(batch.headwater(&flows)?.project()?)
}

/// Requires `bounds` to hold the exact value with bits `reference` and to lie within
/// `RELATIVE_TOLERANCE` of it.
fn holds(bounds: (f64, f64), reference: [u64; 2], what: &str) -> TestResult {
    let (low, high) = (f64::from_bits(reference[0]), f64::from_bits(reference[1]));
    if !(bounds.0 <= low && high <= bounds.1) {
        return Err(format!("{what}: {bounds:?} misses the reference [{low:e}, {high:e}]").into());
    }
    let reach = (high - bounds.0).max(bounds.1 - low);
    if reach > RELATIVE_TOLERANCE * high.abs() {
        return Err(format!("{what}: {bounds:?} reaches {reach:e} from the reference").into());
    }
    Ok(())
}

fn regime_agrees(found: InletRegime, expected: Regime) -> bool {
    use InletRegime as I;
    matches!(
        (found, expected),
        (I::Unsubmerged | I::UnsubmergedLimit, Regime::Unsubmerged)
            | (
                I::UnsubmergedLimit | I::Transition | I::SubmergedLimit,
                Regime::Transition
            )
            | (I::SubmergedLimit | I::Submerged, Regime::Submerged)
    )
}

#[test]
fn arc_codes_equal_the_coefficient_fixture_with_their_shape_and_mitered_flag() -> TestResult {
    assert_eq!(
        INLET_SHA256,
        "cc5f3f1ce0a792fbb742df68d997c7459bc45d3ce4b9d3c28e4144f9863bf35e"
    );
    assert_eq!(
        KU, 0x3ffcfb1f07d87234,
        "Ku = 0.3048^(-1/2), as the circle corpus"
    );
    let codes: Vec<u8> = CODES.iter().map(|entry| entry.code).collect();
    assert_eq!(codes, (30..=47).collect::<Vec<u8>>());
    for entry in CODES {
        let table = COEFFICIENTS
            .iter()
            .find(|row| row.code == entry.code)
            .ok_or("code missing from the coefficient fixture")?;
        let printed = [table.k, table.m, table.c, table.y].map(f64::to_bits);
        assert_eq!(
            entry.coefficients.map(f64::to_bits),
            printed,
            "{}",
            entry.code
        );
        assert_eq!(entry.mitered, table.mitered, "{}", entry.code);
        assert_eq!(table.form, 1, "{}", entry.code);
        let inlet = CulvertInlet::hds5(entry.code)?;
        assert_eq!(
            inlet.coefficients().map(f64::to_bits),
            printed,
            "{}",
            entry.code
        );
        assert_eq!(inlet.mitered(), table.mitered, "{}", entry.code);
        assert_eq!(inlet.form(), CulvertInletForm::One, "{}", entry.code);
        assert_eq!(
            inlet.barrel_shape(),
            Some(shape(entry.shape)),
            "{}",
            entry.code
        );
        let built = barrel(entry.barrel)?;
        assert_eq!(built.shape(), shape(entry.shape), "{}", entry.code);
    }
    Ok(())
}

fn definition_message<T: std::fmt::Debug>(result: Result<T, StructureError>) -> Outcome<String> {
    match result {
        Err(StructureError::InvalidInput {
            kind: StructureInputErrorKind::Definition,
            context,
            ..
        }) => Ok(context),
        other => Err(format!("expected a definition rejection, found {other:?}").into()),
    }
}

#[test]
fn other_codes_reject_naming_the_reason_and_shapes_must_match() -> TestResult {
    for code in 27..=29 {
        let text = definition_message(CulvertInlet::hds5(code))?;
        assert!(text.contains("corrugated metal box"), "{text}");
    }
    for code in 48..=57 {
        let text = definition_message(CulvertInlet::hds5(code))?;
        assert!(text.contains("tapered inlets"), "{text}");
    }
    for code in [0, 58, 255] {
        let text = definition_message(CulvertInlet::hds5(code))?;
        assert!(
            text.contains("supply an HDS-5 code from 1 to 26 or 30 to 47"),
            "{text}"
        );
    }
    // A pipe-arch code on an ellipse, and an ellipse code on a circle.
    let cases = [
        (barrel(0)?, 36, "PipeArch", "HorizontalEllipse"),
        (
            CulvertBarrel::circular(1.2)?,
            31,
            "HorizontalEllipse",
            "Circular",
        ),
        (barrel(4)?, 33, "VerticalEllipse", "Arch"),
    ];
    for (barrel, code, wanted, found) in cases {
        let row = CulvertInletRow {
            barrel,
            inlet: CulvertInlet::hds5(code)?,
            slope: 0.01,
            gravity_m_per_s2: 9.81,
        };
        let text = definition_message(CulvertInletBatch::<Runtime>::new(client(), vec![row]))?;
        assert!(
            text.contains(&format!("barrel_shape={found}"))
                && text.contains(&format!("the code is for {wanted} barrels")),
            "{text}"
        );
    }
    Ok(())
}

#[test]
fn every_arc_inlet_row_lies_inside_its_enclosure_within_the_frozen_tolerance() -> TestResult {
    let found = headwater()?;
    assert_eq!(found.len(), ROWS.len());
    let mut transition = 0;
    for (row, result) in ROWS.iter().zip(&found) {
        let what = format!(
            "code {} barrel {} slope {} at {}",
            row.code, row.barrel, row.slope, row.label
        );
        holds(result.headwater_m, row.headwater, &what)?;
        if !regime_agrees(result.regime, row.regime) {
            return Err(format!("{what}: {:?}, reference {:?}", result.regime, row.regime).into());
        }
        if row.regime == Regime::Transition {
            transition += 1;
        }
    }
    assert!(transition >= CODES.len() * SLOPES.len() * 4);
    Ok(())
}

#[test]
fn arc_limit_anchors_lie_within_tolerance_of_the_rows_beside_each_limit() -> TestResult {
    let found = headwater()?;
    for anchor in ANCHORS {
        let sides = [
            ("3.5-", anchor.at_unsubmerged_limit),
            ("3.5+", anchor.at_unsubmerged_limit),
            ("4-", anchor.at_submerged_limit),
            ("4+", anchor.at_submerged_limit),
        ];
        for (label, reference) in sides {
            let what = format!("code {} slope {} at {label}", anchor.code, anchor.slope);
            let (_, result) = ROWS
                .iter()
                .zip(&found)
                .find(|(row, _)| {
                    (row.barrel, row.code, row.slope, row.label)
                        == (anchor.barrel, anchor.code, anchor.slope, label)
                })
                .ok_or_else(|| format!("{what}: no row"))?;
            let (low, high) = (f64::from_bits(reference[0]), f64::from_bits(reference[1]));
            let reach = (high - result.headwater_m.0)
                .max(result.headwater_m.1 - low)
                .abs();
            if reach > RELATIVE_TOLERANCE * high.abs() {
                return Err(format!(
                    "{what}: {:?} lies {reach:e} from the anchor",
                    result.headwater_m
                )
                .into());
            }
        }
    }
    Ok(())
}
