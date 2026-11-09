use super::{fixture::*, references::catalog};
use hydr::structures::*;

#[test]
fn mixed_layouts_preserve_every_positive_reference() -> Result {
    let cases: Vec<_> = catalog()?
        .into_iter()
        .filter(|row| row.discharge.is_some())
        .collect();
    for count in [1, 31, 32, 33, 64, 65] {
        for rotation in 0..cases.len() {
            let mut elements = Vec::new();
            let mut packets = Vec::new();
            for lane in 0..count {
                let row = &cases[(lane + rotation) % cases.len()];
                let id = format!("structure-{lane}");
                elements.push(element(
                    &id,
                    row.law.clone(),
                    DirectionPolicy::Bidirectional,
                ));
                packets.push(heads(&id, row.high, row.low));
            }
            let batch = StructureBatch::new(elements)?;
            let outputs = batch.propose(&packets)?.outputs()?;
            assert_eq!(outputs.len(), count);
            for (lane, output) in outputs.into_iter().enumerate() {
                let row = &cases[(lane + rotation) % cases.len()];
                let expected = row.discharge.ok_or("Missing admitted reference")?;
                assert_eq!(output.element, lane);
                assert_eq!(output.status, StructureStatus::Active);
                close(output.discharge_m3_per_s, expected, row.ordinary);
            }
        }
    }
    Ok(())
}

#[test]
fn tailwater_reduces_submerged_weir_discharge_and_preserves_free_limit() -> Result {
    let batch = StructureBatch::new(vec![element(
        "weir",
        weir(1.7, 2.0, 10.0)?,
        DirectionPolicy::Bidirectional,
    )])?;
    let mut previous = f64::INFINITY;
    for tailwater in [10.0, 10.25, 10.5, 10.75, 10.99] {
        let output = first(&batch.propose(&[heads("weir", 11.0, tailwater)])?)?;
        assert!(output.discharge_m3_per_s > 0.0);
        assert!(output.discharge_m3_per_s <= previous + 1e-12);
        previous = output.discharge_m3_per_s;
    }
    let equal = first(&batch.propose(&[heads("weir", 11.0, 11.0)])?)?;
    assert_eq!(equal.regime, StructureRegime::EqualHeads);
    assert_eq!(equal.status, StructureStatus::PhysicalZero);
    assert_eq!(equal.discharge_m3_per_s, 0.0);
    let batch = StructureBatch::new(vec![element(
        "limit",
        weir(1.0, 1.0, 0.0)?,
        DirectionPolicy::Bidirectional,
    )])?;
    for tailwater in [0.0, f64::from_bits(1)] {
        close(
            first(&batch.propose(&[heads("limit", 1.0, tailwater)])?)?.discharge_m3_per_s,
            1.0,
            false,
        );
    }
    Ok(())
}

#[test]
fn rounded_depth_plateau_retains_original_positive_head_gap() -> Result {
    for row in catalog()?.into_iter().filter(|row| {
        matches!(
            row.id,
            "weir_rounded_depth_plateau" | "weir_normalized_gap_underflow"
        )
    }) {
        let batch = StructureBatch::new(vec![element(
            row.id,
            row.law,
            DirectionPolicy::Bidirectional,
        )])?;
        let output = first(&batch.propose(&[heads(row.id, row.high, row.low)])?)?;
        match output.hydraulic_operands {
            Some(StructureHydraulicOperands::SubmergedWeir {
                head_gap_m,
                upstream_head_m,
                downstream_head_m,
                submergence_ratio,
                ..
            }) => {
                assert_eq!(head_gap_m.to_bits(), (row.high - row.low).to_bits());
                assert!(head_gap_m > 0.0);
                if row.id == "weir_rounded_depth_plateau" {
                    assert_eq!(upstream_head_m.to_bits(), downstream_head_m.to_bits());
                    assert_eq!(submergence_ratio, 1.0);
                } else {
                    assert_eq!(head_gap_m / upstream_head_m, 0.0);
                }
            }
            other => return Err(format!("Missing submerged operands: {other:?}").into()),
        }
    }
    Ok(())
}
