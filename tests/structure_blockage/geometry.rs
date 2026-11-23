//! Effective geometry: zero blockage keeps every bit, discharge never rises with the blocked
//! share, and the geometry report shows both geometries.

use super::fixture::{
    Result, blocked, blocked_gate, element, floor, gate, lanes, laws, orifice, proposal, stages,
    weir,
};
use hydr::structures::{
    StructureCondition, StructureConditionKind, StructureGeometry, StructureLaw,
};

/// Blocked shares `k / 65` for `k` in `0..65`, in increasing order.
fn shares() -> Vec<f64> {
    (0..65).map(|k| f64::from(k) / 65.0).collect()
}

/// Discharge of `law` under `condition` at every stage pair of `pairs`.
fn discharges(
    law: &StructureLaw,
    condition: &StructureCondition,
    pairs: &[[f64; 2]],
) -> Result<Vec<f64>> {
    let (elements, heads) = lanes(law, condition, pairs, "lane");
    Ok(proposal(elements, &heads)?
        .outputs()?
        .into_iter()
        .map(|output| output.discharge_m3_per_s)
        .collect())
}

#[test]
fn zero_blockage_reproduces_as_built_bits() -> Result {
    for law in laws()? {
        let pairs = stages(floor(&law), 24);
        let (elements, heads) = lanes(&law, &StructureCondition::AsBuilt, &pairs, "lane");
        let built = format!("{:?}", proposal(elements, &heads)?.outputs()?);
        let mut conditions = vec![blocked(&law, 0.0)?, blocked(&law, -0.0)?];
        if matches!(law, StructureLaw::HydrostaticRectangularGate(_)) {
            conditions.push(blocked_gate(0.0, 0.0)?);
        }
        for condition in conditions {
            let (elements, heads) = lanes(&law, &condition, &pairs, "lane");
            let outputs = format!("{:?}", proposal(elements, &heads)?.outputs()?);
            assert_eq!(outputs, built, "{law:?} {condition:?}");
        }
    }
    Ok(())
}

#[test]
fn discharge_never_rises_with_the_blocked_share() -> Result {
    for law in laws()? {
        let pairs = stages(floor(&law), 8);
        let mut previous: Option<Vec<f64>> = None;
        for share in shares() {
            let current = discharges(&law, &blocked(&law, share)?, &pairs)?;
            if let Some(previous) = &previous {
                for (lane, (now, before)) in current.iter().zip(previous).enumerate() {
                    assert!(
                        now.abs() <= before.abs() && now * before >= 0.0,
                        "{law:?} share={share} lane={lane}: {now} after {before}"
                    );
                }
            }
            previous = Some(current);
        }
    }
    Ok(())
}

#[test]
fn discharge_scales_with_the_open_share() -> Result {
    for law in laws()? {
        let pairs = stages(floor(&law), 8);
        let base = discharges(&law, &StructureCondition::AsBuilt, &pairs)?;
        for share in [0.1, 0.25, 0.5, 0.9] {
            let open = 1.0 - share;
            for (lane, (q, q0)) in discharges(&law, &blocked(&law, share)?, &pairs)?
                .iter()
                .zip(&base)
                .enumerate()
            {
                let expected = open * q0;
                assert!(
                    (q - expected).abs() <= 1e-12 * expected.abs(),
                    "{law:?} share={share} lane={lane}: {q} against {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn opening_range_blockage_leaves_discharge_unchanged() -> Result {
    let law = gate()?;
    let pairs = stages(0.0, 8);
    let base = discharges(&law, &StructureCondition::AsBuilt, &pairs)?;
    let ranged = discharges(&law, &blocked_gate(0.0, 0.5)?, &pairs)?;
    assert_eq!(format!("{base:?}"), format!("{ranged:?}"));
    Ok(())
}

#[test]
fn geometry_report_shows_base_and_effective_geometry() -> Result {
    let elements = vec![
        element("w", weir()?, blocked(&weir()?, 0.25)?),
        element("o", orifice()?, StructureCondition::AsBuilt),
        element("g", gate()?, blocked_gate(0.5, 0.25)?),
        element("c", weir()?, super::fixture::closed()),
    ];
    let reports = super::fixture::batch(elements)?.geometry();
    assert_eq!(reports.len(), 4);
    let kinds = reports.iter().map(|r| r.condition).collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [
            StructureConditionKind::Blocked,
            StructureConditionKind::AsBuilt,
            StructureConditionKind::Blocked,
            StructureConditionKind::Closed,
        ]
    );
    assert_eq!(
        reports[0].base,
        StructureGeometry::Weir {
            crest_length_m: 2.0
        }
    );
    assert_eq!(
        reports[0].effective,
        Some(StructureGeometry::Weir {
            crest_length_m: 1.5
        })
    );
    assert_eq!(Some(reports[1].base.clone()), reports[1].effective);
    assert_eq!(
        reports[2].effective,
        Some(StructureGeometry::Gate {
            width_m: 1.0,
            maximum_opening_m: 0.75
        })
    );
    assert_eq!(reports[3].effective, None);
    for (index, report) in reports.iter().enumerate() {
        assert_eq!(report.element, index);
    }
    Ok(())
}
