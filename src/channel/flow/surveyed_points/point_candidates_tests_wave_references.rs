use super::super::super::point_candidate_types::{PointCandidateCertificates, PointRegime};
use super::{Result, fixture_model::Execution, frozen};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn check<R: Runtime<DType = DType>>(
    row: &Execution,
    owner: usize,
    result: &PointCandidateCertificates<R>,
    complete: bool,
) -> Result {
    let owner = i64::try_from(owner)?;
    if row.physical == "both_dry" {
        assert!(row.edges.iter().all(|e| e.bounds == [0, 0]));
        return Ok(());
    }
    let mut available = Vec::new();
    for wave in &result.waves {
        let owners = wave.star.invocation_owner.try_to_vec::<i64>()?;
        let Some(i) = owners.iter().position(|&o| o == owner) else {
            continue;
        };
        for side in &wave.sides {
            for value in [&side.head, &side.tail, &side.shock_speed] {
                if value.available.try_to_vec::<f64>()?[i] == 1. {
                    available.push(frozen::endpoints(&value.physical)?[i]);
                }
            }
        }
    }
    let local = result
        .invocation_owner
        .try_to_vec::<i64>()?
        .iter()
        .position(|&o| o == owner)
        .ok_or("Missing wave evidence owner")?;
    let regime = result.regime.try_to_vec::<i64>()?[local];
    if regime != PointRegime::Wet as i64 && regime != PointRegime::Unresolved as i64 {
        for side in &result.dry_vacuum.edges {
            for value in [&side.head, &side.front] {
                if value.available.try_to_vec::<f64>()?[local] == 1. {
                    available.push(frozen::endpoints(&value.physical)?[local]);
                }
            }
        }
    }
    // Unlabelled authority edges retain their order. Match their physical bounds without inventing branch labels.
    for (i, edge) in row.edges.iter().enumerate() {
        assert_eq!(edge.label, format!("edge_{i}"));
        let expected = edge.bounds.map(f64::from_bits);
        let enclosed = available
            .iter()
            .any(|a| a[0] <= expected[0] && expected[1] <= a[1]);
        if complete {
            assert!(
                enclosed,
                "{} {} {:?} lacks a supporting physical wave certificate {:?}",
                row.id, edge.label, expected, available
            );
        }
    }
    Ok(())
}
