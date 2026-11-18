use crate::channel::flow::surveyed_points::{
    paired_sources::PairedSources,
    source_fixture_support::{Client, Result, bounds, components, endpoints},
    wet_root_types::{WetRootDomains, WetRootEvidenceKind, WetRootStatus},
};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn check<R>(pair: &PairedSources<'_, R>, result: &WetRootDomains<R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let invocation = result.invocation_rows.try_to_vec::<i64>()?;
    let statuses = result.status.try_to_vec::<i64>()?;
    assert_eq!(invocation.len(), statuses.len());
    let diagnostics = pair.original_rows.try_to_vec::<i64>()?;
    let sections = pair.sources[0].owners().try_to_vec::<i64>()?;
    let mut located = vec![0_usize; invocation.len()];
    let mut classified = vec![false; invocation.len()];
    let mut wet_candidate = vec![false; invocation.len()];
    for domain in &result.domains {
        let owners = domain.invocation_owner.try_to_vec::<i64>()?;
        let parents = domain.parent_rows.try_to_vec::<i64>()?;
        let actual_diagnostics = domain.query.query.original_rows.try_to_vec::<i64>()?;
        let actual_sections = domain.query.query.owners.try_to_vec::<i64>()?;
        let lower = endpoints(&domain.lower_residual)?;
        let upper = endpoints(&domain.upper_residual)?;
        let retained_lower = endpoints(&bounds(pair.geometry.client(), &domain.lower_retained)?)?;
        let retained_upper = endpoints(&bounds(pair.geometry.client(), &domain.upper_retained)?)?;
        for length in [
            parents.len(),
            actual_diagnostics.len(),
            actual_sections.len(),
            lower.len(),
            upper.len(),
            retained_lower.len(),
            retained_upper.len(),
        ] {
            assert_eq!(owners.len(), length);
        }
        for (row, &owner) in owners.iter().enumerate() {
            let owner = usize::try_from(owner)?;
            let parent = usize::try_from(parents[row])?;
            assert_eq!(parents[row], invocation[owner]);
            assert_eq!(statuses[owner], WetRootStatus::Located as i64);
            assert_eq!(actual_diagnostics[row], diagnostics[parent]);
            assert_eq!(actual_sections[row], sections[parent]);
            assert!(
                lower[row][1] <= 0.0,
                "lower transformed residual {:?}",
                lower[row]
            );
            assert!(
                upper[row][0] >= 0.0,
                "upper transformed residual {:?}",
                upper[row]
            );
            assert!(retained_lower[row][0] <= lower[row][0]);
            assert!(retained_lower[row][1] >= lower[row][1]);
            assert!(retained_upper[row][0] <= upper[row][0]);
            assert!(retained_upper[row][1] >= upper[row][1]);
            located[owner] += 1;
        }
    }
    for evidence in &result.evidence {
        let owners = evidence.invocation_owner.try_to_vec::<i64>()?;
        let parents = evidence.parent_rows.try_to_vec::<i64>()?;
        let physical = endpoints(&evidence.value.physical)?;
        let retained = components(&evidence.value.retained)?;
        let available = evidence.value.available.try_to_vec::<f64>()?;
        for length in [
            parents.len(),
            physical.len(),
            retained.len(),
            available.len(),
        ] {
            assert_eq!(owners.len(), length);
        }
        for (row, &owner) in owners.iter().enumerate() {
            let owner = usize::try_from(owner)?;
            assert_eq!(parents[row], invocation[owner]);
            assert_eq!(available[row], 1.0);
            let exact_zero = retained[row]
                .iter()
                .all(|&bits| f64::from_bits(bits) == 0.0);
            let status = statuses[owner];
            if evidence.kind == WetRootEvidenceKind::Uniform
                && status == WetRootStatus::Located as i64
            {
                assert!(exact_zero);
                assert_eq!(physical[row], [0.0; 2]);
            }
            if evidence.kind == WetRootEvidenceKind::DryLimit {
                wet_candidate[owner] = physical[row][1] < 0.0;
                if status == WetRootStatus::Vacuum as i64 {
                    assert!(physical[row][0] > 0.0);
                    classified[owner] = true;
                }
                if status == WetRootStatus::TouchingDry as i64 {
                    assert!(exact_zero);
                    assert_eq!(physical[row], [0.0; 2]);
                    classified[owner] = true;
                }
            }
            if evidence.kind == WetRootEvidenceKind::Cap
                && status == WetRootStatus::CapExceeded as i64
            {
                assert!(physical[row][1] < 0.0);
                classified[owner] = true;
            }
        }
    }
    for (row, &status) in statuses.iter().enumerate() {
        assert_eq!(
            located[row],
            usize::from(status == WetRootStatus::Located as i64)
        );
        if status == WetRootStatus::Vacuum as i64
            || status == WetRootStatus::TouchingDry as i64
            || status == WetRootStatus::CapExceeded as i64
        {
            assert!(
                classified[row],
                "row {row} lacks physical classification evidence"
            );
        }
        if status == WetRootStatus::CapExceeded as i64 {
            assert!(
                wet_candidate[row],
                "row {row} lacks a strict wet-candidate invariant"
            );
        }
    }
    Ok(())
}
