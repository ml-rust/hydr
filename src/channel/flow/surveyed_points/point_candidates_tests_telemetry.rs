use super::super::super::point_candidate_types::{
    PointCandidate, PointCandidateCertificates, PointCandidateStatus,
};
use super::{Result, fixture_model::Execution, observed::State};
use numr::{dtype::DType, runtime::Runtime};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct Summary {
    complete: usize,
    pending: usize,
    unresolved: usize,
    rejected: usize,
    expected_labels: usize,
    matched_labels: usize,
    pending_regions: BTreeMap<String, usize>,
}
impl Summary {
    pub(super) fn rejected(&mut self, id: &str) {
        self.rejected += 1;
        eprintln!(
            "POINT_OBSERVED\t{id}\tRejected(WorkLimit)\twet=NotRequested\tsonic=NotRequested,NotRequested\tmatched=[]\tpending=[]"
        );
    }
    pub(super) fn finish(&self, expected: [usize; 3], applicable_labels: usize) {
        assert_eq!(self.complete, expected[0]);
        assert_eq!(self.pending, expected[1]);
        assert_eq!(self.rejected, expected[2]);
        assert_eq!(self.unresolved, 0);
        assert_eq!(applicable_labels, 76);
        assert_eq!(self.expected_labels, applicable_labels);
        assert_eq!(self.matched_labels, applicable_labels);
        eprintln!(
            "POINT_OBSERVED_SUMMARY\tcomplete={}\tpending={}\tregime_unresolved={}\trejected={}\tmatched_labels={}/{}\tpending_regions={:?}",
            self.complete,
            self.pending,
            self.unresolved,
            self.rejected,
            self.matched_labels,
            self.expected_labels,
            self.pending_regions
        );
    }
}
pub(super) fn record<R: Runtime<DType = DType>>(
    summary: &mut Summary,
    row: &Execution,
    owner: usize,
    result: &PointCandidateCertificates<R>,
    actual: &[State],
) -> Result {
    let label = i64::try_from(owner)?;
    let status = result.status.try_to_vec::<i64>()?[owner];
    let status_name = match status {
        x if x == PointCandidateStatus::Complete as i64 => {
            summary.complete += 1;
            "Complete"
        }
        x if x == PointCandidateStatus::Pending as i64 => {
            summary.pending += 1;
            "Pending"
        }
        x if x == PointCandidateStatus::RegimeUnresolved as i64 => {
            summary.unresolved += 1;
            "RegimeUnresolved"
        }
        _ => return Err("Unknown observed point candidate status".into()),
    };
    let mut matched = Vec::new();
    for expected in row.traces {
        summary.expected_labels += 1;
        if actual
            .iter()
            .any(|s| s.owner == label && s.label == super::observed::label(expected.label))
        {
            matched.push(expected.label);
            summary.matched_labels += 1;
        }
    }
    let mut pending = BTreeMap::new();
    for candidate in &result.candidates {
        if let PointCandidate::Pending(value) = candidate {
            for owner in value.invocation_owner.try_to_vec::<i64>()? {
                if owner == label {
                    let key = format!("{:?}/{:?}", value.region, value.reason);
                    *pending.entry(key.clone()).or_insert(0_usize) += 1;
                    *summary.pending_regions.entry(key).or_default() += 1;
                }
            }
        }
    }
    let physical = actual
        .iter()
        .filter(|s| s.owner == label)
        .map(|s| s.label)
        .collect::<Vec<_>>();
    eprintln!(
        "POINT_OBSERVED\t{}\t{}\twet={}\tdepth={}\tsonic={},{}\tmatched={:?}\tactual={:?}\tpending={:?}",
        row.id,
        status_name,
        result.wet_status.try_to_vec::<i64>()?[owner],
        result.depth_status.try_to_vec::<i64>()?[owner],
        result.sonic_status[0].try_to_vec::<i64>()?[owner],
        result.sonic_status[1].try_to_vec::<i64>()?[owner],
        matched,
        physical,
        pending
    );
    Ok(())
}
