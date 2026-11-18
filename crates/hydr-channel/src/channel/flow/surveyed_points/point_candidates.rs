#[cfg(test)]
use super::point_candidate_types::PointFanAuthority;
use super::{
    dry_vacuum_types::{DryVacuumCandidate, DryVacuumRegime, DryVacuumStatus},
    point_candidate_types::{
        PointCandidate, PointCandidateCertificates, PointCandidateStatus, PointDryEvidence,
        PointFan, PointPendingReason, PointRegion,
    },
    point_root_session::{PointRootSession, invalid},
    wet_root_types::WetRootStatus,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

impl<R> PointRootSession<'_, '_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    pub(super) fn certify_point_candidates(
        &mut self,
        invocation_owner: &Tensor<R>,
    ) -> HydraulicResult<PointCandidateCertificates<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.wet.terminal || self.sonic.requests.iter().any(|r| r.terminal) {
            return Err(invalid(
                "point candidates remain sealed after an earlier numerical error",
            ));
        }
        let c = self.sources.geometry.client();
        let (owners, parent) =
            checked::<R, _>(c, "point candidate complete owner admission", |a| {
                super::dry_vacuum_evidence::owners(a, self, invocation_owner)
            })?;
        super::point_candidates_admission::check(self)?;
        let dry = self.certify_dry_vacuum(&owners)?;
        let (wet, demand) = checked::<R, _>(c, "point candidate strict wet regime routing", |a| {
            let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
            let one = Tensor::full_scalar(owners.shape(), a.dtype(), 1., c.device())?;
            let wet = c.where_cond(
                &c.eq(
                    &dry.regime,
                    &Tensor::full_scalar(
                        &[],
                        DType::I64,
                        DryVacuumRegime::NotApplicable as i64 as f64,
                        c.device(),
                    )?,
                )?,
                &one,
                &zero,
            )?;
            let demand = c.index_put(
                &Tensor::zeros(self.invocation_rows.shape(), a.dtype(), c.device())?,
                0,
                &owners,
                &wet,
            )?;
            Ok((wet, demand))
        })?;
        let status = c.index_select(&self.wet.locator_status, 0, &owners)?;
        let missing = c.eq(
            &status,
            &Tensor::full_scalar(
                &[],
                DType::I64,
                WetRootStatus::NotRequested as i64 as f64,
                c.device(),
            )?,
        )?;
        let missing = c.where_cond(
            &missing,
            &wet,
            &Tensor::zeros(wet.shape(), wet.dtype(), c.device())?,
        )?;
        let rows = c.nonzero(&missing)?;
        let rows = rows.reshape(&[rows.numel()])?;
        if rows.numel() != 0 {
            self.start_wet(&c.index_select(&owners, 0, &rows)?)?;
        }
        let waves = self.certify_wet_waves_selected(&demand)?;
        let mut wet_result = super::wet_candidates::evaluate(self, &waves)?;
        let mut candidates = Vec::new();
        checked::<R, _>(c, "point candidate admitted dry payload transfer", |a| {
            for candidate in dry.candidates {
                match candidate {
                    DryVacuumCandidate::Exterior(value) => {
                        candidates.push(PointCandidate::Exterior(value))
                    }
                    DryVacuumCandidate::Dry(value) => candidates.push(PointCandidate::Dry(value)),
                    DryVacuumCandidate::Fan(certificate) => {
                        let zero =
                            Tensor::zeros(certificate.status.shape(), a.dtype(), c.device())?;
                        let one = Tensor::full_scalar(
                            certificate.status.shape(),
                            a.dtype(),
                            1.,
                            c.device(),
                        )?;
                        let active = c.where_cond(
                            &c.eq(
                                &certificate.status,
                                &Tensor::zeros(&[], DType::I64, c.device())?,
                            )?,
                            &one,
                            &zero,
                        )?;
                        let region =
                            if certificate.side == super::source_sonic_types::SonicSide::Left {
                                PointRegion::LeftFan
                            } else {
                                PointRegion::RightFan
                            };
                        super::wet_candidates::pending(
                            a,
                            &mut candidates,
                            &certificate.invocation_owner,
                            &certificate.parent_rows,
                            &c.sub(&one, &active)?,
                            region,
                            PointPendingReason::FanCertificate,
                        )?;
                        candidates.push(PointCandidate::Fan(Box::new(PointFan {
                            certificate: *certificate,
                            active,
                            #[cfg(test)]
                            authority: PointFanAuthority::DryVacuum,
                        })));
                    }
                    DryVacuumCandidate::PendingFan(value) => {
                        let one = Tensor::full_scalar(
                            value.invocation_owner.shape(),
                            a.dtype(),
                            1.,
                            c.device(),
                        )?;
                        let region = if value.side == super::source_sonic_types::SonicSide::Left {
                            PointRegion::LeftFan
                        } else {
                            PointRegion::RightFan
                        };
                        super::wet_candidates::pending(
                            a,
                            &mut candidates,
                            &value.invocation_owner,
                            &value.parent_rows,
                            &one,
                            region,
                            PointPendingReason::FanCertificate,
                        )?;
                    }
                }
            }
            let located = c.index_select(&wet_result.located, 0, &owners)?;
            let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
            let missing = c.mul(&wet, &c.eq(&located, &zero)?)?;
            super::wet_candidates::pending(
                a,
                &mut wet_result.candidates,
                &owners,
                &parent,
                &missing,
                PointRegion::Unlocated,
                PointPendingReason::WetRoot,
            )?;
            Ok(())
        })?;
        candidates.extend(wet_result.candidates);
        #[cfg(test)]
        let (depth, evidence) = super::point_candidates_evidence::snapshot(self, &demand)?;
        let status = checked::<R, _>(c, "point candidate complete physical alternatives", |a| {
            let code = |v: PointCandidateStatus| {
                Tensor::full_scalar(owners.shape(), DType::I64, v as i64 as f64, c.device())
            };
            let drycomplete = c.eq(
                &dry.status,
                &Tensor::full_scalar(
                    &[],
                    DType::I64,
                    DryVacuumStatus::Complete as i64 as f64,
                    c.device(),
                )?,
            )?;
            let drycomplete = c.where_cond(
                &drycomplete,
                &Tensor::full_scalar(owners.shape(), a.dtype(), 1., c.device())?,
                &Tensor::zeros(owners.shape(), a.dtype(), c.device())?,
            )?;
            let validwet = c.index_select(&wet_result.complete, 0, &owners)?;
            let coverage = c.index_select(&wet_result.covered, 0, &owners)?;
            let wetcomplete = c.mul(&validwet, &coverage)?;
            let complete = c.where_cond(&wet, &wetcomplete, &drycomplete)?;
            let mut status = c.where_cond(
                &complete,
                &code(PointCandidateStatus::Complete)?,
                &code(PointCandidateStatus::Pending)?,
            )?;
            status = c.where_cond(
                &c.eq(
                    &dry.status,
                    &Tensor::full_scalar(
                        &[],
                        DType::I64,
                        DryVacuumStatus::RegimeUnresolved as i64 as f64,
                        c.device(),
                    )?,
                )?,
                &code(PointCandidateStatus::RegimeUnresolved)?,
                &status,
            )?;
            Ok(status)
        })?;
        Ok(PointCandidateCertificates {
            invocation_witness: self.invocation.witness(),
            wet_status: c.index_select(&self.wet.locator_status, 0, &owners)?,
            #[cfg(test)]
            depth_status: c.index_select(&depth, 0, &owners)?,
            #[cfg(test)]
            sonic_status: [
                c.index_select(&self.sonic.status[0], 0, &owners)?,
                c.index_select(&self.sonic.status[1], 0, &owners)?,
            ],
            regime: dry.regime.clone(),
            status,
            invocation_owner: owners,
            parent_rows: parent,
            dry_vacuum: PointDryEvidence {
                regime: dry.regime,
                status: dry.status,
                invariant: dry.invariant,
                edges: dry.edges,
                possibilities: dry.possibilities,
            },
            #[cfg(test)]
            wet_evidence: evidence,
            waves,
            candidates,
            #[cfg(test)]
            fan_evidence: wet_result.evidence,
            callback_points: self.callback_points(),
            point_limit: self.point_limit(),
        })
    }
}

#[cfg(test)]
#[path = "point_candidates_fixture.rs"]
pub(in crate::channel::flow::surveyed_points) mod fixture;

#[cfg(test)]
mod tests {
    use super::super::corpus_coverage::Coverage;
    use super::super::{
        chart_fixture,
        point_candidate_types::{
            PointCandidate, PointCandidateStatus, PointFanAuthority, PointRegime,
        },
        point_root_session::PointRootSession,
        source_fixture_support as frozen,
        wet_depth::fixture as support,
    };
    use super::fixture::{
        diagnostics, fixture_model, fixtures, flux_fixture_model, flux_fixtures, observed,
        wave_references,
    };
    use frozen::{Result, hydraulic};
    use numr::{dtype::DType, runtime::Runtime};

    /// Checks one exact mixed control, then the frozen corpus at the given coverage.
    fn matrix<R>(client: &R::Client, coverage: Coverage) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_fixture::config();
        support::with_rectangle::<R, _>(
            client,
            1.,
            4.,
            &[[1., 0., 1., 0.], [0., 0., 0., 0.], [1., 0., 0., 0.]],
            &config,
            |pair, work| {
                let parents = support::rows::<R>(client, &[0, 2, 0, 1])?;
                let mut session = hydraulic(PointRootSession::new(
                    pair,
                    &parents,
                    config.root.max_points,
                ))?;
                let owners = support::rows::<R>(client, &[3, 2, 1, 0])?;
                let result = hydraulic(session.certify_point_candidates(&owners))?;
                hydraulic(session.invocation.check(&result.invocation_witness))?;
                assert_eq!(result.invocation_owner.try_to_vec::<i64>()?, [3, 2, 1, 0]);
                assert_eq!(result.parent_rows.try_to_vec::<i64>()?, [1, 0, 2, 0]);
                assert_eq!(
                    result.regime.try_to_vec::<i64>()?,
                    [
                        PointRegime::BothDry as i64,
                        PointRegime::Wet as i64,
                        PointRegime::LeftWet as i64,
                        PointRegime::Wet as i64
                    ]
                );
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointCandidateStatus::Complete as i64; 4]
                );
                let mut starowners = Vec::new();
                let mut fanowners = Vec::new();
                let mut dryowners = Vec::new();
                for candidate in &result.candidates {
                    match candidate {
                        PointCandidate::Star(star) => {
                            let wave = &result.waves[star.wave];
                            let owners = star.invocation_owner.try_to_vec::<i64>()?;
                            let rows = star.rows.try_to_vec::<i64>()?;
                            let waveowners = wave.star.invocation_owner.try_to_vec::<i64>()?;
                            let q = frozen::endpoints(&wave.star.discharge)?;
                            for (&owner, &row) in owners.iter().zip(&rows) {
                                assert_eq!(waveowners[usize::try_from(row)?], owner);
                                assert_eq!(q[usize::try_from(row)?], [0., 0.]);
                            }
                            starowners.extend(owners);
                        }
                        PointCandidate::Fan(fan) => {
                            assert!(matches!(fan.authority, PointFanAuthority::DryVacuum));
                            fanowners.extend(fan.certificate.invocation_owner.try_to_vec::<i64>()?);
                            assert_eq!(fan.active.try_to_vec::<f64>()?, [1.]);
                        }
                        PointCandidate::Dry(dry) => {
                            dryowners.extend(dry.invocation_owner.try_to_vec::<i64>()?)
                        }
                        PointCandidate::Exterior(_) => {
                            return Err("Resting uniform source produces exterior candidate".into());
                        }
                        PointCandidate::Pending(_) => {
                            return Err("Exact mixed control retains pending candidate".into());
                        }
                    }
                }
                starowners.sort();
                assert_eq!(starowners, [0, 2]);
                assert_eq!(fanowners, [1]);
                assert_eq!(dryowners, [3]);
                let points = session.callback_points();
                let before = work.snapshot().residual_points;
                let repeated = hydraulic(session.certify_point_candidates(&owners))?;
                assert_eq!(repeated.status.try_to_vec::<i64>()?, [0; 4]);
                assert_eq!(session.callback_points(), points);
                assert_eq!(work.snapshot().residual_points, before);
                Ok(())
            },
        )?;
        corpus::run::<R>(client, coverage)
    }

    mod contacts {
        use super::super::super::{
            point_candidate_types::PointCandidateStatus,
            source_sonic_types::{SonicSide, SourceSonicStatus},
        };
        use super::{
            PointCandidate, PointFanAuthority, PointRootSession, Result, chart_fixture, frozen,
            hydraulic, support,
        };
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            // Independent rectangle controls: source contact, critical equality, and star-sonic contact at A=1.
            support::with_rectangle::<R, _>(
                client,
                1.,
                8.,
                &[[1., 1., 0.25, 0.5], [1., 1., 1., 1.], [4., -4., 1., 1.]],
                &config,
                |pair, work| {
                    let parents = support::rows::<R>(client, &[0, 1, 2])?;
                    let mut session = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    for owner in 0..3 {
                        let owners = support::rows::<R>(client, &[owner])?;
                        let mut result = hydraulic(session.certify_point_candidates(&owners))?;
                        if owner == 2 {
                            hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance {
                                atol: 1e-17,
                                rtol: 0.,
                            }))?;
                            hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                                atol: 1e-17,
                                rtol: 0.,
                            }))?;
                            result = hydraulic(session.certify_point_candidates(&owners))?;
                        }
                        super::diagnostics::check(&session, &result)?;
                        assert_eq!(
                            result.status.try_to_vec::<i64>()?,
                            [PointCandidateStatus::Complete as i64],
                            "contact {owner}"
                        );
                        let actual = super::observed::actual(&session, &result)?;
                        if owner == 0 {
                            assert_eq!(
                                result.sonic_status[0].try_to_vec::<i64>()?,
                                [SourceSonicStatus::SourceContact as i64]
                            );
                            assert!(actual.iter().all(|s| s.label == "left-exterior"));
                            assert!(!actual.is_empty());
                            for state in actual {
                                assert_eq!(state.area, [1., 1.]);
                                assert_eq!(state.discharge, [1., 1.]);
                            }
                            assert!(result.fan_evidence.iter().any(|e| {
                                e.side == SonicSide::Left
                                    && e.membership
                                        .proved
                                        .try_to_vec::<f64>()
                                        .is_ok_and(|v| v == [1.])
                            }));
                        } else if owner == 1 {
                            assert!(actual.iter().any(|s| s.label == "left-exterior"));
                            assert!(actual.iter().any(|s| s.label == "star"));
                            assert!(
                                !result
                                    .candidates
                                    .iter()
                                    .any(|s| matches!(s, PointCandidate::Fan(_)))
                            );
                            for state in &result.sonic_status {
                                assert_eq!(
                                    state.try_to_vec::<i64>()?,
                                    [SourceSonicStatus::NotRequested as i64]
                                );
                            }
                        } else {
                            assert!(actual.iter().any(|s| s.label == "star"));
                            assert!(actual.iter().any(|s| s.label == "left-sonic"));
                            for state in actual {
                                super::observed::contains(
                                    state.area,
                                    [1.0_f64.to_bits(); 2],
                                    "star-sonic contact area",
                                );
                                super::observed::contains(
                                    state.discharge,
                                    [1.0_f64.to_bits(); 2],
                                    "star-sonic contact Q",
                                );
                            }
                            let fan = result
                                .candidates
                                .iter()
                                .find_map(|c| {
                                    if let PointCandidate::Fan(f) = c {
                                        Some(f)
                                    } else {
                                        None
                                    }
                                })
                                .ok_or("Missing established star-sonic candidate")?;
                            let PointFanAuthority::Wet(index) = fan.authority else {
                                return Err("Wet contact loses scoped membership".into());
                            };
                            let proof = &result.fan_evidence[index].membership;
                            assert_eq!(proof.coincidence.try_to_vec::<f64>()?, [1.]);
                            assert_eq!(frozen::components(&proof.lower_difference)?, [[0; 3]]);
                            assert_eq!(frozen::components(&proof.upper_difference)?, [[0; 3]]);
                        }
                    }
                    assert_eq!(session.callback_points(), work.snapshot().residual_points);
                    Ok(())
                },
            )
        }
    }

    mod corpus {
        use super::super::super::{
            point_candidate_types::PointCandidateStatus, wet_root_types::WetRootStatus,
        };
        use super::{
            PointRootSession, Result, chart_fixture, fixture_model::Execution, fixtures, frozen,
            hydraulic, support,
        };
        use crate::channel::flow::{
            surveyed::error::{HydraulicError, PrismaticRiemannRejection},
            surveyed_points::corpus_coverage::Coverage,
        };
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let corpus = fixtures::executions();
            assert_eq!(corpus.len(), 70);
            let all = coverage.select(corpus, |row| (row.outcome, row.physical));
            let mut executed = 0;
            let mut summary = super::telemetry::Summary::default();
            for (config_index, declared) in fixtures::CONFIGS.iter().enumerate() {
                let rows = all
                    .iter()
                    .copied()
                    .filter(|r| r.config == config_index)
                    .collect::<Vec<_>>();
                if rows.is_empty() {
                    continue;
                }
                let ids = rows.iter().map(|r| r.id).collect::<Vec<_>>();
                let config = declared.build();
                chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                    let original = source_bits(pair)?;
                    let indices = (0..rows.len())
                        .map(i64::try_from)
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let parents = support::rows::<R>(client, &indices)?;
                    let mut session = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    let result = session.certify_point_candidates(&parents);
                    if rows.iter().all(|r| r.outcome == "WorkLimit") {
                        assert!(
                            matches!(
                                result,
                                Err(HydraulicError::Rejected {
                                    reason: PrismaticRiemannRejection::WorkLimit,
                                    ..
                                })
                            ),
                            "Declared zero work execution loses its WorkLimit rejection"
                        );
                        for row in &rows {
                            summary.rejected(row.id);
                        }
                        assert_eq!(work.snapshot().residual_points, 0);
                        return Ok(());
                    }
                    let result = hydraulic(result)?;
                    super::diagnostics::check(&session, &result)?;
                    assert_eq!(source_bits(session.sources())?, original);
                    assert_eq!(session.callback_points(), work.snapshot().residual_points);
                    assert!(session.callback_points() <= config.root.max_points);
                    let actual = super::observed::actual(&session, &result)?;
                    let status = result.status.try_to_vec::<i64>()?;
                    let root = result.wet_status.try_to_vec::<i64>()?;
                    for (i, row) in rows.iter().enumerate() {
                        super::telemetry::record(&mut summary, row, i, &result, &actual)?;
                        if row.outcome == "CapExceeded" || row.outcome == "physical_cap_rejection" {
                            assert_eq!(root[i], WetRootStatus::CapExceeded as i64, "{}", row.id);
                            assert_eq!(
                                status[i],
                                PointCandidateStatus::Pending as i64,
                                "{}",
                                row.id
                            );
                            let owner = i64::try_from(i)?;
                            assert!(actual.iter().all(|a| a.owner != owner));
                            continue;
                        }
                        assert_eq!(
                            status[i],
                            PointCandidateStatus::Complete as i64,
                            "{} loses physical candidate completeness under its declared config",
                            row.id
                        );
                        check_states(
                            row,
                            i,
                            &actual,
                            status[i] == PointCandidateStatus::Complete as i64,
                        )?;
                        super::wave_check::check(
                            row,
                            i,
                            &result,
                            status[i] == PointCandidateStatus::Complete as i64,
                        )?;
                    }
                    Ok(())
                })?;
                executed += rows.len();
            }
            assert_eq!(executed, all.len());
            let cap_rows = all
                .iter()
                .filter(|r| r.outcome == "CapExceeded" || r.outcome == "physical_cap_rejection")
                .count();
            let work_rows = all.iter().filter(|r| r.outcome == "WorkLimit").count();
            let applicable_labels = all
                .iter()
                .filter(|r| r.outcome != "WorkLimit")
                .map(|r| r.traces.len())
                .sum();
            if coverage.is_full() {
                assert_eq!(applicable_labels, 76);
            }
            summary.finish(
                [all.len() - cap_rows - work_rows, cap_rows, work_rows],
                applicable_labels,
            );
            Ok(())
        }
        fn source_bits<R: Runtime<DType = DType>>(
            pair: &super::super::super::paired_sources::PairedSources<'_, R>,
        ) -> Result<Vec<Vec<[u64; 3]>>> {
            let p = pair.velocity.input();
            [
                &p.left_area,
                &p.left_discharge,
                &p.right_area,
                &p.right_discharge,
            ]
            .iter()
            .map(|p| frozen::components(p))
            .collect()
        }
        fn check_states(
            row: &Execution,
            owner: usize,
            actual: &[super::observed::State],
            complete: bool,
        ) -> Result {
            let owner = i64::try_from(owner)?;
            for expected in row.traces {
                let label = super::observed::label(expected.label);
                let matching = actual
                    .iter()
                    .filter(|a| a.owner == owner && a.label == label)
                    .collect::<Vec<_>>();
                if complete {
                    assert!(
                        !matching.is_empty(),
                        "{} lacks frozen sampled label {}",
                        row.id,
                        expected.label
                    );
                }
                for actual in matching {
                    compare(actual, expected, row.id);
                }
            }
            if let Some(star) = row.star {
                for actual in actual
                    .iter()
                    .filter(|a| a.owner == owner && a.label == "star")
                {
                    compare(actual, &star, row.id);
                }
            }
            Ok(())
        }
        fn compare(
            actual: &super::observed::State,
            expected: &super::fixture_model::State,
            id: &str,
        ) {
            super::observed::contains(actual.area, expected.area, id);
            super::observed::contains(actual.discharge, expected.discharge, id);
            if let Some(depth) = expected.depth {
                super::observed::contains(actual.depth, depth, id);
            }
            if let (Some(actual), Some(expected)) = (actual.velocity, expected.velocity) {
                super::observed::contains(actual, expected, id);
            }
        }
    }

    mod telemetry {
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
    }

    mod wave_check {
        use super::super::super::point_candidate_types::{PointCandidateCertificates, PointRegime};
        use super::{Result, fixture_model::Execution, frozen, wave_references::expected};
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
            for (expected, label) in expected(row) {
                let enclosed = available
                    .iter()
                    .any(|a| a[0] <= expected[0] && expected[1] <= a[1]);
                if complete {
                    assert!(
                        enclosed,
                        "{} {} {:?} lacks a supporting physical wave certificate {:?}",
                        row.id, label, expected, available
                    );
                }
            }
            Ok(())
        }
    }

    mod demand {
        use super::{PointRootSession, Result, chart_fixture, frozen, hydraulic, support};
        use numr::{
            dtype::DType,
            ops::BinaryOps,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::optimize::CertifiedRootTolerance;
        use std::collections::BTreeMap;

        type NativeSnapshot = (usize, usize, usize, bool, Vec<[f64; 2]>);
        type StarSnapshot = ([f64; 2], [f64; 2], [f64; 2], i64);
        type WaveSnapshot = ([f64; 5], i64);

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            selections::<R>(client)?;
            complete_preflight::<R>(client)
        }
        fn state<R: Runtime<DType = DType>>(
            session: &PointRootSession<'_, '_, R>,
        ) -> Result<Vec<NativeSnapshot>> {
            session
                .wet
                .domains
                .iter()
                .map(|d| {
                    Ok((
                        d.iterations,
                        d.callback_points,
                        d.mapped_points,
                        d.is_deferred(),
                        frozen::endpoints(&d.query.query.parameter)?,
                    ))
                })
                .collect()
        }
        fn star_rows<R: Runtime<DType = DType>>(
            domains: &[super::super::super::wet_star_types::WetStarDomain<R>],
        ) -> Result<BTreeMap<i64, StarSnapshot>> {
            let mut map = BTreeMap::new();
            for d in domains {
                let area = frozen::endpoints(&d.query.area)?;
                let u = frozen::endpoints(&d.velocity)?;
                let q = frozen::endpoints(&d.discharge)?;
                let status = d.status.try_to_vec::<i64>()?;
                for (i, owner) in d
                    .invocation_owner
                    .try_to_vec::<i64>()?
                    .into_iter()
                    .enumerate()
                {
                    assert!(
                        map.insert(owner, (area[i], u[i], q[i], status[i]))
                            .is_none()
                    );
                }
            }
            Ok(map)
        }
        fn wave_rows<R: Runtime<DType = DType>>(
            domains: &[super::super::super::wet_wave_types::WetWaveDomain<R>],
        ) -> Result<BTreeMap<i64, WaveSnapshot>> {
            let mut map = BTreeMap::new();
            for d in domains {
                let p = &d.regions;
                let fields = [
                    &p.left_exterior,
                    &p.left_fan,
                    &p.star,
                    &p.right_fan,
                    &p.right_exterior,
                ]
                .iter()
                .map(|t| t.try_to_vec::<f64>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
                let status = d.status.try_to_vec::<i64>()?;
                for (i, owner) in d
                    .star
                    .invocation_owner
                    .try_to_vec::<i64>()?
                    .into_iter()
                    .enumerate()
                {
                    assert!(
                        map.insert(
                            owner,
                            (
                                [
                                    fields[0][i],
                                    fields[1][i],
                                    fields[2][i],
                                    fields[3][i],
                                    fields[4][i]
                                ],
                                status[i]
                            )
                        )
                        .is_none()
                    );
                }
            }
            Ok(map)
        }
        fn selections<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                8.,
                &[[1., 0., 1., 0.], [4., 0., 1., 0.]],
                &config,
                |pair, work| {
                    let parents = support::rows::<R>(client, &[0, 0, 1, 1])?;
                    let mut session = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    hydraulic(session.start_wet(&support::rows::<R>(client, &[0, 1, 2, 3])?))?;
                    let before = state(&session)?;
                    let cache = chart_fixture::cache(session.sources())?;
                    let work_before = work.snapshot();
                    let empty = Tensor::zeros(&[4], DType::F64, client.device())?;
                    assert!(hydraulic(session.certify_wet_star_selected(&empty))?.is_empty());
                    assert!(hydraulic(session.certify_wet_waves_selected(&empty))?.is_empty());
                    let empty_owners = support::rows::<R>(client, &[])?;
                    let point = hydraulic(session.certify_point_candidates(&empty_owners))?;
                    assert!(point.candidates.is_empty() && point.waves.is_empty());
                    assert_eq!(point.status.numel(), 0);
                    assert_eq!(work.snapshot(), work_before);
                    assert_eq!(chart_fixture::cache(session.sources())?, cache);
                    assert_eq!(state(&session)?, before);
                    let full_stars = hydraulic(session.certify_wet_star())?;
                    let expected = star_rows(&full_stars)?;
                    let full_waves = hydraulic(session.certify_wet_waves())?;
                    let wave_expected = wave_rows(&full_waves)?;
                    let mask = Tensor::from_slice(&[1., 0., 0., 1.], &[4], client.device())?;
                    let selected = hydraulic(session.certify_wet_star_selected(&mask))?;
                    let selected_rows = star_rows(&selected)?;
                    assert_eq!(selected_rows.keys().copied().collect::<Vec<_>>(), [0, 3]);
                    for (&owner, value) in &selected_rows {
                        assert_eq!(Some(value), expected.get(&owner));
                    }
                    let selected_waves = hydraulic(session.certify_wet_waves_selected(&mask))?;
                    for (owner, value) in wave_rows(&selected_waves)? {
                        assert_eq!(Some(&value), wave_expected.get(&owner));
                    }
                    assert_eq!(state(&session)?, before);
                    assert_eq!(work.snapshot().residual_points, work_before.residual_points);
                    assert_eq!(work.snapshot().root_iterations, work_before.root_iterations);
                    for bad in [
                        Tensor::zeros(&[4], DType::I64, client.device())?,
                        Tensor::zeros(&[3], DType::F64, client.device())?,
                        Tensor::from_slice(&[0., 1., 2., 0.], &[4], client.device())?,
                    ] {
                        let before = work.snapshot();
                        let cache = chart_fixture::cache(session.sources())?;
                        assert!(session.certify_wet_star_selected(&bad).is_err());
                        assert!(session.certify_wet_waves_selected(&bad).is_err());
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_fixture::cache(session.sources())?, cache);
                    }
                    let all = Tensor::full_scalar(&[4], DType::F64, 1., client.device())?;
                    let owned = hydraulic(session.certify_wet_waves_selected(&all))?;
                    for w in &owned {
                        let changed = Tensor::full_scalar(
                            w.star.query.query.parameter.lower().shape(),
                            DType::F64,
                            99.,
                            client.device(),
                        )?;
                        client.copy_into(w.star.query.query.parameter.lower(), &changed)?;
                    }
                    assert_eq!(
                        state(&session)?,
                        before,
                        "Full-demand output aliases native wet continuation operands"
                    );
                    hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance {
                        atol: 1e-17,
                        rtol: 0.,
                    }))?;
                    assert!(session.callback_points() >= work_before.residual_points);
                    assert!(session.callback_points() <= config.root.max_points);
                    let refined = hydraulic(session.certify_wet_star_selected(&mask))?;
                    assert_eq!(star_rows(&refined)?.len(), 2);
                    Ok(())
                },
            )
        }
        fn complete_preflight<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            for mutate_owner in [true, false] {
                support::with_rectangle::<R, _>(
                    client,
                    1.,
                    8.,
                    &[[1., 0., 1., 0.], [4., 0., 1., 0.]],
                    &config,
                    |pair, work| {
                        let owners = support::rows::<R>(client, &[0, 1])?;
                        let mut session = hydraulic(PointRootSession::new(
                            pair,
                            &owners,
                            config.root.max_points,
                        ))?;
                        hydraulic(session.start_wet(&owners))?;
                        if mutate_owner {
                            session.wet.domains[0].invocation_owner =
                                support::rows::<R>(client, &[1])?;
                        } else {
                            session.wet.domains[0].parent_rows = support::rows::<R>(client, &[1])?;
                        }
                        let mask = Tensor::from_slice(&[0., 1.], &[2], client.device())?;
                        let before = work.snapshot();
                        let cache = chart_fixture::cache(session.sources())?;
                        assert!(session.certify_wet_star_selected(&mask).is_err());
                        assert!(session.certify_wet_waves_selected(&mask).is_err());
                        assert!(
                            session
                                .certify_point_candidates(&support::rows::<R>(client, &[1])?)
                                .is_err()
                        );
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_fixture::cache(session.sources())?, cache);
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }
    }

    mod membership {
        use super::super::super::{
            source_sonic_types::SonicSide, wet_candidates, wet_candidates_star, wet_fan_membership,
            wet_star_types::WetStarStatus, wet_wave_types::WetWaveDomain, wet_waves_endpoint,
            wet_waves_regions, wet_waves_side,
        };
        use super::{
            PointCandidate, PointRootSession, Result, chart_fixture, frozen, hydraulic, support,
        };
        use crate::channel::flow::surveyed::error::checked;
        use numr::{
            dtype::DType,
            ops::BinaryOps,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            // Exact rectangle: left invariant=3, G=3*sqrt(A)-3, and the right source is A=Q=1.
            support::with_rectangle::<R, _>(
                client,
                1.,
                8.,
                &[[4., -4., 1., 1.]],
                &config,
                |pair, work| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                    hydraulic(session.start_wet(&owners))?;
                    hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
                    let original_star = frozen::endpoints(&session.wet.domains[0].query.area)?[0];
                    let original_sonic =
                        frozen::endpoints(&session.sonic.requests[0].domains[0].query.area)?[0];
                    assert!(0.5 <= original_star[0] && original_star[1] <= 1.5);
                    assert!(0.75 <= original_sonic[0] && original_sonic[1] <= 1.25);
                    let star = chart_fixture::chart(
                        session.sources(),
                        chart_fixture::Anchor::Zero,
                        [1.0_f64.to_bits(), 0],
                        &[[0.5, 1.5]],
                    )?;
                    let sonic = chart_fixture::chart(
                        session.sources(),
                        chart_fixture::Anchor::Zero,
                        [1.0_f64.to_bits(), 0],
                        &[[0.75, 1.25]],
                    )?;
                    let before = work.snapshot();
                    let points = session.callback_points();
                    let level = hydraulic(session.sources.panel_level(0))?;
                    let star = hydraulic(wet_candidates_star::evaluate(
                        session.sources,
                        &owners,
                        &owners,
                        &star,
                        &level,
                    ))?;
                    assert_eq!(
                        star.status.try_to_vec::<i64>()?,
                        [WetStarStatus::Certified as i64]
                    );
                    let wave = hydraulic(checked::<R, _>(
                        client,
                        "independent broad root wave fixture",
                        |a| {
                            let selected = session.sources.select_rows(&owners, &star.query)?;
                            let sound =
                                wet_waves_endpoint::sounds(a, session.sources, &selected, &star)?;
                            let sides = [
                                wet_waves_side::evaluate(a, &selected, &star, &sound, 1., 0)?,
                                wet_waves_side::evaluate(a, &selected, &star, &sound, 1., 1)?,
                            ];
                            let regions = wet_waves_regions::evaluate(a, &sides)?;
                            let covered = client.maximum(
                                &regions.left_exterior,
                                &client.maximum(
                                    &regions.left_fan,
                                    &client.maximum(
                                        &regions.star,
                                        &client
                                            .maximum(&regions.right_fan, &regions.right_exterior)?,
                                    )?,
                                )?,
                            )?;
                            assert_eq!(covered.try_to_vec::<f64>()?, [1.]);
                            for side in &sides {
                                assert_eq!(side.valid.try_to_vec::<f64>()?, [1.]);
                            }
                            Ok(WetWaveDomain {
                                star,
                                endpoint_sound: sound,
                                sides,
                                regions,
                                status: Tensor::zeros(&[1], DType::I64, client.device())?,
                            })
                        },
                    ))?;
                    let tail = frozen::endpoints(&wave.sides[0].tail.physical)?[0];
                    assert!(tail[0] < 0. && tail[1] > 0.);
                    assert_eq!(wave.regions.left_fan.try_to_vec::<f64>()?, [1.]);
                    assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.]);
                    let proof = hydraulic(checked::<R, _>(
                        client,
                        "overlapping established roots retain uncertainty",
                        |a| {
                            wet_fan_membership::evaluate(
                                a,
                                session.sources(),
                                &wave,
                                &owners,
                                &sonic,
                                SonicSide::Left,
                            )
                        },
                    ))?;
                    assert_eq!(proof.proved.try_to_vec::<f64>()?, [0.]);
                    assert_eq!(proof.excluded.try_to_vec::<f64>()?, [0.]);
                    assert_eq!(proof.pending.try_to_vec::<f64>()?, [1.]);
                    assert_eq!(proof.coincidence.try_to_vec::<f64>()?, [0.]);
                    let difference = frozen::endpoints(&proof.separation.physical)?[0];
                    assert!(difference[0] < 0. && difference[1] > 0.);
                    let waves = [wave];
                    let pending = hydraulic(wet_candidates::evaluate(&mut session, &waves))?;
                    assert_eq!(pending.complete.try_to_vec::<f64>()?, [0.]);
                    assert_eq!(pending.covered.try_to_vec::<f64>()?, [1.]);
                    assert!(pending.candidates.iter().any(|c|matches!(c,PointCandidate::Pending(p) if p.region==super::super::super::point_candidate_types::PointRegion::LeftFan)));
                    assert!(
                        !pending
                            .candidates
                            .iter()
                            .any(|c| matches!(c, PointCandidate::Fan(_)))
                    );
                    assert_eq!(session.callback_points(), points);
                    assert_eq!(work.snapshot().residual_points, before.residual_points);
                    assert_eq!(work.snapshot().root_iterations, before.root_iterations);
                    assert_eq!(
                        frozen::endpoints(&session.wet.domains[0].query.area)?[0],
                        original_star
                    );
                    assert_eq!(
                        frozen::endpoints(&session.sonic.requests[0].domains[0].query.area)?[0],
                        original_sonic
                    );
                    hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance {
                        atol: 1e-17,
                        rtol: 0.,
                    }))?;
                    hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                        atol: 1e-17,
                        rtol: 0.,
                    }))?;
                    let resumed = hydraulic(session.certify_point_candidates(&owners))?;
                    assert_eq!(resumed.status.try_to_vec::<i64>()?, [0]);
                    super::diagnostics::check(&session, &resumed)?;
                    Ok(())
                },
            )
        }
    }

    mod ownership {
        use super::super::super::source_sonic_types::SonicSide;
        use super::{
            PointCandidate, PointRootSession, Result, chart_fixture, frozen, hydraulic, support,
        };
        use numr::{
            dtype::DType,
            ops::BinaryOps,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            returned_binding::<R>(client)?;
            for initialized in [false, true] {
                foreign_wet::<R>(client, initialized)?;
            }
            foreign_sonic::<R>(client)?;
            snapshots_and_policy::<R>(client)
        }
        fn returned_binding<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                4.,
                &[[0., 0., 0., 0.]],
                &config,
                |pair, work| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let output = {
                        let mut first = hydraulic(PointRootSession::new(pair, &owners, 1))?;
                        let r = hydraulic(first.certify_point_candidates(&owners))?;
                        hydraulic(first.invocation.check(&r.invocation_witness))?;
                        r
                    };
                    let second = hydraulic(PointRootSession::new(pair, &owners, 1))?;
                    let before = work.snapshot();
                    assert!(second.invocation.check(&output.invocation_witness).is_err());
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }
        fn foreign_wet<R>(client: &R::Client, initialized: bool) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            let state = if initialized {
                [4., 0., 1., 0.]
            } else {
                [1., 0., 1., 0.]
            };
            support::with_rectangle::<R, _>(client, 1., 8., &[state], &config, |pair, work| {
                let parents = support::rows::<R>(client, &[0, 0])?;
                let owners = support::rows::<R>(client, &[0])?;
                let donor = {
                    let mut first = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    hydraulic(first.start_wet(&owners))?;
                    assert_eq!(first.wet.domains[0].is_deferred(), !initialized);
                    first
                        .wet
                        .domains
                        .pop()
                        .ok_or("Missing foreign wet continuation")?
                };
                let mut second = hydraulic(PointRootSession::new(
                    pair,
                    &parents,
                    config.root.max_points,
                ))?;
                hydraulic(second.start_wet(&owners))?;
                second.wet.domains[0] = donor;
                let before = work.snapshot();
                let cache = chart_fixture::cache(second.sources())?;
                let empty = Tensor::zeros(&[2], DType::F64, client.device())?;
                assert!(second.certify_wet_star_selected(&empty).is_err());
                assert!(second.certify_wet_waves_selected(&empty).is_err());
                assert!(
                    second
                        .certify_point_candidates(&support::rows::<R>(client, &[1])?)
                        .is_err()
                );
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_fixture::cache(second.sources())?, cache);
                Ok(())
            })
        }
        fn foreign_sonic<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                4.,
                &[[1., 0., 0., 0.]],
                &config,
                |pair, work| {
                    let parents = support::rows::<R>(client, &[0, 0])?;
                    let owners = support::rows::<R>(client, &[0])?;
                    let donor = {
                        let mut first = hydraulic(PointRootSession::new(
                            pair,
                            &parents,
                            config.root.max_points,
                        ))?;
                        hydraulic(first.start_source_sonic(&owners, SonicSide::Left))?;
                        first
                            .sonic
                            .requests
                            .pop()
                            .ok_or("Missing foreign sonic request")?
                    };
                    let mut second = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    hydraulic(second.start_source_sonic(&owners, SonicSide::Left))?;
                    second.sonic.requests[0] = donor;
                    let before = work.snapshot();
                    let cache = chart_fixture::cache(second.sources())?;
                    assert!(
                        second
                            .certify_point_candidates(&support::rows::<R>(client, &[1])?)
                            .is_err()
                    );
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_fixture::cache(second.sources())?, cache);
                    Ok(())
                },
            )
        }
        fn snapshots_and_policy<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                8.,
                &[[4., 0., 1., 0.], [1., 0., 0., 0.]],
                &config,
                |pair, work| {
                    let parents = support::rows::<R>(client, &[0, 1, 0])?;
                    let owners = support::rows::<R>(client, &[2, 1, 0])?;
                    let mut session = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    let result = hydraulic(session.certify_point_candidates(&owners))?;
                    super::diagnostics::check(&session, &result)?;
                    let wet = session
                        .wet
                        .domains
                        .iter()
                        .map(|d| frozen::endpoints(&d.query.query.parameter))
                        .collect::<Result<Vec<_>>>()?;
                    let sonic = session
                        .sonic
                        .requests
                        .iter()
                        .flat_map(|r| &r.domains)
                        .map(|d| frozen::endpoints(&d.query.query.parameter))
                        .collect::<Result<Vec<_>>>()?;
                    for wave in &result.waves {
                        let change = Tensor::full_scalar(
                            wave.star.query.query.parameter.lower().shape(),
                            DType::F64,
                            99.,
                            client.device(),
                        )?;
                        client.copy_into(wave.star.query.query.parameter.lower(), &change)?;
                    }
                    for candidate in &result.candidates {
                        if let PointCandidate::Fan(fan) = candidate {
                            let change = Tensor::full_scalar(
                                fan.certificate.query.query.parameter.lower().shape(),
                                DType::F64,
                                99.,
                                client.device(),
                            )?;
                            client.copy_into(
                                fan.certificate.query.query.parameter.lower(),
                                &change,
                            )?;
                        }
                    }
                    let changed =
                        Tensor::full_scalar(owners.shape(), DType::I64, 99., client.device())?;
                    client.copy_into(&result.invocation_owner, &changed)?;
                    client.copy_into(&result.parent_rows, &changed)?;
                    client.copy_into(&result.wet_status, &changed)?;
                    assert_eq!(
                        session
                            .wet
                            .domains
                            .iter()
                            .map(|d| frozen::endpoints(&d.query.query.parameter))
                            .collect::<Result<Vec<_>>>()?,
                        wet
                    );
                    assert_eq!(
                        session
                            .sonic
                            .requests
                            .iter()
                            .flat_map(|r| &r.domains)
                            .map(|d| frozen::endpoints(&d.query.query.parameter))
                            .collect::<Result<Vec<_>>>()?,
                        sonic
                    );
                    let points = session.callback_points();
                    let before = work.snapshot().residual_points;
                    let repeat = hydraulic(session.certify_point_candidates(&owners))?;
                    assert_eq!(repeat.invocation_owner.try_to_vec::<i64>()?, [2, 1, 0]);
                    assert_eq!(repeat.status.try_to_vec::<i64>()?, [0; 3]);
                    assert_eq!(session.callback_points(), points);
                    assert_eq!(work.snapshot().residual_points, before);
                    for bad in [&[-1][..], &[3][..], &[0, 0][..]] {
                        let before = work.snapshot();
                        let cache = chart_fixture::cache(session.sources())?;
                        assert!(
                            session
                                .certify_point_candidates(&support::rows::<R>(client, bad)?)
                                .is_err()
                        );
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_fixture::cache(session.sources())?, cache);
                    }
                    session.sources.evaluation.panels *= 2;
                    let before = work.snapshot();
                    let cache = chart_fixture::cache(session.sources())?;
                    assert!(session.certify_point_candidates(&owners).is_err());
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_fixture::cache(session.sources())?, cache);
                    Ok(())
                },
            )
        }
    }

    mod precision {
        use super::super::super::{
            point_candidate_types::{PointCandidateStatus, PointRegime},
            source_sonic_types::SonicSide,
        };
        use super::{
            PointCandidate, PointFanAuthority, PointRootSession, Result, chart_fixture, fixtures,
            frozen, hydraulic, support,
        };
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            film::<R>(client)?;
            exterior_and_capacity::<R>(client)
        }
        fn film<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let reference = fixtures::executions()
                .into_iter()
                .find(|r| r.id == "retained_vacuum_inside_film")
                .ok_or("Missing independent film reference")?;
            let config = fixtures::CONFIGS[reference.config].build();
            chart_fixture::with_pair::<R, _>(client, &[reference.id], &config, |pair, work| {
                let cap = pair.bands.cap_depth.upper().try_to_vec::<f64>()?[0];
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let initial = hydraulic(session.certify_point_candidates(&owners))?;
                assert_eq!(
                    initial.regime.try_to_vec::<i64>()?,
                    [PointRegime::Wet as i64]
                );
                assert_eq!(
                    initial.status.try_to_vec::<i64>()?,
                    [PointCandidateStatus::Complete as i64]
                );
                let overlap = initial
                    .fan_evidence
                    .iter()
                    .find(|e| e.side == SonicSide::Right)
                    .ok_or("Film lacks initial right-fan evidence")?;
                assert_eq!(overlap.membership.pending.try_to_vec::<f64>()?, [0.]);
                assert_eq!(overlap.membership.proved.try_to_vec::<f64>()?, [1.]);
                let separation = frozen::endpoints(&overlap.membership.separation.physical)?[0];
                let tail = frozen::endpoints(&initial.waves[0].sides[1].tail.physical)?[0];
                assert!(tail[1] <= 0. || separation[1] <= 0.);
                eprintln!(
                    "POINT_FILM_INITIAL\tright_tail={tail:?}\tstar_minus_sonic={separation:?}"
                );
                let truth = reference.star.ok_or("Film lacks independent star truth")?;
                let atol = f64::from_bits(truth.area[0]) / cap * 1e-12;
                assert!(atol > 0. && atol < config.root.atol);
                hydraulic(
                    session.refine_wet_depth_in_place(CertifiedRootTolerance { atol, rtol: 0. }),
                )?;
                hydraulic(session.refine_source_sonic(CertifiedRootTolerance { atol, rtol: 0. }))?;
                let before = work.snapshot();
                let points = session.callback_points();
                let result = hydraulic(session.certify_point_candidates(&owners))?;
                assert_eq!(session.callback_points(), points);
                assert_eq!(work.snapshot().residual_points, before.residual_points);
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointCandidateStatus::Complete as i64]
                );
                super::diagnostics::check(&session, &result)?;
                let actual = super::observed::actual(&session, &result)?;
                assert!(actual.iter().all(|s| s.label == "right-sonic"));
                assert_eq!(actual.len(), 1);
                let expected = &reference.traces[0];
                let state = &actual[0];
                assert!(state.area[0] > f64::from_bits(truth.area[1]));
                for (actual, expected) in [
                    (state.area, expected.area),
                    (
                        state.depth,
                        expected.depth.ok_or("Film missing depth truth")?,
                    ),
                    (state.discharge, expected.discharge),
                    (
                        state.velocity.ok_or("Film missing sonic velocity")?,
                        expected
                            .velocity
                            .ok_or("Film missing independent velocity")?,
                    ),
                ] {
                    super::observed::contains(actual, expected, reference.id);
                    let scale = f64::from_bits(expected[0]).abs();
                    assert!(
                        (actual[1] - actual[0]) / scale < 1e-10,
                        "Film precision {actual:?}"
                    );
                }
                let fan = result
                    .candidates
                    .iter()
                    .find_map(|c| {
                        if let PointCandidate::Fan(f) = c {
                            Some(f)
                        } else {
                            None
                        }
                    })
                    .ok_or("Missing complete film sonic candidate")?;
                let PointFanAuthority::Wet(index) = fan.authority else {
                    return Err("Film fan lacks wet membership".into());
                };
                assert_eq!(
                    result.fan_evidence[index]
                        .membership
                        .proved
                        .try_to_vec::<f64>()?,
                    [1.]
                );
                assert_eq!(session.point_limit(), config.root.max_points);
                assert!(session.callback_points() <= session.point_limit());
                Ok(())
            })
        }
        fn exterior_and_capacity<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            for id in [
                "tiny_rectangle_compression",
                "uniform_cap_retained_deficit_dyadic_velocity",
            ] {
                let row = fixtures::executions()
                    .into_iter()
                    .find(|r| r.id == id)
                    .ok_or("Missing independent source-scale reference")?;
                let config = fixtures::CONFIGS[row.config].build();
                chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, _| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                    let result = hydraulic(session.certify_point_candidates(&owners))?;
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [PointCandidateStatus::Complete as i64]
                    );
                    super::diagnostics::check(&session, &result)?;
                    let actual = super::observed::actual(&session, &result)?;
                    let expected = &row.traces[0];
                    assert!(!actual.is_empty());
                    for state in actual {
                        assert_eq!(state.label, expected.label);
                        for (actual, expected) in [
                            (state.area, expected.area),
                            (state.discharge, expected.discharge),
                        ] {
                            super::observed::contains(actual, expected, id);
                            assert!(
                                (actual[1] - actual[0]) / f64::from_bits(expected[0]).abs() < 1e-10,
                                "{id} source-scale gate {actual:?}"
                            );
                        }
                    }
                    if id == "uniform_cap_retained_deficit_dyadic_velocity" {
                        let source = frozen::cases()
                            .iter()
                            .find(|r| r.id == id)
                            .ok_or("Missing captured source")?;
                        for wave in &result.waves {
                            for endpoint in &wave.star.endpoints {
                                for side in &endpoint.sides {
                                    assert_eq!(
                                        frozen::components(&side.source_discharge)?,
                                        [source.inputs[1]]
                                    );
                                }
                            }
                        }
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }
    }

    mod shocks {
        use super::super::super::source_sonic_types::SourceSonicStatus;
        use super::{
            PointCandidate, PointRootSession, Result, chart_fixture, frozen, hydraulic, support,
        };
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            for weak_change in [false, true] {
                let mut config = chart_fixture::config();
                config.gravity_m_per_s2 = 3.;
                let section = chart_fixture::rectangle::<R>(client, 1., 4.)?;
                let owners = support::rows::<R>(client, &[0])?;
                let low = if weak_change {
                    2.0_f64.powi(-80).to_bits()
                } else {
                    0
                };
                let inputs = [
                    [1.0_f64.to_bits(), 0, 0],
                    [3.0_f64.to_bits(), low, 0],
                    [2.0_f64.to_bits(), 0, 0],
                    [3.0_f64.to_bits(), low, 0],
                ]
                .iter()
                .map(|p| frozen::expansion::<R>(client, &[*p], true))
                .collect::<Result<Vec<_>>>()?;
                // At Q=3 both source momenta equal21/2. Q=3+e gives their difference3e+e²/2>0.
                chart_fixture::with_inputs(
                    client,
                    section,
                    owners,
                    inputs,
                    &config,
                    |pair, work| {
                        let owners = support::rows::<R>(client, &[0])?;
                        let mut session = hydraulic(PointRootSession::new(
                            pair,
                            &owners,
                            config.root.max_points,
                        ))?;
                        let result = hydraulic(session.certify_point_candidates(&owners))?;
                        assert_eq!(result.status.try_to_vec::<i64>()?, [0]);
                        super::diagnostics::check(&session, &result)?;
                        let wave = &result.waves[0];
                        let shock = &wave.sides[0];
                        assert_eq!(shock.shock_actual.try_to_vec::<f64>()?, [1.]);
                        assert_eq!(shock.shock_speed.available.try_to_vec::<f64>()?, [1.]);
                        let speed = frozen::endpoints(&shock.shock_speed.physical)?[0];
                        assert!(
                            speed[0] <= 0. && speed[1] >= 0.,
                            "Ambiguous left shock loses adjacent alternatives: {speed:?}"
                        );
                        assert_eq!(wave.regions.left_exterior.try_to_vec::<f64>()?, [1.]);
                        assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.]);
                        assert!(
                            result
                                .candidates
                                .iter()
                                .any(|c| matches!(c, PointCandidate::Exterior(_)))
                        );
                        assert!(
                            result
                                .candidates
                                .iter()
                                .any(|c| matches!(c, PointCandidate::Star(_)))
                        );
                        assert!(
                            !result
                                .candidates
                                .iter()
                                .any(|c| matches!(c, PointCandidate::Fan(_)))
                        );
                        if weak_change {
                            assert_eq!(shock.shock_stationary.try_to_vec::<f64>()?, [0.]);
                        } else if shock.shock_stationary.try_to_vec::<f64>()? == [1.] {
                            assert_eq!(speed, [0., 0.]);
                        }
                        for endpoint in &wave.star.endpoints {
                            for side in &endpoint.sides {
                                assert_eq!(
                                    frozen::components(&side.source_discharge)?,
                                    [[3.0_f64.to_bits(), low, 0]]
                                );
                            }
                        }
                        for status in &result.sonic_status {
                            assert_eq!(
                                status.try_to_vec::<i64>()?,
                                [SourceSonicStatus::NotRequested as i64]
                            );
                        }
                        let before = work.snapshot();
                        let points = session.callback_points();
                        let again = hydraulic(session.certify_point_candidates(&owners))?;
                        assert_eq!(again.status.try_to_vec::<i64>()?, [0]);
                        assert_eq!(session.callback_points(), points);
                        assert_eq!(work.snapshot().residual_points, before.residual_points);
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }
    }

    mod work {
        use super::super::super::{
            point_candidate_types::{PointCandidateStatus, PointRegime},
            source_sonic_types::{SonicSide, SourceSonicStatus},
        };
        use super::{PointRootSession, Result, chart_fixture, hydraulic, support};
        use crate::channel::flow::surveyed::error::{HydraulicError, PrismaticRiemannRejection};
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            shared_ceiling::<R>(client)?;
            callback_error::<R>(client)?;
            wet_error::<R>(client)
        }
        fn shared_ceiling<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                4.,
                &[[1., 0., 0., 0.], [0., 0., 0., 0.]],
                &config,
                |pair, work| {
                    let mut parents = vec![0_i64; 100];
                    parents[99] = 1;
                    let parents = support::rows::<R>(client, &parents)?;
                    let mut session = hydraulic(PointRootSession::new(pair, &parents, 2))?;
                    let first = support::rows::<R>(client, &[73])?;
                    hydraulic(session.start_source_sonic(&first, SonicSide::Left))?;
                    assert_eq!(session.callback_points(), 2);
                    assert_eq!(session.point_limit(), 2);
                    let before = work.snapshot();
                    let result = hydraulic(
                        session.certify_point_candidates(&support::rows::<R>(client, &[74])?),
                    )?;
                    assert_eq!(
                        result.sonic_status[0].try_to_vec::<i64>()?,
                        [SourceSonicStatus::WorkLimit as i64]
                    );
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [PointCandidateStatus::Pending as i64]
                    );
                    assert_eq!(session.callback_points(), 2);
                    assert_eq!(work.snapshot(), before);
                    let again = hydraulic(
                        session.certify_point_candidates(&support::rows::<R>(client, &[74])?),
                    )?;
                    assert_eq!(
                        again.status.try_to_vec::<i64>()?,
                        [PointCandidateStatus::Pending as i64]
                    );
                    assert_eq!(session.callback_points(), 2);
                    assert_eq!(work.snapshot(), before);
                    let dry = hydraulic(
                        session.certify_point_candidates(&support::rows::<R>(client, &[99])?),
                    )?;
                    assert_eq!(
                        dry.regime.try_to_vec::<i64>()?,
                        [PointRegime::BothDry as i64]
                    );
                    assert_eq!(
                        dry.status.try_to_vec::<i64>()?,
                        [PointCandidateStatus::Complete as i64]
                    );
                    assert_eq!(work.snapshot(), before);
                    assert!(session.start_source_sonic(&first, SonicSide::Left).is_err());
                    assert_eq!(session.callback_points(), 2);
                    Ok(())
                },
            )
        }
        fn callback_error<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = chart_fixture::config();
            config.max_native_query_points = 6;
            support::with_rectangle::<R, _>(
                client,
                1.,
                4.,
                &[[1., 0., 0., 0.]],
                &config,
                |pair, work| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let parents = support::rows::<R>(client, &[0, 0])?;
                    let mut session = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    assert!(matches!(
                        session.certify_point_candidates(&owners),
                        Err(HydraulicError::Rejected {
                            reason: PrismaticRiemannRejection::WorkLimit,
                            ..
                        })
                    ));
                    assert_eq!(session.callback_points(), 6);
                    assert_eq!(work.snapshot().residual_points, 6);
                    assert!(session.sonic.requests[0].terminal);
                    let before = work.snapshot();
                    let cache = chart_fixture::cache(session.sources())?;
                    assert!(session.certify_point_candidates(&owners).is_err());
                    assert!(
                        session
                            .certify_point_candidates(&support::rows::<R>(client, &[1])?)
                            .is_err()
                    );
                    assert_eq!(session.callback_points(), 6);
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_fixture::cache(session.sources())?, cache);
                    Ok(())
                },
            )
        }
        fn wet_error<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = chart_fixture::config();
            config.max_native_query_points = 3;
            support::with_rectangle::<R, _>(
                client,
                1.,
                3.,
                &[[1., 1., 1., 1.]],
                &config,
                |pair, work| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                    hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
                    hydraulic(session.start_wet(&owners))?;
                    assert_eq!(session.callback_points(), 3);
                    assert!(
                        session
                            .refine_wet_depth_in_place(CertifiedRootTolerance {
                                atol: 1e-20,
                                rtol: 0.
                            })
                            .is_err()
                    );
                    assert!(session.wet.terminal);
                    let before = work.snapshot();
                    let cache = chart_fixture::cache(session.sources())?;
                    assert!(session.certify_point_candidates(&owners).is_err());
                    assert_eq!(session.callback_points(), 3);
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_fixture::cache(session.sources())?, cache);
                    Ok(())
                },
            )
        }
    }

    mod fixture_export {
        use super::{
            Result,
            fixture_model::{Execution, State},
            fixtures, frozen,
        };
        use std::collections::BTreeSet;

        fn optional(value: Option<[u64; 2]>) -> String {
            value.map_or_else(|| "-".into(), |v| format!("{v:?}"))
        }
        fn state(id: &str, role: &str, index: usize, value: &State) {
            eprintln!(
                "POINT_STATE\t{id}\t{role}\t{index}\t{}\t{:?}\t{}\t{}\t{:?}\t{}",
                value.label,
                value.area,
                optional(value.depth),
                optional(value.velocity),
                value.discharge,
                value.velocity_provenance
            );
        }
        fn emit(rows: &[Execution]) -> Result {
            let mut ids = BTreeSet::new();
            for (owner, (id, cap, coordinates)) in frozen::profiles().enumerate() {
                eprintln!("POINT_PROFILE\t{owner}\t{id}\t{cap}\t{coordinates:?}");
            }
            for (i, cfg) in fixtures::CONFIGS.iter().enumerate() {
                eprintln!("POINT_CONFIG\t{i}\t{:?}\t{:?}", cfg.floats, cfg.counts);
            }
            for row in rows {
                assert!(ids.insert(row.id), "Repeated execution identity {}", row.id);
                let source = frozen::cases()
                    .iter()
                    .find(|c| c.id == row.id)
                    .ok_or("Missing captured source fixture identity")?;
                eprintln!(
                    "POINT_ROW\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    row.id,
                    row.config,
                    row.source_group,
                    row.group,
                    row.declared,
                    row.outcome,
                    row.physical,
                    row.config_identity,
                    row.expected_identity
                );
                eprintln!(
                    "POINT_INPUT\t{}\t{}\t{:?}",
                    row.id, source.owner, source.inputs
                );
                for (i, trace) in row.traces.iter().enumerate() {
                    state(row.id, "trace", i, trace);
                }
                if let Some(star) = row.star {
                    state(row.id, "star", 0, &star);
                }
                for (i, edge) in row.edges.iter().enumerate() {
                    eprintln!(
                        "POINT_EDGE\t{}\t{i}\t{}\t{:?}",
                        row.id, edge.label, edge.bounds
                    );
                }
            }
            assert_eq!(ids.len(), 70);
            Ok(())
        }

        #[test]
        fn point_candidate_reference_fixture_export() -> Result {
            emit(&fixtures::executions())
        }
    }

    mod flux_fixture_export {
        use super::{fixtures, flux_fixture_model::Bounds, flux_fixtures};
        use std::collections::BTreeMap;

        fn pair(value: Option<Bounds>) -> String {
            value.map_or_else(
                || "-".into(),
                |[lower, upper]| format!("{lower:016x},{upper:016x}"),
            )
        }

        fn emit() -> Result<(), Box<dyn std::error::Error>> {
            let executions = fixtures::executions();
            let mut by_id = BTreeMap::new();
            for execution in &executions {
                if by_id.insert(execution.id, execution).is_some() {
                    return Err(format!(
                        "Repeated execution identity {}. Check candidate fixtures",
                        execution.id
                    )
                    .into());
                }
            }
            let mut emitted = BTreeMap::new();
            for reference in flux_fixtures::REFERENCES {
                if emitted.insert(reference.id, ()).is_some() {
                    return Err(format!(
                        "Repeated flux identity {}. Check flux fixtures",
                        reference.id
                    )
                    .into());
                }
                let execution = by_id.get(reference.id).ok_or_else(|| {
                    format!(
                        "Missing execution identity {}. Check candidate fixtures",
                        reference.id
                    )
                })?;
                let config = fixtures::CONFIGS.get(execution.config).ok_or_else(|| {
                    format!(
                        "Missing config {} for {}. Check candidate fixtures",
                        execution.config, reference.id
                    )
                })?;
                eprintln!(
                    "POINT_FLUX_REFERENCE\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    reference.id,
                    reference.source_group,
                    reference.physical_group,
                    reference.outcome,
                    reference.config_identity,
                    reference.expected_identity,
                    reference.truth_identity,
                    reference.flux_gate_identity,
                    reference.water_identity,
                    reference.momentum_identity,
                    pair(reference.water),
                    pair(reference.momentum),
                    reference.positive_water_required,
                );
                eprintln!(
                    "POINT_FLUX_JOIN\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    execution.id,
                    execution.config,
                    execution.source_group,
                    execution.group,
                    execution.declared,
                    execution.outcome,
                    execution.physical,
                    execution.config_identity,
                    execution.expected_identity,
                );
                eprintln!(
                    "POINT_FLUX_CONFIG\t{}\t{:?}\t{:?}",
                    execution.id, config.floats, config.counts
                );
                for (index, declaration) in reference.quantization.iter().enumerate() {
                    eprintln!(
                        "POINT_FLUX_QUANTIZATION\t{}\t{index}\t{}\t{}\t{}\t{}\t{}\t{}",
                        reference.id,
                        declaration.quantity,
                        declaration.exact_rational,
                        pair(Some(declaration.projected)),
                        declaration.maximum_projected_width_rational,
                        declaration.positive_water_not_dry,
                        declaration.declaration_identity,
                    );
                }
            }
            if emitted.len() != by_id.len() {
                return Err(
                    "Flux identities omit candidate executions. Check both fixture identity sets"
                        .into(),
                );
            }
            Ok(())
        }

        #[test]
        fn point_flux_reference_fixture_export() -> Result<(), Box<dyn std::error::Error>> {
            emit()
        }
    }

    #[test]
    fn configured_cpu_point_candidates_preserve_all_regime_alternatives() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        matrix::<CpuRuntime>(&client, Coverage::Full)?;
        contacts::run::<CpuRuntime>(&client)?;
        precision::run::<CpuRuntime>(&client)?;
        demand::run::<CpuRuntime>(&client)?;
        membership::run::<CpuRuntime>(&client)?;
        ownership::run::<CpuRuntime>(&client)?;
        work::run::<CpuRuntime>(&client)?;
        shocks::run::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_candidates_preserve_all_regime_alternatives() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
    }
}
