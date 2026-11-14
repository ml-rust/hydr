use super::super::super::{
    point_candidate_types::{
        PointCandidate, PointCandidateCertificates, PointCandidateStatus, PointFanAuthority,
        PointPendingReason, PointRegime, PointRegion,
    },
    point_root_session::PointRootSession,
};
use super::{Result, frozen, hydraulic, support};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn check<R>(
    session: &PointRootSession<'_, '_, R>,
    result: &PointCandidateCertificates<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    hydraulic(session.invocation.check(&result.invocation_witness))?;
    let owners = result.invocation_owner.try_to_vec::<i64>()?;
    let parents = result.parent_rows.try_to_vec::<i64>()?;
    let captured = session.invocation_rows.try_to_vec::<i64>()?;
    let statuses = result.status.try_to_vec::<i64>()?;
    for (&owner, &parent) in owners.iter().zip(&parents) {
        assert_eq!(captured[usize::try_from(owner)?], parent);
    }
    assert_eq!(result.callback_points, session.callback_points());
    assert_eq!(result.point_limit, session.point_limit());
    assert_eq!(result.wet_status.numel(), owners.len());
    assert_eq!(result.depth_status.numel(), owners.len());
    for state in &result.sonic_status {
        assert_eq!(state.numel(), owners.len());
    }
    assert_eq!(
        result.regime.try_to_vec::<i64>()?,
        result.dry_vacuum.regime.try_to_vec::<i64>()?
    );
    let regimes = [
        PointRegime::BothDry,
        PointRegime::LeftWet,
        PointRegime::RightWet,
        PointRegime::Vacuum,
        PointRegime::Touching,
        PointRegime::Wet,
        PointRegime::Unresolved,
    ]
    .map(|v| v as i64);
    for value in result.regime.try_to_vec::<i64>()? {
        assert!(regimes.contains(&value));
    }
    for (i, status) in result
        .dry_vacuum
        .status
        .try_to_vec::<i64>()?
        .into_iter()
        .enumerate()
    {
        if status == 3 {
            assert_eq!(statuses[i], PointCandidateStatus::RegimeUnresolved as i64);
        }
    }
    let inv = frozen::endpoints(&result.dry_vacuum.invariant.physical)?;
    let inv_available = result.dry_vacuum.invariant.available.try_to_vec::<f64>()?;
    let regime = result.regime.try_to_vec::<i64>()?;
    for (i, bounds) in inv.into_iter().enumerate() {
        if regime[i] == PointRegime::Wet as i64 {
            assert_eq!(inv_available[i], 1.);
            assert!(bounds[1] < 0.);
        }
        if regime[i] == PointRegime::Vacuum as i64 {
            assert_eq!(inv_available[i], 1.);
            assert!(bounds[0] > 0.);
        }
        if regime[i] == PointRegime::Touching as i64 {
            assert_eq!(
                frozen::components(&result.dry_vacuum.invariant.retained)?[i],
                [0; 3]
            );
        }
    }
    let p = &result.dry_vacuum.possibilities;
    for mask in [
        &p.left_exterior,
        &p.left_fan,
        &p.dry,
        &p.right_fan,
        &p.right_exterior,
    ] {
        for value in mask.try_to_vec::<f64>()? {
            assert!(value == 0. || value == 1.);
        }
    }
    for side in &result.dry_vacuum.edges {
        for (value, exact) in [
            (&side.head, &side.head_exact_zero),
            (&side.front, &side.front_exact_zero),
        ] {
            let retained = frozen::components(&value.retained)?;
            let available = value.available.try_to_vec::<f64>()?;
            for (i, exact) in exact.try_to_vec::<f64>()?.into_iter().enumerate() {
                if exact == 1. {
                    assert_eq!(available[i], 1.);
                    assert_eq!(retained[i], [0; 3]);
                }
            }
        }
    }
    for evidence in &result.wet_evidence {
        let rows = evidence.parent_rows.try_to_vec::<i64>()?;
        let labels = evidence.invocation_owner.try_to_vec::<i64>()?;
        assert_eq!(rows.len(), labels.len());
        for (&owner, &parent) in labels.iter().zip(&rows) {
            assert_eq!(captured[usize::try_from(owner)?], parent);
            assert!(owners.contains(&owner));
        }
        let physical = frozen::endpoints(&evidence.value.physical)?;
        let available = evidence.value.available.try_to_vec::<f64>()?;
        for (i, bounds) in physical.into_iter().enumerate() {
            if available[i] == 1. {
                assert!(bounds[0].is_finite() && bounds[1].is_finite() && bounds[0] <= bounds[1]);
            }
        }
    }
    for evidence in &result.fan_evidence {
        let labels = evidence.invocation_owner.try_to_vec::<i64>()?;
        let rows = evidence.parent_rows.try_to_vec::<i64>()?;
        for (&owner, &parent) in labels.iter().zip(&rows) {
            assert_eq!(captured[usize::try_from(owner)?], parent);
        }
        let proof = &evidence.membership;
        let yes = proof.proved.try_to_vec::<f64>()?;
        let no = proof.excluded.try_to_vec::<f64>()?;
        let pending = proof.pending.try_to_vec::<f64>()?;
        for i in 0..labels.len() {
            assert_eq!(yes[i] + no[i] + pending[i], 1.);
        }
        let lower = frozen::components(&proof.lower_difference)?;
        let upper = frozen::components(&proof.upper_difference)?;
        for (i, coincident) in proof
            .coincidence
            .try_to_vec::<f64>()?
            .into_iter()
            .enumerate()
        {
            if coincident == 1. {
                assert_eq!(lower[i], [0; 3]);
                assert_eq!(upper[i], [0; 3]);
            }
        }
        let separation = frozen::endpoints(&proof.separation.physical)?;
        for (i, &owner) in labels.iter().enumerate() {
            let (wave, r) = result
                .waves
                .iter()
                .find_map(|w| {
                    w.star
                        .invocation_owner
                        .try_to_vec::<i64>()
                        .ok()?
                        .iter()
                        .position(|&o| o == owner)
                        .map(|r| (w, r))
                })
                .ok_or("Fan proof lacks its wet-wave owner")?;
            let side = &wave.sides[evidence.side.index()];
            let definite = side.rarefaction_possible.try_to_vec::<f64>()?[r] == 1.
                && side.shock_possible.try_to_vec::<f64>()?[r] == 0.
                && side.equality_possible.try_to_vec::<f64>()?[r] == 0.
                && side.tail.available.try_to_vec::<f64>()?[r] == 1.;
            let tail = frozen::endpoints(&side.tail.physical)?[r];
            let left = evidence.side == super::super::super::source_sonic_types::SonicSide::Left;
            if yes[i] == 1. {
                assert!(
                    separation[i][1] <= 0.
                        || definite && if left { tail[0] >= 0. } else { tail[1] <= 0. }
                );
            }
            if no[i] == 1. {
                assert!(
                    separation[i][0] > 0.
                        || definite && if left { tail[1] < 0. } else { tail[0] > 0. }
                );
            }
        }

        for (i, valid) in proof
            .separation
            .available
            .try_to_vec::<f64>()?
            .into_iter()
            .enumerate()
        {
            if valid == 1. {
                assert!(separation[i][0] <= separation[i][1]);
            }
        }
        assert_eq!(evidence.physical_status.numel(), labels.len());
    }
    for candidate in &result.candidates {
        match candidate {
            PointCandidate::Exterior(e) => {
                hydraulic(session.sources.select_rows(&e.parent_rows, &e.query))?;
                assert_eq!(e.recovery.owners().numel(), e.invocation_owner.numel());
            }
            PointCandidate::Dry(d) => {
                hydraulic(session.sources.select_rows(&d.parent_rows, &d.query))?;
                assert!(frozen::components(&d.area)?.iter().all(|v| *v == [0; 3]));
                assert!(
                    frozen::components(&d.discharge)?
                        .iter()
                        .all(|v| *v == [0; 3])
                );
            }
            PointCandidate::Star(s) => {
                let wave = &result.waves[s.wave];
                let rows = s.rows.try_to_vec::<i64>()?;
                let labels = s.invocation_owner.try_to_vec::<i64>()?;
                let parents = s.parent_rows.try_to_vec::<i64>()?;
                let wo = wave.star.invocation_owner.try_to_vec::<i64>()?;
                let wp = wave.star.parent_rows.try_to_vec::<i64>()?;
                let possible = wave.regions.star.try_to_vec::<f64>()?;
                for (i, row) in rows.into_iter().enumerate() {
                    let r = usize::try_from(row)?;
                    assert_eq!(wo[r], labels[i]);
                    assert_eq!(wp[r], parents[i]);
                    assert_eq!(possible[r], 1.);
                }
            }
            PointCandidate::Fan(f) => {
                hydraulic(
                    session
                        .sources
                        .select_rows(&f.certificate.parent_rows, &f.certificate.query),
                )?;
                let status = f.certificate.status.try_to_vec::<i64>()?;
                for (i, active) in f.active.try_to_vec::<f64>()?.into_iter().enumerate() {
                    if active == 1. {
                        assert_eq!(status[i], 0);
                    }
                }
                if let PointFanAuthority::Wet(index) = f.authority {
                    let e = &result.fan_evidence[index];
                    assert_eq!(e.side, f.certificate.side);
                    assert_eq!(
                        e.invocation_owner.try_to_vec::<i64>()?,
                        f.certificate.invocation_owner.try_to_vec::<i64>()?
                    );
                    for (active, proved) in f
                        .active
                        .try_to_vec::<f64>()?
                        .into_iter()
                        .zip(e.membership.proved.try_to_vec::<f64>()?)
                    {
                        if active == 1. {
                            assert_eq!(proved, 1.);
                        }
                    }
                }
            }
            PointCandidate::Pending(p) => {
                let rows = p.parent_rows.try_to_vec::<i64>()?;
                for (i, owner) in p
                    .invocation_owner
                    .try_to_vec::<i64>()?
                    .into_iter()
                    .enumerate()
                {
                    let r = owners
                        .iter()
                        .position(|&o| o == owner)
                        .ok_or("Pending owner escapes requested rows")?;
                    assert_ne!(statuses[r], PointCandidateStatus::Complete as i64);
                    assert_eq!(rows[i], parents[r]);
                    match p.reason {
                        PointPendingReason::WetRoot => {
                            assert_eq!(p.region, PointRegion::Unlocated);
                            assert!(!result.waves.iter().any(|w| {
                                w.star
                                    .invocation_owner
                                    .try_to_vec::<i64>()
                                    .is_ok_and(|v| v.contains(&owner))
                            }));
                        }
                        PointPendingReason::WavePositions => {
                            assert_eq!(p.region, PointRegion::Unlocated)
                        }
                        PointPendingReason::StarState => assert_eq!(p.region, PointRegion::Star),
                        PointPendingReason::FanCertificate => {
                            let side = match p.region {
                                PointRegion::LeftFan => 0,
                                PointRegion::RightFan => 1,
                                _ => return Err("Pending fan loses its source side".into()),
                            };
                            assert_ne!(result.sonic_status[side].try_to_vec::<i64>()?[r],super::super::super::source_sonic_types::SourceSonicStatus::NotRequested as i64);
                        }
                    }
                }
            }
        }
    }
    let actual = super::observed::actual(session, result)?;
    for (i, &owner) in owners.iter().enumerate() {
        if statuses[i] == PointCandidateStatus::Complete as i64 {
            assert!(
                actual.iter().any(|s| s.owner == owner),
                "Complete owner {owner} lacks a physical candidate"
            );
        }
    }
    Ok(())
}
