#[cfg(test)]
use super::point_candidate_types::PointFanEvidence;
use super::{
    point_candidate_types::{
        PointCandidate, PointPending, PointPendingReason, PointRegion, PointStar,
    },
    point_root_session::PointRootSession,
    wet_wave_types::WetWaveDomain,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult, checked};
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

pub(super) struct WetCandidates<R: Runtime<DType = DType>> {
    pub(super) candidates: Vec<PointCandidate<R>>,
    #[cfg(test)]
    pub(super) evidence: Vec<PointFanEvidence<R>>,
    pub(super) complete: Tensor<R>,
    pub(super) covered: Tensor<R>,
    pub(super) located: Tensor<R>,
}
pub(super) fn evaluate<R>(
    session: &mut PointRootSession<'_, '_, R>,
    waves: &[WetWaveDomain<R>],
) -> HydraulicResult<WetCandidates<R>>
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
    let c = session.sources.geometry.client();
    let (mut complete, mut covered) =
        checked::<R, _>(c, "wet candidate invocation coverage storage", |a| {
            Ok((
                Tensor::zeros(session.invocation_rows.shape(), a.dtype(), c.device())?,
                Tensor::zeros(session.invocation_rows.shape(), a.dtype(), c.device())?,
            ))
        })?;
    let mut located = covered.clone();
    let mut candidates = Vec::new();
    #[cfg(test)]
    let mut evidence = Vec::new();
    for (index, wave) in waves.iter().enumerate() {
        let (mut valid, mut coverage) =
            checked::<R, _>(c, "wet candidate physical wave prerequisites", |a| {
                let owners = &wave.star.invocation_owner;
                let parent = &wave.star.parent_rows;
                let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
                let one = Tensor::full_scalar(owners.shape(), a.dtype(), 1., c.device())?;
                let statuszero = Tensor::zeros(&[], DType::I64, c.device())?;
                let valid = c.where_cond(&c.eq(&wave.status, &statuszero)?, &one, &zero)?;
                pending(
                    a,
                    &mut candidates,
                    owners,
                    parent,
                    &c.sub(&one, &valid)?,
                    PointRegion::Unlocated,
                    PointPendingReason::WavePositions,
                )?;
                let starvalid =
                    c.where_cond(&c.eq(&wave.star.status, &statuszero)?, &one, &zero)?;
                pending(
                    a,
                    &mut candidates,
                    owners,
                    parent,
                    &c.sub(&one, &starvalid)?,
                    PointRegion::Star,
                    PointPendingReason::StarState,
                )?;
                for (mask, side) in [
                    (
                        &wave.regions.left_exterior,
                        super::source_sonic_types::SonicSide::Left,
                    ),
                    (
                        &wave.regions.right_exterior,
                        super::source_sonic_types::SonicSide::Right,
                    ),
                ] {
                    let rows = c.nonzero(mask)?;
                    let rows = rows.reshape(&[rows.numel()])?;
                    if rows.numel() != 0 {
                        candidates.push(PointCandidate::Exterior(Box::new(
                            super::point_candidates_source::exterior(
                                a,
                                session.sources,
                                &c.index_select(owners, 0, &rows)?,
                                &c.index_select(parent, 0, &rows)?,
                                side,
                            )?,
                        )));
                    }
                }
                let rows = c.nonzero(&wave.regions.star)?;
                let rows = rows.reshape(&[rows.numel()])?;
                if rows.numel() != 0 {
                    candidates.push(PointCandidate::Star(PointStar {
                        wave: index,
                        invocation_owner: c.index_select(owners, 0, &rows)?,
                        parent_rows: c.index_select(parent, 0, &rows)?,
                        rows,
                    }));
                }
                let coverage = c.maximum(
                    &wave.regions.left_exterior,
                    &c.maximum(&wave.regions.star, &wave.regions.right_exterior)?,
                )?;
                Ok((
                    c.mul(&valid, &c.where_cond(&wave.regions.star, &starvalid, &one)?)?,
                    coverage,
                ))
            })?;
        for side in [
            super::source_sonic_types::SonicSide::Left,
            super::source_sonic_types::SonicSide::Right,
        ] {
            let resolved = super::wet_candidates_fans::evaluate(
                session,
                wave,
                side,
                &mut candidates,
                #[cfg(test)]
                &mut evidence,
            )?;
            valid = c.mul(&valid, &resolved.resolved)?;
            coverage = c.maximum(&coverage, &resolved.active)?;
        }
        checked::<R, _>(c, "wet candidate nonempty physical coverage", |a| {
            let zero = Tensor::zeros(coverage.shape(), a.dtype(), c.device())?;
            pending(
                a,
                &mut candidates,
                &wave.star.invocation_owner,
                &wave.star.parent_rows,
                &c.eq(&coverage, &zero)?,
                PointRegion::Unlocated,
                PointPendingReason::WavePositions,
            )?;
            located = c.index_put(
                &located,
                0,
                &wave.star.invocation_owner,
                &Tensor::full_scalar(coverage.shape(), a.dtype(), 1., c.device())?,
            )?;
            Ok(())
        })?;
        covered = c.index_put(&covered, 0, &wave.star.invocation_owner, &coverage)?;
        complete = c.index_put(
            &complete,
            0,
            &wave.star.invocation_owner,
            &c.mul(&valid, &coverage)?,
        )?;
    }
    Ok(WetCandidates {
        candidates,
        #[cfg(test)]
        evidence,
        complete,
        covered,
        located,
    })
}

pub(super) fn pending<R>(
    a: &Arithmetic<'_, R>,
    output: &mut Vec<PointCandidate<R>>,
    owners: &Tensor<R>,
    parent: &Tensor<R>,
    mask: &Tensor<R>,
    region: PointRegion,
    reason: PointPendingReason,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let rows = c.nonzero(mask)?;
    let rows = rows.reshape(&[rows.numel()])?;
    if rows.numel() != 0 {
        output.push(PointCandidate::Pending(PointPending {
            invocation_owner: c.index_select(owners, 0, &rows)?,
            parent_rows: c.index_select(parent, 0, &rows)?,
            region,
            reason,
        }));
    }
    Ok(())
}
