#[cfg(test)]
use super::point_candidate_types::{PointFanAuthority, PointFanEvidence};
use super::{
    point_candidate_types::{PointCandidate, PointFan, PointPendingReason, PointRegion},
    point_root_session::PointRootSession,
    source_sonic_types::{SonicSide, SourceSonicStatus},
    wet_wave_types::WetWaveDomain,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

pub(super) struct FanResolution<R: Runtime<DType = DType>> {
    pub(super) resolved: Tensor<R>,
    pub(super) active: Tensor<R>,
}

pub(super) fn evaluate<R>(
    session: &mut PointRootSession<'_, '_, R>,
    wave: &WetWaveDomain<R>,
    side: SonicSide,
    candidates: &mut Vec<PointCandidate<R>>,
    #[cfg(test)] evidence: &mut Vec<PointFanEvidence<R>>,
) -> HydraulicResult<FanResolution<R>>
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
    let i = side.index();
    let owners = &wave.star.invocation_owner;
    let parent = &wave.star.parent_rows;
    let fan = if i == 0 {
        &wave.regions.left_fan
    } else {
        &wave.regions.right_fan
    };
    let (zero, one, fullzero) = checked::<R, _>(c, "wet fan candidate mask storage", |a| {
        Ok((
            Tensor::zeros(owners.shape(), a.dtype(), c.device())?,
            Tensor::full_scalar(owners.shape(), a.dtype(), 1., c.device())?,
            Tensor::zeros(session.invocation_rows.shape(), a.dtype(), c.device())?,
        ))
    })?;
    let state = c.index_select(&session.sonic.status[i], 0, owners)?;
    let missing = c.eq(
        &state,
        &Tensor::full_scalar(
            &[],
            DType::I64,
            SourceSonicStatus::NotRequested as i64 as f64,
            c.device(),
        )?,
    )?;
    let missing = c.where_cond(&missing, fan, &zero)?;
    let rows = c.nonzero(&missing)?;
    let rows = rows.reshape(&[rows.numel()])?;
    if rows.numel() != 0 {
        session.start_source_sonic(&c.index_select(owners, 0, &rows)?, side)?;
    }
    let state = c.index_select(&session.sonic.status[i], 0, owners)?;
    let is = |kind: SourceSonicStatus| -> HydraulicResult<Tensor<R>> {
        Ok(c.where_cond(
            &c.eq(
                &state,
                &Tensor::full_scalar(&[], DType::I64, kind as i64 as f64, c.device())?,
            )?,
            &one,
            &zero,
        )?)
    };
    // A positive wet root excludes a dry contact. Strict no-root endpoint signs exclude a wet fan.
    let excluded = c.maximum(
        &is(SourceSonicStatus::DryContact)?,
        &c.maximum(
            &is(SourceSonicStatus::NoRootDry)?,
            &is(SourceSonicStatus::NoRootSource)?,
        )?,
    )?;
    let mut resolved = c.where_cond(fan, &excluded, &one)?;
    let mut coverage = zero.clone();
    let local = c.arange(0., owners.numel() as f64, 1., DType::I64)?;
    let lookup = c.index_put(
        &Tensor::zeros(session.invocation_rows.shape(), DType::I64, c.device())?,
        0,
        owners,
        &local,
    )?;
    let contact = c.mul(fan, &is(SourceSonicStatus::SourceContact)?)?;
    let rows = c.nonzero(&contact)?;
    let rows = rows.reshape(&[rows.numel()])?;
    if rows.numel() != 0 {
        checked::<R, _>(c, "wet fan source-contact membership", |a| {
            let sourceowners = c.index_select(owners, 0, &rows)?;
            let sourceparent = c.index_select(parent, 0, &rows)?;
            let source = super::point_candidates_source::exterior(
                a,
                session.sources,
                &sourceowners,
                &sourceparent,
                side,
            )?;
            let proof = super::wet_fan_membership::evaluate(
                a,
                session.sources,
                wave,
                &rows,
                &source.query,
                side,
            )?;
            resolved = c.index_put(
                &resolved,
                0,
                &rows,
                &c.maximum(&proof.proved, &proof.excluded)?,
            )?;
            coverage = c.index_put(&coverage, 0, &rows, &proof.proved)?;
            let selected = c.nonzero(&proof.proved)?;
            let selected = selected.reshape(&[selected.numel()])?;
            if selected.numel() != 0 {
                candidates.push(PointCandidate::Exterior(Box::new(
                    super::point_candidates_source::exterior(
                        a,
                        session.sources,
                        &c.index_select(&sourceowners, 0, &selected)?,
                        &c.index_select(&sourceparent, 0, &selected)?,
                        side,
                    )?,
                )));
            }
            #[cfg(test)]
            evidence.push(PointFanEvidence {
                invocation_owner: sourceowners,
                parent_rows: sourceparent,
                side,
                membership: proof,
                physical_status: Tensor::zeros(rows.shape(), DType::I64, c.device())?,
            });
            Ok(())
        })?;
    }
    let demand = c.index_put(&fullzero, 0, owners, fan)?;
    let demanded = if i == 0 {
        [&demand, &fullzero]
    } else {
        [&fullzero, &demand]
    };
    let certificates = super::source_sonic_certificates::selected(session, demanded)?;
    for certificate in certificates {
        checked::<R, _>(c, "wet fan physical candidate membership", |a| {
            let rows = c.index_select(&lookup, 0, &certificate.invocation_owner)?;
            let proof = super::wet_fan_membership::evaluate(
                a,
                session.sources,
                wave,
                &rows,
                &certificate.query,
                side,
            )?;
            let valid = c.where_cond(
                &c.eq(
                    &certificate.status,
                    &Tensor::zeros(&[], DType::I64, c.device())?,
                )?,
                &Tensor::full_scalar(rows.shape(), a.dtype(), 1., c.device())?,
                &Tensor::zeros(rows.shape(), a.dtype(), c.device())?,
            )?;
            let active = c.mul(&proof.proved, &valid)?;
            let accepted = c.maximum(&proof.excluded, &active)?;
            resolved = c.index_put(&resolved, 0, &rows, &accepted)?;
            coverage = c.index_put(&coverage, 0, &rows, &active)?;
            #[cfg(test)]
            let proof_index = evidence.len();
            #[cfg(test)]
            evidence.push(PointFanEvidence {
                invocation_owner: certificate.invocation_owner.clone(),
                parent_rows: certificate.parent_rows.clone(),
                side,
                membership: proof,
                physical_status: certificate.status.clone(),
            });
            if c.nonzero(&active)?.numel() != 0 {
                candidates.push(PointCandidate::Fan(Box::new(PointFan {
                    certificate,
                    active,
                    #[cfg(test)]
                    authority: PointFanAuthority::Wet(proof_index),
                })));
            }
            Ok(())
        })?;
    }
    checked::<R, _>(c, "wet fan unresolved candidate alternatives", |a| {
        let pending = c.mul(fan, &c.sub(&one, &resolved)?)?;
        let region = if side == SonicSide::Left {
            PointRegion::LeftFan
        } else {
            PointRegion::RightFan
        };
        super::wet_candidates::pending(
            a,
            candidates,
            owners,
            parent,
            &pending,
            region,
            PointPendingReason::FanCertificate,
        )
    })?;
    Ok(FanResolution {
        resolved,
        active: coverage,
    })
}
