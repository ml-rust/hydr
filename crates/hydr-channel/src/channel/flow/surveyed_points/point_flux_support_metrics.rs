use super::{
    point_candidate_types::{PointCandidate, PointCandidateCertificates, PointCandidateStatus},
    point_flux_association::positions,
    wet_depth_types::WetDepthStatus,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) struct FanMetrics<R: Runtime<DType = DType>> {
    pub(super) blocked: Tensor<R>,
    pub(super) present: Tensor<R>,
    pub(super) lower: Tensor<R>,
    pub(super) upper: Tensor<R>,
    pub(super) normalized_upper: Tensor<R>,
    pub(super) work: Tensor<R>,
    pub(super) floor: Tensor<R>,
}
pub(super) struct Metrics<R: Runtime<DType = DType>> {
    pub(super) fans: [FanMetrics<R>; 2],
    pub(super) other: Tensor<R>,
    pub(super) complete: Tensor<R>,
}

pub(super) fn collect<R>(
    a: &Arithmetic<'_, R>,
    certificates: &PointCandidateCertificates<R>,
    inverse: &Tensor<R>,
) -> HydraulicResult<Metrics<R>>
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
    let zero = Tensor::zeros(certificates.invocation_owner.shape(), a.dtype(), c.device())?;
    let fan = || FanMetrics {
        blocked: zero.clone(),
        present: zero.clone(),
        lower: zero.clone(),
        upper: zero.clone(),
        normalized_upper: zero.clone(),
        work: zero.clone(),
        floor: zero.clone(),
    };
    let complete = c.where_cond(
        &c.eq(
            &certificates.status,
            &Tensor::full_scalar(
                &[],
                DType::I64,
                PointCandidateStatus::Complete as i64 as f64,
                c.device(),
            )?,
        )?,
        &Tensor::ones(zero.shape(), a.dtype(), c.device())?,
        &zero,
    )?;
    let mut metrics = Metrics {
        fans: [fan(), fan()],
        other: zero.clone(),
        complete,
    };
    for candidate in &certificates.candidates {
        let (owners, lower) = match candidate {
            PointCandidate::Fan(fan) => {
                let certificate = &fan.certificate;
                let rows = c.nonzero(&fan.active)?.flatten()?;
                if rows.numel() == 0 {
                    continue;
                }
                let owners = c.index_select(&certificate.invocation_owner, 0, &rows)?;
                let output = positions(a, inverse, &owners)?;
                let lo = c.index_select(certificate.query.area.lower(), 0, &rows)?;
                let hi = c.index_select(certificate.query.area.upper(), 0, &rows)?;
                let normalized = c.index_select(certificate.normalized_depth.upper(), 0, &rows)?;
                let local_zero = Tensor::zeros(rows.shape(), a.dtype(), c.device())?;
                let blocked = c.le(&lo, &local_zero)?;
                let status = c.index_select(&certificate.depth_status, 0, &rows)?;
                let status_mask = |code: WetDepthStatus| -> HydraulicResult<Tensor<R>> {
                    let code =
                        Tensor::full_scalar(&[], DType::I64, code as i64 as f64, c.device())?;
                    Ok(c.where_cond(
                        &c.eq(&status, &code)?,
                        &Tensor::ones(rows.shape(), a.dtype(), c.device())?,
                        &local_zero,
                    )?)
                };
                let side = &mut metrics.fans[certificate.side.index()];
                side.blocked = c.index_put(&side.blocked, 0, &output, &blocked)?;
                side.present = c.index_put(
                    &side.present,
                    0,
                    &output,
                    &Tensor::ones(output.shape(), a.dtype(), c.device())?,
                )?;
                side.lower = c.index_put(&side.lower, 0, &output, &lo)?;
                side.upper = c.index_put(&side.upper, 0, &output, &hi)?;
                side.normalized_upper =
                    c.index_put(&side.normalized_upper, 0, &output, &normalized)?;
                side.work = c.index_put(
                    &side.work,
                    0,
                    &output,
                    &status_mask(WetDepthStatus::WorkLimit)?,
                )?;
                side.floor = c.index_put(
                    &side.floor,
                    0,
                    &output,
                    &status_mask(WetDepthStatus::AccuracyFloor)?,
                )?;
                continue;
            }
            PointCandidate::Star(star) => {
                let wave = certificates.waves.get(star.wave).ok_or_else(|| {
                    super::point_root_session::invalid("positive support star wave is absent")
                })?;
                (
                    &star.invocation_owner,
                    c.index_select(wave.star.query.area.lower(), 0, &star.rows)?,
                )
            }
            PointCandidate::Exterior(exterior) => (
                &exterior.invocation_owner,
                exterior.query.area.lower().clone(),
            ),
            PointCandidate::Dry(_) | PointCandidate::Pending(_) => continue,
        };
        let output = positions(a, inverse, owners)?;
        let local_zero = Tensor::zeros(lower.shape(), a.dtype(), c.device())?;
        let blocked = c.le(&lower, &local_zero)?;
        let previous = c.index_select(&metrics.other, 0, &output)?;
        metrics.other =
            c.index_put(&metrics.other, 0, &output, &c.maximum(&previous, &blocked)?)?;
    }
    Ok(metrics)
}

pub(super) fn improved<R>(
    a: &Arithmetic<'_, R>,
    before: &FanMetrics<R>,
    after: &FanMetrics<R>,
) -> HydraulicResult<Tensor<R>>
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
    let nested = c.mul(
        &c.ge(&after.lower, &before.lower)?,
        &c.le(&after.upper, &before.upper)?,
    )?;
    let strict = c.maximum(
        &c.gt(&after.lower, &before.lower)?,
        &c.lt(&after.upper, &before.upper)?,
    )?;
    Ok(c.mul(
        &after.present,
        &c.mul(&before.blocked, &c.mul(&nested, &strict)?)?,
    )?)
}

pub(super) fn finite_quarter<R>(
    a: &Arithmetic<'_, R>,
    upper: &Tensor<R>,
) -> HydraulicResult<(Tensor<R>, Tensor<R>)>
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
    let zero = Tensor::zeros(upper.shape(), a.dtype(), c.device())?;
    let quarter = c.mul_scalar(upper, 0.25)?;
    let half = c.mul_scalar(upper, 0.5)?;
    let positive = c.mul(&c.gt(&quarter, &zero)?, &c.lt(&quarter, &half)?)?;
    Ok((quarter, positive))
}
