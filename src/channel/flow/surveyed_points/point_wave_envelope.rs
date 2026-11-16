//! Certified signal-speed envelopes for CFL control, read from point-candidate certificates.

use super::{
    point_candidate_types::{PointCandidateCertificates, PointRegime},
    point_flux_controller::{equals, flag},
    point_flux_types::{PointFluxCertificates, PointFluxStatus},
    wave_curve_types::CurveValue,
    wet_wave_types::{WetWaveSide, WetWaveStatus},
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        ScatterReduceOp, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

/// Signal speeds per row: the lower bound is the left signal, the upper bound the right signal.
/// Rejected rows hold [0, 0] with `valid` zero. Those bounds certify nothing.
pub(super) struct PointWaveEnvelope<R: Runtime<DType = DType>> {
    pub(super) wave_speed_m_per_s: Interval<R>,
    pub(super) valid: Tensor<R>,
}

/// Envelope per face from the selection that owns it, plus the folded face status.
/// An accepted face without a certified envelope becomes `WaveEnvelopeUnresolved`.
pub(super) fn resolve<R>(
    a: &mut Arithmetic<'_, R>,
    invocation_count: usize,
    faces: &Tensor<R>,
    selections: &[PointFluxCertificates<R>],
    status: &Tensor<R>,
) -> HydraulicResult<(PointWaveEnvelope<R>, Tensor<R>)>
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
    let zero = Tensor::zeros(&[invocation_count], a.dtype(), c.device())?;
    let (mut lower, mut upper, mut valid) = (zero.clone(), zero.clone(), zero);
    // Later selections overwrite earlier ones, as the controller overwrites face values.
    for selection in selections {
        let rows = &selection.candidates.invocation_owner;
        if rows.numel() == 0 {
            continue;
        }
        let (l, u, v) = bounds(a, invocation_count, &selection.candidates)?;
        lower = c.index_put(&lower, 0, rows, &l)?;
        upper = c.index_put(&upper, 0, rows, &u)?;
        valid = c.index_put(&valid, 0, rows, &v)?;
    }
    let valid = c.index_select(&valid, 0, faces)?;
    let wave_speed_m_per_s = a.bounds(
        &c.index_select(&lower, 0, faces)?,
        &c.index_select(&upper, 0, faces)?,
    )?;
    let accepted = equals(a, status, PointFluxStatus::Accepted as i64)?;
    let rejected = c.mul(&accepted, &c.sub(&ones(a, valid.shape())?, &valid)?)?;
    let status = c.where_cond(
        &c.gt(&rejected, &zeros(a, rejected.shape())?)?,
        &Tensor::full_scalar(
            status.shape(),
            DType::I64,
            PointFluxStatus::WaveEnvelopeUnresolved as i64 as f64,
            c.device(),
        )?,
        status,
    )?;
    Ok((
        PointWaveEnvelope {
            wave_speed_m_per_s,
            valid,
        },
        status,
    ))
}

/// Raw lower, upper and validity tensors per candidate row.
fn bounds<R>(
    a: &mut Arithmetic<'_, R>,
    invocation_count: usize,
    candidates: &PointCandidateCertificates<R>,
) -> HydraulicResult<(Tensor<R>, Tensor<R>, Tensor<R>)>
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
    let owners = &candidates.invocation_owner;
    let shape = owners.shape().to_vec();
    let (wet, wet_ok) = wet_bounds(a, invocation_count, candidates)?;
    let wet = Running {
        lower: c.index_select(&wet.lower, 0, owners)?,
        upper: c.index_select(&wet.upper, 0, owners)?,
    };
    let wet_ok = c.index_select(&wet_ok, 0, owners)?;
    let mut dry = Running::new(a, &shape)?;
    for edges in &candidates.dry_vacuum.edges {
        for value in [&edges.head, &edges.front] {
            let available = positive(a, &value.available)?;
            dry.include(a, &available, value)?;
        }
    }
    let regime = &candidates.regime;
    let left_wet = equals(a, regime, PointRegime::LeftWet as i64)?;
    let right_wet = equals(a, regime, PointRegime::RightWet as i64)?;
    let both_wet = c.maximum(
        &equals(a, regime, PointRegime::Vacuum as i64)?,
        &equals(a, regime, PointRegime::Touching as i64)?,
    )?;
    let one = ones(a, &shape)?;
    let dryish = c.maximum(&c.maximum(&left_wet, &right_wet)?, &both_wet)?;
    let mut dry_ok = dryish.clone();
    // Each wet side of a dry or vacuum face needs both its head and its dry front.
    for (edges, needed) in candidates.dry_vacuum.edges.iter().zip([
        c.maximum(&left_wet, &both_wet)?,
        c.maximum(&right_wet, &both_wet)?,
    ]) {
        let available = c.mul(
            &positive(a, &edges.head.available)?,
            &positive(a, &edges.front.available)?,
        )?;
        dry_ok = c.mul(&dry_ok, &require(a, &needed, &available)?)?;
    }
    let wet_regime = equals(a, regime, PointRegime::Wet as i64)?;
    let both_dry = equals(a, regime, PointRegime::BothDry as i64)?;
    let valid = c.add(
        &both_dry,
        &c.add(&c.mul(&wet_regime, &wet_ok)?, &c.mul(&dryish, &dry_ok)?)?,
    )?;
    let valid = c.minimum(&valid, &one)?;
    let zero = zeros(a, &shape)?;
    let is_wet = c.gt(&wet_regime, &zero)?;
    let is_dry = c.gt(&dryish, &zero)?;
    let is_valid = c.gt(&valid, &zero)?;
    // Both-dry faces and rejected faces take exact zeros.
    let take = |wet: &Tensor<R>, dry: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        let value = c.where_cond(&is_wet, wet, &c.where_cond(&is_dry, dry, &zero)?)?;
        Ok(c.where_cond(&is_valid, &value, &zero)?)
    };
    let lower = take(&wet.lower, &dry.lower)?;
    let upper = take(&wet.upper, &dry.upper)?;
    Ok((lower, upper, valid))
}

/// Wet bounds and validity in invocation order, joined over every wave domain of each owner.
fn wet_bounds<R>(
    a: &mut Arithmetic<'_, R>,
    invocation_count: usize,
    candidates: &PointCandidateCertificates<R>,
) -> HydraulicResult<(Running<R>, Tensor<R>)>
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
    let mut wet = Running::new(a, &[invocation_count])?;
    let mut covered = zeros(a, &[invocation_count])?;
    let mut rejected = zeros(a, &[invocation_count])?;
    for wave in &candidates.waves {
        let rows = &wave.star.invocation_owner;
        if rows.numel() == 0 {
            continue;
        }
        let mut run = Running::new(a, rows.shape())?;
        let mut ok = equals(a, &wave.status, WetWaveStatus::Certified as i64)?;
        for side in &wave.sides {
            ok = c.mul(&ok, &run.include_side(a, side)?)?;
        }
        let one = ones(a, rows.shape())?;
        // A star owner that appears twice in one wave domain rejects its face.
        let counts = c.scatter_reduce(
            &zeros(a, &[invocation_count])?,
            0,
            rows,
            &one,
            ScatterReduceOp::Sum,
            true,
        )?;
        let repeated = c.index_select(&counts, 0, rows)?;
        let unique = flag(a, &c.le(&repeated, &one)?)?;
        ok = c.mul(&ok, &unique)?;
        wet.lower = c.index_put(
            &wet.lower,
            0,
            rows,
            &c.minimum(&c.index_select(&wet.lower, 0, rows)?, &run.lower)?,
        )?;
        wet.upper = c.index_put(
            &wet.upper,
            0,
            rows,
            &c.maximum(&c.index_select(&wet.upper, 0, rows)?, &run.upper)?,
        )?;
        covered = c.index_put(&covered, 0, rows, &one)?;
        rejected = c.index_put(
            &rejected,
            0,
            rows,
            &c.maximum(&c.index_select(&rejected, 0, rows)?, &c.sub(&one, &ok)?)?,
        )?;
    }
    let ok = c.mul(&covered, &c.sub(&ones(a, &[invocation_count])?, &rejected)?)?;
    Ok((wet, ok))
}

/// Running bounds. Empty rows hold +inf and -inf until a value is included.
struct Running<R: Runtime<DType = DType>> {
    lower: Tensor<R>,
    upper: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Running<R> {
    fn new(a: &Arithmetic<'_, R>, shape: &[usize]) -> HydraulicResult<Self>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let device = a.client().device();
        Ok(Self {
            lower: Tensor::full_scalar(shape, a.dtype(), f64::INFINITY, device)?,
            upper: Tensor::full_scalar(shape, a.dtype(), f64::NEG_INFINITY, device)?,
        })
    }

    /// Joins the outward physical bounds of `value` where `mask` is one and the value is available.
    fn include(
        &mut self,
        a: &Arithmetic<'_, R>,
        mask: &Tensor<R>,
        value: &CurveValue<R>,
    ) -> HydraulicResult<()>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let c = a.client();
        let shape = mask.shape();
        let take = c.mul(mask, &positive(a, &value.available)?)?;
        let take = c.gt(&take, &zeros(a, shape)?)?;
        let empty = Self::new(a, shape)?;
        self.lower = c.minimum(
            &self.lower,
            &c.where_cond(&take, value.physical.lower(), &empty.lower)?,
        )?;
        self.upper = c.maximum(
            &self.upper,
            &c.where_cond(&take, value.physical.upper(), &empty.upper)?,
        )?;
        Ok(())
    }

    /// Joins every possible branch speed of one wet side and returns its validity mask.
    /// A possible branch needs its speed: the shock speed, or the head and the rarefaction tail.
    fn include_side(
        &mut self,
        a: &Arithmetic<'_, R>,
        side: &WetWaveSide<R>,
    ) -> HydraulicResult<Tensor<R>>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let c = a.client();
        let shock = positive(a, &side.shock_possible)?;
        let rarefaction = positive(a, &side.rarefaction_possible)?;
        let head = c.maximum(&rarefaction, &positive(a, &side.equality_possible)?)?;
        let tail = positive(a, &side.tail.available)?;
        self.include(a, &shock, &side.shock_speed)?;
        self.include(a, &head, &side.head)?;
        self.include(a, &tail, &side.tail)?;
        let mut ok = positive(a, &side.valid)?;
        ok = c.mul(&ok, &c.maximum(&shock, &head)?)?;
        let shock_speed = positive(a, &side.shock_speed.available)?;
        ok = c.mul(&ok, &require(a, &shock, &shock_speed)?)?;
        ok = c.mul(
            &ok,
            &require(a, &head, &positive(a, &side.head.available)?)?,
        )?;
        ok = c.mul(&ok, &require(a, &rarefaction, &tail)?)?;
        Ok(ok)
    }
}

/// One where `needed` is zero or `available` is one.
fn require<R>(
    a: &Arithmetic<'_, R>,
    needed: &Tensor<R>,
    available: &Tensor<R>,
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
    let one = ones(a, needed.shape())?;
    Ok(c.sub(&one, &c.mul(needed, &c.sub(&one, available)?)?)?)
}

/// Numeric zero-one mask of the entries above zero.
fn positive<R>(a: &Arithmetic<'_, R>, values: &Tensor<R>) -> HydraulicResult<Tensor<R>>
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
    let zero = Tensor::zeros(values.shape(), values.dtype(), c.device())?;
    flag(a, &c.gt(values, &zero)?)
}

fn ones<R>(a: &Arithmetic<'_, R>, shape: &[usize]) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(Tensor::ones(shape, a.dtype(), a.client().device())?)
}

fn zeros<R>(a: &Arithmetic<'_, R>, shape: &[usize]) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(Tensor::zeros(shape, a.dtype(), a.client().device())?)
}

#[cfg(test)]
#[path = "point_wave_envelope_tests.rs"]
mod tests;
