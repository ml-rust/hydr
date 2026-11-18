//! Certified signal-speed envelopes for CFL control, read from point-candidate certificates.

use super::{
    point_candidate_types::{PointCandidateCertificates, PointRegime},
    point_flux_masks::{equals, flag},
    point_flux_types::{PointFluxCertificates, PointFluxStatus},
    wet_wave_types::WetWaveStatus,
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
use running::{Running, ones, positive, require, zeros};
use solvr::common::arithmetic::Interval;

mod running;

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

#[cfg(test)]
mod tests {
    use super::super::corpus_coverage::Coverage;
    use super::super::{
        chart_fixture,
        point_candidates::fixture::{fixtures, wave_references},
        point_flux_types::PointFluxStatus,
        point_root_session::PointRootSession,
        source_fixture_support as frozen,
        wet_depth::fixture as support,
    };
    use super::*;
    use crate::channel::flow::surveyed::error::checked;
    use frozen::{Result, hydraulic};

    /// Every frozen edge of every evaluated corpus row lies inside its face envelope.
    fn corpus<R>(client: &R::Client, coverage: Coverage) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let executions = coverage.select(fixtures::executions(), |row| (row.outcome, row.physical));
        let mut evaluated = 0;
        for (config_index, declared) in fixtures::CONFIGS.iter().enumerate() {
            let rows = executions
                .iter()
                .filter(|e| e.config == config_index)
                .collect::<Vec<_>>();
            if rows.is_empty() || rows.iter().all(|row| row.outcome == "WorkLimit") {
                continue;
            }
            let ids = rows.iter().map(|row| row.id).collect::<Vec<_>>();
            let config = declared.build();
            chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, _| {
                let parents = (0..rows.len())
                    .map(i64::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let parents = support::rows::<R>(client, &parents)?;
                let mut session = hydraulic(PointRootSession::new(
                    pair,
                    &parents,
                    config.root.max_points,
                ))?;
                let result = hydraulic(session.resolve_point_fluxes(&parents))?;
                let bounds = frozen::endpoints(&result.wave_speed_m_per_s)?;
                let valid = result.wave_valid.try_to_vec::<f64>()?;
                let status = result.status.try_to_vec::<i64>()?;
                for (i, row) in rows.iter().enumerate() {
                    if status[i] == PointFluxStatus::PendingCandidates as i64 {
                        assert!(
                            row.outcome == "CapExceeded" || row.outcome == "physical_cap_rejection",
                            "{} loses its candidates",
                            row.id
                        );
                        continue;
                    }
                    assert_eq!(status[i], PointFluxStatus::Accepted as i64, "{}", row.id);
                    assert_eq!(valid[i], 1., "{} lacks a certified envelope", row.id);
                    if row.physical == "both_dry" {
                        assert_eq!(bounds[i], [0., 0.], "{}", row.id);
                    }
                    for (edge, label) in wave_references::expected(row) {
                        assert!(
                            bounds[i][0] <= edge[0] && edge[1] <= bounds[i][1],
                            "{} {label} {edge:?} escapes the envelope {:?}",
                            row.id,
                            bounds[i]
                        );
                    }
                    evaluated += 1;
                }
                Ok(())
            })?;
        }
        if coverage.is_full() {
            assert_eq!(evaluated, 66);
        }
        Ok(())
    }

    /// Still water on a rectangle reaches ±√(gh). A both-dry face is exactly [0, 0].
    /// Removing a possible branch speed rejects only its face.
    fn controls<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_fixture::config();
        // Width 1 makes depth equal to area.
        let depth = 2.;
        let gh = config.gravity_m_per_s2 * depth;
        support::with_rectangle::<R, _>(
            client,
            1.,
            4.,
            &[[depth, 0., depth, 0.], [0., 0., 0., 0.]],
            &config,
            |pair, _| {
                let owners = support::rows::<R>(client, &[0, 1])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let result = hydraulic(session.resolve_point_fluxes(&owners))?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::Accepted as i64; 2]
                );
                assert_eq!(result.wave_valid.try_to_vec::<f64>()?, [1., 1.]);
                let bounds = frozen::endpoints(&result.wave_speed_m_per_s)?;
                // The sign of x² - gh is exact under one fused rounding.
                let reaches = |x: f64| x > 0. && x.mul_add(x, -gh) >= 0.;
                assert!(reaches(-bounds[0][0]), "{:?} misses -√(gh)", bounds[0]);
                assert!(reaches(bounds[0][1]), "{:?} misses +√(gh)", bounds[0]);
                // Tightness: still water carries no wave faster than the gravity wave.
                let sound = gh.sqrt();
                assert!(-bounds[0][0] - sound < 1e-12 && bounds[0][1] - sound < 1e-12);
                assert_eq!(bounds[1], [0., 0.]);

                let mut selection = hydraulic(session.select_point_fluxes(&owners))?;
                let waves = &mut selection.candidates.waves;
                assert!(!waves.is_empty());
                for wave in waves.iter_mut() {
                    let side = &mut wave.sides[0];
                    assert_eq!(side.equality_possible.try_to_vec::<f64>()?, [1.]);
                    let available = &side.head.available;
                    side.head.available =
                        Tensor::zeros(available.shape(), available.dtype(), available.device())?;
                }
                let status = selection.status.clone();
                let (envelope, status) =
                    hydraulic(checked::<R, _>(client, "wave envelope", |a| {
                        super::resolve(a, 2, &owners, std::slice::from_ref(&selection), &status)
                    }))?;
                assert_eq!(envelope.valid.try_to_vec::<f64>()?, [0., 1.]);
                assert_eq!(
                    status.try_to_vec::<i64>()?,
                    [
                        PointFluxStatus::WaveEnvelopeUnresolved as i64,
                        PointFluxStatus::Accepted as i64
                    ]
                );
                assert_eq!(
                    frozen::endpoints(&envelope.wave_speed_m_per_s)?,
                    [[0., 0.], [0., 0.]]
                );
                let direct = hydraulic(checked::<R, _>(client, "wave envelope rows", |a| {
                    super::bounds(a, 2, &selection.candidates)
                }))?;
                assert_eq!(direct.2.try_to_vec::<f64>()?, [0., 1.]);
                Ok(())
            },
        )
    }

    /// A star owner repeated inside one wave domain rejects its face and no other.
    fn duplicate<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_fixture::config();
        let wet = [2., 0., 2., 0.];
        support::with_rectangle::<R, _>(
            client,
            1.,
            4.,
            &[wet, wet, wet, [0., 0., 0., 0.]],
            &config,
            |pair, _| {
                let owners = support::rows::<R>(client, &[0, 1, 2, 3])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let mut selection = hydraulic(session.select_point_fluxes(&owners))?;
                assert_eq!(
                    selection.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::Accepted as i64; 4]
                );
                let waves = &mut selection.candidates.waves;
                assert_eq!(waves.len(), 1);
                let star = &mut waves[0].star.invocation_owner;
                assert_eq!(star.try_to_vec::<i64>()?, [0, 1, 2]);
                // Owner 0 appears twice. Owner 1 loses its only wave domain.
                *star = support::rows::<R>(client, &[0, 0, 2])?;
                let status = selection.status.clone();
                let (envelope, status) =
                    hydraulic(checked::<R, _>(client, "wave envelope", |a| {
                        super::resolve(a, 4, &owners, std::slice::from_ref(&selection), &status)
                    }))?;
                assert_eq!(envelope.valid.try_to_vec::<f64>()?, [0., 0., 1., 1.]);
                let unresolved = PointFluxStatus::WaveEnvelopeUnresolved as i64;
                let accepted = PointFluxStatus::Accepted as i64;
                assert_eq!(
                    status.try_to_vec::<i64>()?,
                    [unresolved, unresolved, accepted, accepted]
                );
                Ok(())
            },
        )
    }

    #[test]
    fn configured_cpu_point_wave_envelopes_enclose_every_signal() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        let client = CpuClient::new(CpuDevice::new());
        controls::<CpuRuntime>(&client)?;
        duplicate::<CpuRuntime>(&client)?;
        corpus::<CpuRuntime>(&client, Coverage::Full)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_wave_envelopes_enclose_every_signal() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        controls::<CudaRuntime>(&client)?;
        duplicate::<CudaRuntime>(&client)?;
        corpus::<CudaRuntime>(&client, Coverage::Sample)
    }
}
