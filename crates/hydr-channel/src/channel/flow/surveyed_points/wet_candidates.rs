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
                // A mirror face solves its own reflection, so u* = 0. A left shock then moves
                // at -Q/(A* - A) < 0, and a right shock mirrors it. A left rarefaction tail
                // moves at -c*, and a right tail at +c*. The face lies in the star, and no
                // exterior state reaches it.
                let mirror = super::point_flux_reflection::mirror_mask(
                    a,
                    session.sources.velocity.input(),
                    &session.invocation_rows,
                    owners,
                )?;
                let open = c.sub(&one, &mirror)?;
                let left_exterior = c.mul(&wave.regions.left_exterior, &open)?;
                let right_exterior = c.mul(&wave.regions.right_exterior, &open)?;
                for (mask, side) in [
                    (&left_exterior, super::source_sonic_types::SonicSide::Left),
                    (&right_exterior, super::source_sonic_types::SonicSide::Right),
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
                    &left_exterior,
                    &c.maximum(&wave.regions.star, &right_exterior)?,
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

#[cfg(test)]
mod tests {
    use super::super::{
        chart_fixture::wall_face, point_output::PrismaticPointTraceKind,
        source_fixture_support as frozen, wet_depth::fixture as support,
    };
    use frozen::Result;
    use numr::{dtype::DType, runtime::Runtime};

    /// Star depth of a film of depth `h` that hits a wall at `u`, on a unit rectangle.
    ///
    /// The left shock satisfies `u = (s - h) sqrt(g (s + h) / (2 s h))`. Bisection runs in F64.
    fn wall_shock_depth(h: f64, u: f64, g: f64) -> f64 {
        let residual = |s: f64| (s - h) * (g * (s + h) / (2.0 * s * h)).sqrt() - u;
        let (mut lower, mut upper) = (h, 1.0);
        for _ in 0..400 {
            let middle = if upper > 4.0 * lower {
                (lower * upper).sqrt()
            } else {
                0.5 * (lower + upper)
            };
            if residual(middle) > 0.0 {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        lower
    }

    /// A wet front of a dam break into a dry reach reaches the far wall.
    ///
    /// The end cell of a 1 m rectangle holds a film near 1.4e-24 m² at about 2 m/s. The star
    /// depth lies twelve orders above the film, so the shock speed `-Q / (A* - A)` stays near
    /// 0 and its enclosure holds both signs. The mirror identity still places the face in the
    /// star.
    ///
    /// Frozen criteria:
    /// - the face accepts with a star trace
    /// - the water flux is exactly +0.0 in every component
    /// - the momentum bounds hold `g s² / 2` of the F64 reference star depth `s`, widened by
    ///   `1e-12` relative for the reference rounding
    fn film_shock<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let area = [0x3afa_51b7_7122_b4b1, 0x379b_7440_7874_9bc1];
        let discharge = [0x3b0a_9b5b_56b1_fda7, 0xb7a3_fc5a_25c2_b625];
        let result = wall_face::<R>(client, 1.0, [area, discharge])?
            .map_err(|error| format!("the film wall face rejects: {error:?}"))?;
        assert_eq!(
            result.trace_kind().try_to_vec::<i64>()?,
            [PrismaticPointTraceKind::Star as i64]
        );
        assert_eq!(
            frozen::components(result.water_flux_m3_per_s())?,
            [[0, 0, 0]]
        );
        let value = |[high, low]: [u64; 2]| f64::from_bits(high) + f64::from_bits(low);
        let (h, q) = (value(area), value(discharge));
        let gravity = 9.81;
        let depth = wall_shock_depth(h, q / h, gravity);
        let pressure = gravity * depth * depth / 2.0;
        let slack = 1e-12 * pressure;
        let [lower, upper] = frozen::endpoints(result.momentum_flux_bound_m4_per_s2())?[0];
        assert!(
            lower - slack <= pressure && pressure <= upper + slack,
            "[{lower:e}, {upper:e}] excludes {pressure:e}"
        );
        Ok(())
    }

    #[test]
    fn cpu_mirror_film_shock_against_a_wall_takes_the_star() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        film_shock::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_mirror_film_shock_against_a_wall_takes_the_star() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        film_shock::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
