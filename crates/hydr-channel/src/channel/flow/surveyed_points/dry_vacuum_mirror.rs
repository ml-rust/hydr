//! Mirror faces in the dry and vacuum candidate set.
//!
//! A mirror face holds `(A, -Q)` on the left and `(A, Q)` on the right, in every component.
//! Its Riemann problem is its own reflection, so the face velocity is `u* = 0` and the water
//! flux is 0. With `u = Q / A` the vacuum invariant is `W = 2u - 2Phi(A)`.
//!
//! - `W > 0`: vacuum. The left dry front moves at `-W / 2 < 0` and the right one at `W / 2`,
//!   so the face lies in the empty interval.
//! - `W = 0`: the two fans touch at the face, where the area is 0.
//! - `W < 0`: two rarefactions. The star satisfies `2 Phi(A*) = -W`, and the face lies in it.
//!
//! An invariant enclosure that holds 0 leaves the sign open. The face then holds a dry state
//! or a star with `Phi(A*) <= P`, where `P = max(-W_lower, 0) / 2`. Certified widths do not
//! decrease, so `A(h) <= T(h) h`, `T / A >= 1 / h`, and `Phi(h) >= 2 sqrt(g h)`. The star depth
//! therefore stays below `h_r = P^2 / (4 g)`. Its momentum flux is `g I(A*)`, which lies in
//! `[0, g I(h_r)]`. The dry alternative has momentum 0, inside the same box.

use super::{
    dry_vacuum_types::{DryVacuumPossibilities, DryVacuumRegime},
    point_root_session::PointRootSession,
    wave_curve_types::CurveValue,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps, UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Mirror rows of one dry and vacuum certification, in owner order.
pub(super) struct MirrorRows<R: Runtime<DType = DType>> {
    /// One where the captured states are exact mirrors.
    pub(super) mirror: Tensor<R>,
    /// One where a mirror row's vacuum sign stays open and its star box holds.
    pub(super) overlap: Tensor<R>,
    /// Upper bound of the star depth `h_r` per row. Zero outside the overlap.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "tests read the star depth bound of each dry candidate"
        )
    )]
    pub(super) depth: Tensor<R>,
    /// Upper bound of the star momentum flux `g I(h_r)` per row. Zero outside the overlap.
    pub(super) momentum: Tensor<R>,
}

/// Mirror rows of `owners`, and the star box of each unresolved mirror row.
///
/// A row whose box depth reaches the certified cap keeps its open sign, and so rejects.
pub(super) fn rows<R>(
    a: &mut Arithmetic<'_, R>,
    session: &PointRootSession<'_, '_, R>,
    owners: &Tensor<R>,
    parent: &Tensor<R>,
    regime: &Tensor<R>,
    invariant: &CurveValue<R>,
) -> HydraulicResult<MirrorRows<R>>
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
    let sources = &session.sources;
    let mirror = super::point_flux_reflection::mirror_mask(
        a,
        sources.velocity.input(),
        &session.invocation_rows,
        owners,
    )?;
    let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
    let unresolved = c.eq(
        regime,
        &Tensor::full_scalar(
            &[],
            DType::I64,
            DryVacuumRegime::Unresolved as i64 as f64,
            c.device(),
        )?,
    )?;
    let open = c.where_cond(&unresolved, &mirror, &zero)?;
    let mut overlap = zero.clone();
    let mut depth = zero.clone();
    let mut momentum = zero.clone();
    let rows = c.nonzero(&open)?.flatten()?;
    if rows.numel() != 0 {
        let selected = sources.select_sources(a, &c.index_select(parent, 0, &rows)?)?;
        let n = rows.numel();
        let row_zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
        let lower = c.index_select(invariant.physical.lower(), 0, &rows)?;
        let limit = a.point(&c.maximum(&c.neg(&lower)?, &row_zero)?)?;
        let half = a.point(&Tensor::full_scalar(&[n], a.dtype(), 0.5, c.device())?)?;
        let limit = a.mul(&limit, &half)?;
        let gravity = sources.evaluation.config.gravity_m_per_s2;
        let four_g = a.point(&Tensor::full_scalar(
            &[n],
            a.dtype(),
            4.0 * gravity,
            c.device(),
        )?)?;
        let square = a.mul(&limit, &limit)?;
        let bound = a.div_positive_interval(&square, &four_g)?;
        // The box needs the whole depth range inside the certified cap.
        let inside = c.le(bound.upper(), selected.bands.cap_depth.lower())?;
        let inside = c.where_cond(
            &inside,
            &Tensor::ones(&[n], a.dtype(), c.device())?,
            &row_zero,
        )?;
        let kept = c.nonzero(&inside)?.flatten()?;
        if kept.numel() != 0 {
            let bands = selected.bands.select(a, &kept)?;
            let top = c.index_select(bound.upper(), 0, &kept)?;
            let range = a.bounds(&Tensor::zeros(top.shape(), a.dtype(), c.device())?, &top)?;
            let state = sources.geometry.at_depth(a, &bands, &range, None)?;
            let g = a.point(&Tensor::full_scalar(
                top.shape(),
                a.dtype(),
                gravity,
                c.device(),
            )?)?;
            let pressure = a.mul(&g, &state.integral)?;
            let at = c.index_select(&rows, 0, &kept)?;
            overlap = c.index_put(&overlap, 0, &at, &c.index_select(&inside, 0, &kept)?)?;
            depth = c.index_put(&depth, 0, &at, &top)?;
            momentum = c.index_put(&momentum, 0, &at, pressure.upper())?;
        }
    }
    Ok(MirrorRows {
        mirror,
        overlap,
        depth,
        momentum,
    })
}

/// `regime` with each overlap row read as touching, so both edges stay certified.
pub(super) fn regime<R>(
    a: &Arithmetic<'_, R>,
    regime: &Tensor<R>,
    rows: &MirrorRows<R>,
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
    let zero = Tensor::zeros(rows.overlap.shape(), a.dtype(), c.device())?;
    Ok(c.where_cond(
        &c.gt(&rows.overlap, &zero)?,
        &Tensor::full_scalar(
            regime.shape(),
            DType::I64,
            DryVacuumRegime::Touching as i64 as f64,
            c.device(),
        )?,
        regime,
    )?)
}

/// `possible` with every both-wet mirror row at the dry face alone.
pub(super) fn restrict<R>(
    a: &Arithmetic<'_, R>,
    regime: &Tensor<R>,
    rows: &MirrorRows<R>,
    possible: DryVacuumPossibilities<R>,
) -> HydraulicResult<DryVacuumPossibilities<R>>
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
    let zero = Tensor::zeros(rows.mirror.shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(rows.mirror.shape(), a.dtype(), c.device())?;
    let is = |value: DryVacuumRegime| -> HydraulicResult<Tensor<R>> {
        Ok(c.where_cond(
            &c.eq(
                regime,
                &Tensor::full_scalar(&[], DType::I64, value as i64 as f64, c.device())?,
            )?,
            &one,
            &zero,
        )?)
    };
    let both = c.maximum(
        &is(DryVacuumRegime::Vacuum)?,
        &is(DryVacuumRegime::Touching)?,
    )?;
    let face = c.mul(&rows.mirror, &both)?;
    let open = c.sub(&one, &face)?;
    Ok(DryVacuumPossibilities {
        left_exterior: c.mul(&possible.left_exterior, &open)?,
        left_fan: c.mul(&possible.left_fan, &open)?,
        dry: c.maximum(&possible.dry, &face)?,
        right_fan: c.mul(&possible.right_fan, &open)?,
        right_exterior: c.mul(&possible.right_exterior, &open)?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        chart_fixture::wall_face, point_output::PrismaticPointTraceKind,
        source_fixture_support as frozen, wet_depth::fixture as support,
    };
    use frozen::Result;
    use numr::{dtype::DType, runtime::Runtime};

    /// A wall face whose vacuum sign the invariant enclosure leaves open.
    ///
    /// The end cell of a 4 m rectangle holds `A = 2 m²` and flows away from the wall at
    /// Froude 2, so `u = 2c = Phi(A)` and the two fans touch in exact arithmetic. The rounded
    /// discharge leaves `W` inside a radius of a few ulps around 0.
    ///
    /// Frozen criteria:
    /// - the face accepts with a dry trace
    /// - the water flux is exactly +0.0 in every component, with bounds `[0, 0]`
    /// - the momentum bounds hold 0 and the star box: `0 <= lower`, `0 < upper <= 1e-60`
    fn open_sign<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let discharge = (-(2.0 * 0.5 * (9.81_f64 * 0.5).sqrt()) * 4.0).to_bits();
        let result = wall_face::<R>(
            client,
            4.0,
            [[2.0_f64.to_bits(), 0], [discharge, (-0.0_f64).to_bits()]],
        )?
        .map_err(|error| format!("the open-sign wall face rejects: {error:?}"))?;
        assert_eq!(
            result.trace_kind().try_to_vec::<i64>()?,
            [PrismaticPointTraceKind::Dry as i64]
        );
        assert_eq!(
            frozen::components(result.water_flux_m3_per_s())?,
            [[0, 0, 0]]
        );
        assert_eq!(
            frozen::endpoints(result.water_flux_bound_m3_per_s())?,
            [[0.0, 0.0]]
        );
        let [lower, upper] = frozen::endpoints(result.momentum_flux_bound_m4_per_s2())?[0];
        assert!(
            lower <= 0.0 && 0.0 < upper && upper <= 1e-60,
            "[{lower}, {upper}]"
        );
        Ok(())
    }

    #[test]
    fn cpu_mirror_face_with_an_open_vacuum_sign_takes_the_star_box() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        open_sign::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_mirror_face_with_an_open_vacuum_sign_takes_the_star_box() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        open_sign::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
