//! Momentum gap `M(y2) - M(y_S2(x(y2)))` at S1 coordinates, and the S1 distance it needs.

use super::super::March;
use super::super::coordinates::depth;
use super::super::depth::depth_at;
use super::super::levels::Accuracy;
use super::super::partition::Profile;
use super::super::quadrature::integrate;
use super::scan::Bracket;
use super::steep::{Steep, momentum, pair};
use crate::structures::StructureError;
use crate::structures::numerics::culvert::{BarrelClient, choose};
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorOutletContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, IndexingOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedBisectionAlgorithms;

/// S1 distance `U1(v) = U1_high + integral_{v}^{v_high} G1` at the points `point` of rows `owner`.
pub(super) fn upstream_distance<R>(
    c: &R::Client,
    profile: &Profile<R>,
    b: &Bracket<R>,
    owner: &Tensor<R>,
    point: &Tensor<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedSimpsonAlgorithms<R>,
{
    let take = |value: &Tensor<R>| c.index_select(value, 0, owner);
    let high = take(&b.high)?;
    let left = c.minimum(point, &high)?;
    let right = c.maximum(point, &high)?;
    let params = profile.spec.params.select(c, owner)?;
    let piece = integrate(c, &params, &left, &right, &take(&b.level)?)?;
    let [piece_low, piece_high] = &piece.bounds;
    let base = [take(&b.reach[0])?, take(&b.reach[1])?];
    let far = take(&b.far_reach)?;
    Ok(with_interval_arithmetic(
        c,
        DType::F64,
        "culvert outlet jump distance",
        |a| {
            let start = a.bounds(&base[0], &base[1])?;
            let part = a.bounds(piece_low, piece_high)?;
            let total = a.add(&start, &part)?;
            // An unknown piece leaves the distance between the bracket nodes, where it rises.
            let between = a.bounds(&base[0], &c.maximum(&base[0], &far)?)?;
            let total = choose(a, &piece.known, &total, &between)?;
            Ok(pair(&total))
        },
    )?)
}

/// `L - U` as an interval.
pub(super) fn position<R>(
    c: &R::Client,
    length: &Tensor<R>,
    distance: &Bounds<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    Ok(with_interval_arithmetic(
        c,
        DType::F64,
        "culvert outlet jump position",
        |a| {
            let l = a.point(length)?;
            let u = a.bounds(&distance[0], &distance[1])?;
            let x = a.sub(&l, &u)?;
            Ok(pair(&x))
        },
    )?)
}

/// Everything one gap evaluation reads.
pub(super) struct Solve<'s, R: Runtime<DType = DType>> {
    pub(super) barrel: &'s TensorBarrelContext<R>,
    pub(super) outlet: &'s TensorOutletContext<R>,
    pub(super) march: &'s March<R>,
    pub(super) steep: &'s Steep<R>,
    pub(super) s2: &'s Profile<R>,
    pub(super) bracket: &'s Bracket<R>,
}

/// Momentum gap bounds at the S1 coordinates `point` of rows `owner`.
pub(super) fn gap<R>(
    c: &R::Client,
    solve: &Solve<'_, R>,
    owner: &Tensor<R>,
    point: &Tensor<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R> + CertifiedSimpsonAlgorithms<R>,
{
    let b = solve.bracket;
    let take = |value: &Tensor<R>| c.index_select(value, 0, owner);
    let wanted = take(&b.bisect)?;
    let distance = upstream_distance(c, &solve.march.profile, b, owner, point)?;
    let x = position(c, &take(&solve.outlet.length)?, &distance)?;
    let rtol = Accuracy::Fine.depth_tolerance();
    let found = depth_at(c, solve.s2, owner, &x, &wanted, rtol)?;
    let rows = solve.barrel.select(owner)?;
    let shapes = solve.outlet.select(c, owner)?;
    let outline = shapes.outline(&rows);
    // A wanted lane without a supercritical depth gets inverted bounds, which reject the solve.
    let missing = c.mul(&wanted, &c.sub(&rows.one, &found.solved)?)?;
    let scale = [
        take(&solve.steep.momentum_scale[0])?,
        take(&solve.steep.momentum_scale[1])?,
    ];
    Ok(with_interval_arithmetic(
        c,
        DType::F64,
        "culvert outlet jump gap",
        |a| {
            let scale = a.bounds(&scale[0], &scale[1])?;
            let v2 = a.point(point)?;
            let y2 = depth(a, &outline, &rows.rise, &v2)?;
            let middle = outline.harmless()?;
            let v1 = a.bounds(
                &c.where_cond(&found.solved, &found.lower, &middle)?,
                &c.where_cond(&found.solved, &found.upper, &middle)?,
            )?;
            let y1 = depth(a, &outline, &rows.rise, &v1)?;
            let m2 = momentum(a, &rows, &scale, &y2)?;
            let m1 = momentum(a, &rows, &scale, &y1)?;
            let value = a.sub(&m2, &m1)?;
            let zero = &rows.zero;
            let lower = c.where_cond(&wanted, value.lower(), zero)?;
            let upper = c.where_cond(&wanted, value.upper(), zero)?;
            let lower = c.where_cond(&missing, &rows.one, &lower)?;
            let upper = c.where_cond(&missing, zero, &upper)?;
            let checked = a.bounds(&lower, &upper)?;
            Ok(pair(&checked))
        },
    )?)
}
