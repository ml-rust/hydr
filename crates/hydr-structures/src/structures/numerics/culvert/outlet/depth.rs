//! Certified depth at a given distance along a built profile.
//!
//! The distance `U(v)` rises along the profile. For a target `T`, the node scan picks the last
//! node certified below `T` and the first node certified above it. Solvr certified bisection
//! then encloses the root of `U(v) - T` between them, in the oriented coordinate `w = sign v`,
//! with the slope floor from [`floor`]. Each residual integrates `G` from the nearer bracket
//! node of a one-lane bracket. Where no floor holds, as next to a critical start, where the
//! bracket holds an unknown lane, or where the upper node is the pole, the bracket itself is the
//! enclosure. At the pole it reaches the far bound of the normal-depth enclosure. A residual
//! piece whose integrand is not certified takes the distances at both bracket nodes.

use super::integrand::floor;
use super::levels::{level_for, span_level};
use super::partition::{LANES, Profile};
use super::quadrature::integrate;
use super::trig::fill;
use crate::structures::StructureError;
use crate::structures::numerics::culvert::{BarrelClient, choose};
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootProbe,
    CertifiedRootStatus, CertifiedRootValues,
};
use std::cell::Cell;

/// Probe rounds allowed per solve.
const MAX_ITERATIONS: usize = 256;

/// Coordinate enclosure per target lane, with a mask of the lanes it holds.
pub(crate) struct DepthEnclosure<R: Runtime<DType = DType>> {
    pub(crate) lower: Tensor<R>,
    pub(crate) upper: Tensor<R>,
    pub(crate) solved: Tensor<R>,
}

/// Per-lane bracket gathered from the profile rows.
struct Bracket<R: Runtime<DType = DType>> {
    sign: Tensor<R>,
    start: Tensor<R>,
    low: Tensor<R>,
    high: Tensor<R>,
    reach: Bounds<R>,
    bisect: Tensor<R>,
    direct: Tensor<R>,
    slope: Tensor<R>,
    /// Highest quadrature level of the lanes inside the bracket.
    level: Tensor<R>,
    /// Distance enclosure at the upper node.
    upper_reach: Bounds<R>,
    /// 1 where the bracket is one lane.
    single: Tensor<R>,
    /// Survey remainder of the lane that starts at the lower node.
    remainder: Tensor<R>,
    /// Lane remainder budget of the row.
    budget: Tensor<R>,
}

fn gather_column<R>(
    c: &R::Client,
    table: &Tensor<R>,
    index: &Tensor<R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let lanes = index.numel();
    let column = c.cast(index, DType::I64)?.reshape(&[lanes, 1])?;
    c.gather(table, 1, &column)?.reshape(&[lanes])
}

fn bracket<R>(
    c: &R::Client,
    profile: &Profile<R>,
    rows: &Tensor<R>,
    target: &Bounds<R>,
    wanted: &Tensor<R>,
) -> Result<Bracket<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let p = rows.numel();
    let count = LANES + 1;
    let device = rows.device().clone();
    let take = |value: &Tensor<R>| c.index_select(value, 0, rows);
    let nodes = take(&profile.nodes)?;
    let reach_low = take(&profile.reach[0])?;
    let reach_high = take(&profile.reach[1])?;
    let asymptotic = take(&profile.spec.asymptotic)?;
    let sign = take(&profile.spec.sign)?;
    let far = take(&profile.spec.start_far)?;
    let wide = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        value
            .reshape(&[p, 1])?
            .broadcast_to(&[p, count])?
            .contiguous()
    };
    let below = c.lt(&reach_high, &wide(&target[0])?)?;
    let mut above = c.gt(&reach_low, &wide(&target[1])?)?;
    let pole: Vec<f64> = (0..count)
        .map(|k| f64::from(u8::from(k == LANES)))
        .collect();
    let pole = Tensor::<R>::from_slice(&pole, &[1, count], &device)?
        .broadcast_to(&[p, count])?
        .contiguous()?;
    above = c.maximum(&above, &c.mul(&pole, &wide(&asymptotic)?)?)?;
    let below_count = c.sum(&below, &[1], false)?;
    let above_count = c.sum(&above, &[1], false)?;
    let zero = fill(&below_count, 0.0)?;
    let one = fill(&below_count, 1.0)?;
    let lanes = fill(&below_count, LANES as f64)?;
    let total = fill(&below_count, count as f64)?;
    let low_index = c.maximum(&c.sub(&below_count, &one)?, &zero)?;
    let high_index = c.sub(&total, &above_count)?;
    let found = c.le(&high_index, &lanes)?;
    let high_index = c.minimum(&high_index, &lanes)?;
    let ordered = c.gt(&high_index, &low_index)?;
    let level = span_level(c, &take(&profile.levels)?, &low_index, &high_index)?;

    let low = gather_column(c, &nodes, &low_index)?;
    let at_pole = c.mul(&asymptotic, &c.eq(&high_index, &lanes)?)?;
    // Every finite distance lies short of the far bound of the pole enclosure.
    let high = c.where_cond(
        &at_pole,
        &take(&profile.spec.pole_far)?,
        &gather_column(c, &nodes, &high_index)?,
    )?;
    let reach = [
        gather_column(c, &reach_low, &low_index)?,
        gather_column(c, &reach_high, &low_index)?,
    ];
    let unbracketed = c.eq(&below_count, &zero)?;
    let start = c.where_cond(&unbracketed, &far, &low)?;
    // A bracket over an unknown lane has no distance to bisect on.
    let known = take(&profile.known)?;
    let unknown = c.sub(&fill(&known, 1.0)?, &known)?;
    let blind = span_level(c, &unknown, &low_index, &high_index)?;
    let floor_rows = c.minimum(&low, &high)?;
    let ceiling_rows = c.maximum(&low, &high)?;
    let params = profile.spec.params.select(c, rows)?;
    let slope = with_interval_arithmetic(c, DType::F64, "culvert outlet depth floor", |a| {
        floor(a, &params, &floor_rows, &ceiling_rows)
    })?;
    let usable = c.gt(&slope, &zero)?;
    let valid = c.mul(&c.mul(wanted, &found)?, &ordered)?;
    let solvable = c.mul(
        &c.mul(&usable, &c.sub(&one, &at_pole)?)?,
        &c.mul(&c.sub(&one, &unbracketed)?, &c.sub(&one, &blind)?)?,
    )?;
    let bisect = c.mul(&valid, &solvable)?;
    // A bracket taken as the enclosure needs its upper node certified above the target.
    let past = gather_column(c, &reach_low, &high_index)?;
    let certain = c.maximum(&at_pole, &c.gt(&past, &target[1])?)?;
    let direct = c.mul(&c.sub(&valid, &bisect)?, &certain)?;
    let upper_reach = [
        gather_column(c, &reach_low, &high_index)?,
        gather_column(c, &reach_high, &high_index)?,
    ];
    let single = c.eq(&high_index, &c.add(&low_index, &one)?)?;
    let lane = c.minimum(&low_index, &fill(&low_index, (LANES - 1) as f64)?)?;
    Ok(Bracket {
        sign,
        start,
        low,
        high,
        reach,
        bisect,
        direct,
        slope: c.where_cond(&usable, &slope, &one)?,
        level,
        upper_reach,
        single,
        remainder: gather_column(c, &take(&profile.remainder)?, &lane)?,
        budget: take(&profile.budget)?,
    })
}

/// Encloses the coordinate where the profile distance equals `target`, per target lane.
///
/// `rows` (I64) names the profile row of each lane. Lanes outside `wanted` report 0 in `solved`.
/// A solve stops at the relative width `rtol`.
pub(crate) fn depth_at<R>(
    c: &R::Client,
    profile: &Profile<R>,
    rows: &Tensor<R>,
    target: &Bounds<R>,
    wanted: &Tensor<R>,
    rtol: f64,
) -> Result<DepthEnclosure<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R> + CertifiedSimpsonAlgorithms<R>,
{
    let b = bracket(c, profile, rows, target, wanted)?;
    let lanes = rows.numel();
    // Lanes outside the solver get the single point v_j, so one endpoint settles them.
    let lower = c.mul(&b.sign, &b.low)?;
    let upper = c.where_cond(&b.bisect, &c.mul(&b.sign, &b.high)?, &lower)?;
    let params = profile.spec.params.select(c, rows)?;
    let options = CertifiedBisectionOptions {
        atol: 0.0,
        rtol,
        max_iterations: MAX_ITERATIONS,
        max_points: lanes.saturating_mul(2 * (MAX_ITERATIONS + 1)),
        probe: CertifiedRootProbe::SlackSecant,
    };
    let cause = Cell::new(None);
    let solved = c.bisect_certified(
        |query| {
            let outcome = residual(c, &params, &b, target, &query.owner, &query.point);
            match outcome {
                Ok([lower, upper]) => Ok(CertifiedRootValues { lower, upper }),
                Err(error) => {
                    let reason = error.to_string();
                    cause.set(Some(error));
                    Err(numr::error::Error::InvalidArgument {
                        arg: "culvert outlet depth residual",
                        reason,
                    })
                }
            }
        },
        &lower,
        &upper,
        &b.slope,
        &options,
    );
    let roots = match (solved, cause.take()) {
        (_, Some(error)) => return Err(error),
        (Ok(roots), None) => roots,
        (Err(error), None) => return Err(error.into()),
    };
    let status = c.cast(roots.status(), DType::F64)?;
    let code = |value: CertifiedRootStatus| fill(&status, value as i64 as f64);
    let accepted = c.maximum(
        &c.eq(&status, &code(CertifiedRootStatus::Converged)?)?,
        &c.eq(&status, &code(CertifiedRootStatus::AccuracyFloor)?)?,
    )?;
    let owner = roots.root_owner();
    let (mut root_low, mut root_high) = (lower.clone(), upper.clone());
    if owner.numel() != 0 {
        root_low = c.index_put(&root_low, 0, owner, roots.root_lower())?;
        root_high = c.index_put(&root_high, 0, owner, roots.root_upper())?;
    }
    // Back from w = sign v to v, with the ends swapped where sign is -1.
    let rising = c.gt(&b.sign, &fill(&b.sign, 0.0)?)?;
    let solved_low = c.where_cond(&rising, &root_low, &c.neg(&root_high)?)?;
    let solved_high = c.where_cond(&rising, &root_high, &c.neg(&root_low)?)?;
    let direct_low = c.minimum(&b.start, &b.high)?;
    let direct_high = c.maximum(&b.start, &b.high)?;
    let good = c.mul(&b.bisect, &accepted)?;
    let missing = fill(&status, f64::NAN)?;
    let low = c.where_cond(
        &good,
        &solved_low,
        &c.where_cond(&b.direct, &direct_low, &missing)?,
    )?;
    let high = c.where_cond(
        &good,
        &solved_high,
        &c.where_cond(&b.direct, &direct_high, &missing)?,
    )?;
    Ok(DepthEnclosure {
        lower: low,
        upper: high,
        solved: c.maximum(&good, &b.direct)?,
    })
}

/// Residual bounds `U(v) - T` at the oriented points `point` of lanes `owner`.
///
/// A one-lane bracket integrates from its nearer node: `U(v) = U_j + integral_{v_j}^{v} G` or
/// `U(v) = U_{j+1} - integral_{v}^{v_{j+1}} G`. The piece takes the fewest panels whose
/// predicted remainder, the lane survey remainder times the fifth power of the width ratio,
/// fits the lane budget. A wider bracket integrates from `v_j` at its highest lane level.
fn residual<R>(
    c: &R::Client,
    params: &super::integrand::ProfileParams<R>,
    b: &Bracket<R>,
    target: &Bounds<R>,
    owner: &Tensor<R>,
    point: &Tensor<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedSimpsonAlgorithms<R>,
{
    let take = |value: &Tensor<R>| c.index_select(value, 0, owner);
    let sign = take(&b.sign)?;
    let low = take(&b.low)?;
    let high = take(&b.high)?;
    let single = take(&b.single)?;
    let v = c.mul(&sign, point)?;
    let from_low = c.abs(&c.sub(&v, &low)?)?;
    let from_high = c.abs(&c.sub(&high, &v)?)?;
    let upward = c.mul(&single, &c.lt(&from_high, &from_low)?)?;
    let anchor = c.where_cond(&upward, &high, &low)?;
    let left = c.minimum(&anchor, &v)?;
    let right = c.maximum(&anchor, &v)?;
    let span = c.abs(&c.sub(&high, &low)?)?;
    let zero = fill(&span, 0.0)?;
    let ratio = c.div(
        &c.sub(&right, &left)?,
        &c.where_cond(&c.gt(&span, &zero)?, &span, &fill(&span, 1.0)?)?,
    )?;
    let square = c.mul(&ratio, &ratio)?;
    let predicted = c.mul(
        &take(&b.remainder)?,
        &c.mul(&c.mul(&square, &square)?, &ratio)?,
    )?;
    let piece_level = level_for(c, &predicted, &take(&b.budget)?)?;
    let level = c.where_cond(&single, &piece_level, &take(&b.level)?)?;
    let lane = params.select(c, owner)?;
    let piece = integrate(c, &lane, &left, &right, &level)?;
    let [piece_low, piece_high] = &piece.bounds;
    let from = [take(&b.reach[0])?, take(&b.reach[1])?];
    let back = [take(&b.upper_reach[0])?, take(&b.upper_reach[1])?];
    let goal = [take(&target[0])?, take(&target[1])?];
    Ok(with_interval_arithmetic(
        c,
        DType::F64,
        "culvert outlet depth residual",
        |a| {
            let part = a.bounds(piece_low, piece_high)?;
            let start = a.bounds(&from[0], &from[1])?;
            let ahead = a.add(&start, &part)?;
            let end = a.bounds(&back[0], &back[1])?;
            let behind = a.sub(&end, &part)?;
            let total = choose(a, &upward, &behind, &ahead)?;
            // An unknown piece leaves the distance between its bracket nodes, where it rises.
            let between = a.bounds(&from[0], &c.maximum(&from[0], &back[1])?)?;
            let total = choose(a, &piece.known, &total, &between)?;
            let goal = a.bounds(&goal[0], &goal[1])?;
            let value = a.sub(&total, &goal)?;
            Ok([value.lower().clone(), value.upper().clone()])
        },
    )?)
}
