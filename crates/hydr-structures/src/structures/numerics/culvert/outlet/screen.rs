//! Screening of Simpson queries: denominator bounds at panel-group ends and lane validity.
//!
//! An interval jet over a wide panel group overstates the range of `Sf`. Near the conveyance
//! peak `Sf` is nearly flat, so that overstatement can hide the sign of `S - Sf`. The bound of
//! [`end_bound`] uses the shape of `Sf` instead, from the value-node bounds at the group ends.
//! A circle's `Sf` has one minimum, at its frozen peak. An arc barrel's construction certificate
//! proves `Sf` falls with depth up to `s_N` and rises from `s_F`, and bounds `Sf` over
//! `[s_N, s_F]`. A query whose integrand stays uncertified takes harmless values, and its lane
//! reports 0 in the lane mask of [`lane_known`].

use super::bands::Outline;
use super::coordinates::fraction_coordinate;
use super::integrand::ProfileParams;
use super::jet;
use super::trig::fill;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// `value` on rows inside `valid` and `[0, 0]` elsewhere.
pub(super) fn screened<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    valid: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = fill(valid, 0.0)?;
    let lower = c.where_cond(valid, value.lower(), &zero)?;
    let upper = c.where_cond(valid, value.upper(), &zero)?;
    a.bounds(&lower, &upper)
}

/// Value-node indices `[first, last]` at the ends of each panel group, as I64 tensors.
///
/// Each of the `active` lanes lists `2 panels + 1` value nodes in order. Group `g` of a lane
/// spans its panels `g per .. (g + 1) per`, so it runs from node `2 g per` to `2 (g + 1) per`.
pub(super) fn group_ends<R>(
    like: &Tensor<R>,
    active: usize,
    groups: usize,
    per: usize,
    panels: usize,
) -> numr::error::Result<[Tensor<R>; 2]>
where
    R: Runtime<DType = DType>,
{
    let nodes = 2 * panels + 1;
    let index = |offset: usize| -> Vec<i64> {
        (0..active * groups)
            .map(|k| {
                let node = (k / groups) * nodes + 2 * per * (k % groups + offset);
                i64::try_from(node).unwrap_or(i64::MAX)
            })
            .collect()
    };
    let shape = [active * groups];
    Ok([
        Tensor::<R>::from_slice(&index(0), &shape, like.device())?,
        Tensor::<R>::from_slice(&index(1), &shape, like.device())?,
    ])
}

/// `params` with its conveyance peak fields from the per-row certificates of `barrel`.
///
/// `Sf = K / k^2`, with `k` the unit conveyance. On circle rows the peak enclosure holds the one
/// maximum of `k`, and `log_peak` encloses `ln k_max`. On arc rows `k` rises up to `s_N` and
/// falls from `s_F`, and `log_peak` encloses `ln k` over all of `[s_N, s_F]`. So over the peak
/// enclosure `Sf` stays inside `K exp(-2 log_peak)`:
///
/// - Where `Sf > S` along the profile, on circle and arc rows, the least `Sf - S` there.
/// - Where `S > Sf`, on arc rows, the least `S - Sf` there. On a circle `Sf` has no maximum
///   inside the enclosure, so the ends of a group bound it.
///
/// Other rows take `f64::MAX`.
pub(super) fn with_peak<R>(
    a: &mut Arithmetic<'_, R>,
    params: ProfileParams<R>,
    barrel: &TensorBarrelContext<R>,
) -> ArithmeticResult<ProfileParams<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = &params.rise;
    let fraction = a.bounds(&barrel.conveyance_peak[0], &barrel.conveyance_peak[1])?;
    // Box rows keep the circle form, whose angle lies past every box coordinate.
    let rounded = match &params.arcs {
        None => fill(like, 1.0)?,
        Some(arcs) => c.sub(&fill(like, 1.0)?, &arcs.arced)?,
    };
    let outline = Outline {
        circular: &rounded,
        beta: &params.beta,
        bed: &params.bed,
        arcs: params.arcs.as_ref(),
    };
    let angle = fraction_coordinate(a, &outline, &fraction)?;
    let log_peak = a.bounds(&barrel.log_peak[0], &barrel.log_peak[1])?;
    let two = a.point(&fill(like, 2.0)?)?;
    let twice = a.mul(&two, &log_peak)?;
    let log_friction = a.bounds(&params.log_friction[0], &params.log_friction[1])?;
    let exponent = a.sub(&log_friction, &twice)?;
    let friction = jet::exp_interval(a, &exponent)?;
    let slope = a.point(&params.slope)?;
    let excess = a.sub(&friction, &slope)?;
    let rubbing = c.lt(&params.denominator, &fill(like, 0.0)?)?;
    let dips = c.mul(&params.circular, &rubbing)?;
    let most = fill(like, f64::MAX)?;
    let mut peak_excess = c.where_cond(&dips, excess.lower(), &most)?;
    if let Some(arcs) = &params.arcs {
        let deficit = a.sub(&slope, &friction)?;
        let arc = c.where_cond(&rubbing, excess.lower(), deficit.lower())?;
        peak_excess = c.where_cond(&arcs.arced, &arc, &peak_excess)?;
    }
    Ok(ProfileParams {
        peak: [angle.lower().clone(), angle.upper().clone()],
        peak_excess,
        ..params
    })
}

/// Lower bound of the signed denominator `mu (S - Sf)` over each panel group `v`, from its
/// lower bounds `ends` at the value nodes that hold the group ends.
///
/// `Sf` falls with depth up to the conveyance peak enclosure and rises past it. A box has no
/// peak. So outside the enclosure `Sf` over a group is largest and smallest at its ends. A
/// group that may reach the enclosure also takes the bound of
/// [`with_peak`] over it. `lane` holds the parameters of each group.
pub(super) fn end_bound<R>(
    c: &R::Client,
    lane: &ProfileParams<R>,
    v: &Bounds<R>,
    ends: &[Tensor<R>; 2],
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let nearer = c.minimum(&ends[0], &ends[1])?;
    let reaches = c.mul(&c.le(&v[0], &lane.peak[1])?, &c.ge(&v[1], &lane.peak[0])?)?;
    let lowest = c.minimum(&nearer, &lane.peak_excess)?;
    c.where_cond(&reaches, &lowest, &nearer)
}

/// 1 on each of `lanes` lanes whose queries all passed, 0 on the others.
///
/// The `active` queried lanes list `values` value queries and `groups` panel groups each, in
/// lane order, with a 0/1 pass mask per query. `owner` (I64) names the lane of every value
/// query. Unqueried lanes pass.
pub(super) fn lane_known<R>(
    c: &R::Client,
    lanes: usize,
    owner: &Tensor<R>,
    [values, groups]: [usize; 2],
    [value_passed, group_passed]: [&Tensor<R>; 2],
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let active = owner.numel() / values.max(1);
    let known = Tensor::<R>::ones(&[lanes], DType::F64, owner.device())?;
    if active == 0 {
        return Ok(known);
    }
    let per_lane = |passed: &Tensor<R>, per: usize| -> numr::error::Result<Tensor<R>> {
        c.min(&passed.reshape(&[active, per])?, &[1], false)
    };
    let passed = c.minimum(
        &per_lane(value_passed, values)?,
        &per_lane(group_passed, groups)?,
    )?;
    let lane = owner
        .reshape(&[active, values])?
        .narrow(1, 0, 1)?
        .contiguous()?
        .reshape(&[active])?;
    c.index_put(&known, 0, &lane, &passed)
}
