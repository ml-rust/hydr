//! Root of `HW_o(Q) - HW` in `Q` at fixed tailwater, in three stages.
//!
//! 1. Certified bisection with slack-secant probes brackets the root to [`BRACKET_TOLERANCE`],
//!    or until the coarse noise stops it. Each residual runs a coarse march. Points whose coarse
//!    march is unresolved run the fine march.
//! 2. One fine march at both bracket ends gives a secant root `x` and the residual noise. One
//!    more fine march at `x - d` and `x + d` then certifies the root between them, where `d`
//!    covers a few noise widths over the secant slope and the secant's curvature miss.
//! 3. Lanes that stage 2 leaves wider than [`SETTLED_TOLERANCE`] bisect their narrowed bracket
//!    to that width. Points whose coarse residual holds zero then run the fine march.
//!
//! Stage 1 alone would bisect the far bracket end through the fine noise band one bit per
//! round. Stage 2 replaces those rounds with two fine marches.

use super::Selected;
use super::inlet::any;
use super::lanes::Lanes;
use crate::structures::numerics::culvert::RowNames;
use crate::structures::numerics::culvert::outlet::{self, Accuracy};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{Bounds, TensorCulvertLaw};
use crate::structures::{CulvertClient, StructureError};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps, ShapeOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::optimize::{CertifiedBisectionAlgorithms, CertifiedRootValues};
use std::cell::Cell;

/// Upper headwater bound `2^500` m of an unresolved march point, far above any admitted barrel.
const UNKNOWN: f64 = f64::from_bits(0x5f30_0000_0000_0000);

/// Solver slope floor `2^-600`, exact.
const SLOPE_FLOOR: f64 = f64::from_bits(0x1a70_0000_0000_0000);

/// Relative bracket width `2^-22` at which stage 1 hands over to stage 2.
const BRACKET_TOLERANCE: f64 = 1.0 / 4_194_304.0;

/// Relative width `2^-40` that a settled enclosure meets.
const SETTLED_TOLERANCE: f64 = 1.0 / 1_099_511_627_776.0;

/// Noise widths over the secant slope on each side of the secant root.
const NOISE_MARGIN: f64 = 4.0;

/// Relative floor of the stage 2 offset, `2^-48` of the secant root.
const OFFSET_FLOOR: f64 = 1.0 / 281_474_976_710_656.0;

/// Encloses the root on the `search` lanes, whose roots lie in `[0, upper]`.
pub(super) fn solve<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    search: &Tensor<R>,
    upper: &Tensor<R>,
) -> Result<super::Outcome<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let zero = &barrel.zero;
    let upper = c.where_cond(search, upper, zero)?;
    let coarse = bisect(
        law,
        rows,
        lanes,
        zero,
        &upper,
        BRACKET_TOLERANCE,
        Refine::Never,
    )?;
    // A lane whose coarse march left the bracket unresolved solves again with fine marches
    // wherever a coarse residual holds zero.
    let settled_status = c.maximum(&coarse.enclosed, &coarse.absent)?;
    let unsure = c.mul(search, &c.sub(&barrel.one, &settled_status)?)?;
    let found = if any(c, &unsure)? {
        let top = c.where_cond(&unsure, &upper, zero)?;
        let again = bisect(
            law,
            rows,
            lanes,
            zero,
            &top,
            BRACKET_TOLERANCE,
            Refine::Uncertain,
        )?;
        let pick = |on: &Tensor<R>, off: &Tensor<R>| c.where_cond(&unsure, on, off);
        super::Outcome {
            enclosed: pick(&again.enclosed, &coarse.enclosed)?,
            absent: pick(&again.absent, &coarse.absent)?,
            bounds: [
                pick(&again.bounds[0], &coarse.bounds[0])?,
                pick(&again.bounds[1], &coarse.bounds[1])?,
            ],
        }
    } else {
        coarse
    };
    let enclosed = c.mul(search, &found.enclosed)?;
    let settled = settle(law, rows, lanes, &enclosed, &found.bounds)?;
    let narrowed = settled.bounds;
    let [low, high] = &narrowed;
    // A straddled bracket sits at the noise floor already. Another solve cannot narrow it.
    let scale = c.mul_scalar(&c.abs(high)?, SETTLED_TOLERANCE)?;
    let wide = c.gt(&c.sub(high, low)?, &scale)?;
    let open = c.sub(&barrel.one, &settled.straddled)?;
    let loose = c.mul(&c.mul(&enclosed, &wide)?, &open)?;
    let bounds = if any(c, &loose)? {
        let low = c.where_cond(&loose, low, zero)?;
        let high = c.where_cond(&loose, high, zero)?;
        let again = bisect(
            law,
            rows,
            lanes,
            &low,
            &high,
            SETTLED_TOLERANCE,
            Refine::Uncertain,
        )?;
        // A lane whose second solve did not enclose keeps its stage 2 bracket.
        let take = c.mul(&loose, &again.enclosed)?;
        [
            c.where_cond(&take, &again.bounds[0], &narrowed[0])?,
            c.where_cond(&take, &again.bounds[1], &narrowed[1])?,
        ]
    } else {
        narrowed
    };
    Ok(super::Outcome {
        enclosed: found.enclosed,
        absent: found.absent,
        bounds: [
            c.where_cond(&enclosed, &bounds[0], &found.bounds[0])?,
            c.where_cond(&enclosed, &bounds[1], &found.bounds[1])?,
        ],
    })
}

/// Points of a residual call that run the fine march after the coarse one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Refine {
    /// No point. The coarse noise then ends stage 1.
    Never,
    /// Points whose coarse residual holds zero.
    Uncertain,
}

/// Bisects the residual on `[lower, upper]` per lane to the relative width `rtol`.
fn bisect<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    rtol: f64,
    refine: Refine,
) -> Result<super::Outcome<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let floor = barrel.constant(SLOPE_FLOOR)?;
    let widening = c.next_up(&c.mul(upper, &floor)?)?;
    let cause = Cell::new(None);
    let solved = c.bisect_certified(
        |query| match residual(
            law,
            rows,
            lanes,
            &widening,
            &query.owner,
            &query.point,
            refine,
        ) {
            Ok([lower, upper]) => Ok(CertifiedRootValues { lower, upper }),
            Err(error) => {
                let reason = error.to_string();
                cause.set(Some(error));
                Err(numr::error::Error::InvalidArgument {
                    arg: "culvert outlet inversion residual",
                    reason,
                })
            }
        },
        lower,
        upper,
        &floor,
        &super::options_with(barrel.width(), rtol),
    );
    let roots = match (solved, cause.take()) {
        (_, Some(error)) => return Err(error),
        (Ok(roots), None) => roots,
        (Err(error), None) => return Err(error.into()),
    };
    super::outcome(c, &roots, &[lower.clone(), upper.clone()])
}

/// A stage 2 bracket, with a mask of the lanes whose secant probes straddle the root.
struct Settled<R: Runtime<DType = DType>> {
    bounds: Bounds<R>,
    straddled: Tensor<R>,
}

/// Fine residuals at the bracket ends, then at the secant root plus and minus the offset.
///
/// Returns the bracket narrowed by every certified sign. Lanes outside `wanted` keep `bracket`.
fn settle<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    wanted: &Tensor<R>,
    bracket: &Bounds<R>,
) -> Result<Settled<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let zero = &barrel.zero;
    // Lanes outside `wanted` march dry at zero discharge.
    let low = c.where_cond(wanted, &bracket[0], zero)?;
    let high = c.where_cond(wanted, &bracket[1], zero)?;
    let [at_low, at_high] = pair(law, rows, lanes, &low, &high)?;
    let middle = |value: &Bounds<R>| -> numr::error::Result<Tensor<R>> {
        c.mul_scalar(&c.add(&value[0], &value[1])?, 0.5)
    };
    let (r_low, r_high) = (middle(&at_low)?, middle(&at_high)?);
    let width = c.sub(&high, &low)?;
    let rise = c.sub(&r_high, &r_low)?;
    let slope = c.div(&rise, &width)?;
    let usable = c.mul(&c.gt(&rise, zero)?, &c.gt(&width, zero)?)?;
    let step = c.div(
        &c.mul(&r_low, &width)?,
        &c.where_cond(&usable, &rise, &barrel.one)?,
    )?;
    let secant = c.sub(&low, &step)?;
    let secant = c.minimum(&c.maximum(&secant, &low)?, &high)?;
    let noise = c.maximum(
        &c.sub(&at_low[1], &at_low[0])?,
        &c.sub(&at_high[1], &at_high[0])?,
    )?;
    let offset = c.div(
        &c.mul_scalar(&noise, NOISE_MARGIN)?,
        &c.where_cond(&usable, &slope, &barrel.one)?,
    )?;
    // `HW_o` grows about as `Q^2`, so the secant misses the root by about `W^2 / (8 Q)` over a
    // bracket of width `W`. The offset covers four times that.
    let magnitude = c.maximum(&c.abs(&secant)?, &c.mul_scalar(&width, 0.5)?)?;
    let curvature = c.div(
        &c.mul_scalar(&c.mul(&width, &width)?, 0.5)?,
        &c.where_cond(&c.gt(&magnitude, zero)?, &magnitude, &barrel.one)?,
    )?;
    let offset = c.add(&offset, &curvature)?;
    let offset = c.maximum(&offset, &c.mul_scalar(&c.abs(&secant)?, OFFSET_FLOOR)?)?;
    // Without a usable secant, the probes split the bracket in thirds.
    let third = c.mul_scalar(&width, 1.0 / 3.0)?;
    let left = c.where_cond(&usable, &c.sub(&secant, &offset)?, &c.add(&low, &third)?)?;
    let right = c.where_cond(&usable, &c.add(&secant, &offset)?, &c.sub(&high, &third)?)?;
    let left = c.minimum(&c.maximum(&left, &low)?, &high)?;
    let right = c.minimum(&c.maximum(&right, &left)?, &high)?;
    let [at_left, at_right] = pair(law, rows, lanes, &left, &right)?;
    // A residual certified at most zero puts the root at or above its point, and one certified
    // at least zero puts it at or below.
    let mut lower = low.clone();
    let mut upper = high.clone();
    for (point, value) in [(&left, &at_left), (&right, &at_right)] {
        let below = c.le(&value[1], zero)?;
        let above = c.ge(&value[0], zero)?;
        lower = c.where_cond(&below, &c.maximum(&lower, point)?, &lower)?;
        upper = c.where_cond(&above, &c.minimum(&upper, point)?, &upper)?;
    }
    let ordered = c.mul(wanted, &c.le(&lower, &upper)?)?;
    // The secant probes straddle the root: the bracket is as narrow as the noise allows.
    let straddled = c.mul(
        &c.mul(&c.le(&at_left[1], zero)?, &c.ge(&at_right[0], zero)?)?,
        &usable,
    )?;
    Ok(Settled {
        bounds: [
            c.where_cond(&ordered, &lower, &bracket[0])?,
            c.where_cond(&ordered, &upper, &bracket[1])?,
        ],
        straddled: c.mul(&ordered, &straddled)?,
    })
}

/// Fine residuals of every lane at `first` and at `second`, in one march over both.
fn pair<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    first: &Tensor<R>,
    second: &Tensor<R>,
) -> Result<[Bounds<R>; 2], StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let m = barrel.width();
    let lane: Vec<i64> = (0..2 * m)
        .map(|index| i64::try_from(index % m).unwrap_or(i64::MAX))
        .collect();
    let owner = Tensor::<R>::from_slice(&lane, &[2 * m], &barrel.device)?;
    let point = c.cat(&[first, second], 0)?;
    let [lower, upper] = gap_at(law, rows, lanes, &owner, &point, Accuracy::Fine)?;
    let half = |value: &Tensor<R>, index: usize| value.narrow(0, index * m, m);
    Ok([
        [half(&lower, 0)?, half(&upper, 0)?],
        [half(&lower, 1)?, half(&upper, 1)?],
    ])
}

/// `HW_o(Q) - HW` at the solver's query points, with tiny endpoints widened.
pub(super) fn residual<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    widening: &Tensor<R>,
    owner: &Tensor<R>,
    point: &Tensor<R>,
    refine: Refine,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let c = &rows.barrel.client;
    let zero = &rows.barrel.zero;
    let zero = &c.index_select(zero, 0, owner)?;
    let tau = c.index_select(widening, 0, owner)?;
    // Most points lie far from the root, where a coarse march already certifies the sign.
    // With `Uncertain`, points whose coarse residual holds zero run the fine march. The rest
    // march dry at zero discharge.
    let [coarse_lower, coarse_upper] = gap_at(law, rows, lanes, owner, point, Accuracy::Coarse)?;
    let (lower, upper) = match refine {
        Refine::Never => (coarse_lower, coarse_upper),
        Refine::Uncertain => {
            let open = c.mul(&c.le(&coarse_lower, zero)?, &c.ge(&coarse_upper, zero)?)?;
            let fine_point = c.where_cond(&open, point, zero)?;
            let [fine_lower, fine_upper] =
                gap_at(law, rows, lanes, owner, &fine_point, Accuracy::Fine)?;
            (
                c.where_cond(&open, &fine_lower, &coarse_lower)?,
                c.where_cond(&open, &fine_upper, &coarse_upper)?,
            )
        }
    };
    let negative_tau = c.neg(&tau)?;
    let small_upper = c.mul(&c.gt(&upper, zero)?, &c.lt(&upper, &tau)?)?;
    let small_lower = c.mul(&c.lt(&lower, zero)?, &c.gt(&lower, &negative_tau)?)?;
    Ok([
        c.where_cond(&small_lower, &negative_tau, &lower)?,
        c.where_cond(&small_upper, &tau, &upper)?,
    ])
}

/// `HW_o(Q) - HW` at the discharges `point` of lanes `owner`, from one march at `accuracy`.
pub(super) fn gap_at<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    owner: &Tensor<R>,
    point: &Tensor<R>,
    accuracy: Accuracy,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let c = &rows.barrel.client;
    let take = |value: &Tensor<R>| c.index_select(value, 0, owner);
    let picked = rows.select(owner)?;
    let oriented = take(&lanes.rows)?;
    let tailwater = take(&lanes.tailwater)?;
    let headwater = take(&lanes.headwater)?;
    let barrel = &picked.barrel;
    let march = outlet::march(
        RowNames::mapped(&law.barrels, &oriented),
        barrel,
        &picked.unit,
        &picked.outlet,
        point,
        &tailwater,
        accuracy,
    )?;
    // An unresolved march point, as at a discharge too small for the critical-depth floor, holds
    // only `HW_o >= 0`. Its residual `[-HW, UNKNOWN]` lets the solver cut nothing there.
    let unresolved = c.eq(
        &march.headwater.status,
        &barrel.constant(codes::OUTLET_UNRESOLVED)?,
    )?;
    let [low, high] = &march.headwater.headwater;
    let low = c.where_cond(&unresolved, &barrel.zero, low)?;
    let high = c.where_cond(&unresolved, &barrel.constant(UNKNOWN)?, high)?;
    Ok(with_interval_arithmetic(
        c,
        DType::F64,
        "culvert outlet inversion residual",
        |a| {
            let headwater_interval = a.bounds(&low, &high)?;
            let target = a.point(&headwater)?;
            let value = a.sub(&headwater_interval, &target)?;
            Ok([value.lower().clone(), value.upper().clone()])
        },
    )?)
}
