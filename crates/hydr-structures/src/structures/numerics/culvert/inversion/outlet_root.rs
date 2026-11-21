//! Root of `HW_o(Q) - HW` in `Q` at fixed tailwater, in three stages.
//!
//! 1. Certified bisection with slack-secant probes brackets the root to [`BRACKET_TOLERANCE`],
//!    or until the coarse noise stops it. It starts from `[0, upper]`, or from a warm window
//!    around an earlier root. Each residual runs a coarse march. Points whose coarse march is
//!    unresolved run the fine march.
//! 2. Two fine marches narrow the bracket to the noise floor, see [`settle`](super::settle). A
//!    warm window as narrow as the stage 1 width starts here, and stage 2 certifies its ends.
//! 3. Lanes that stage 2 leaves wider than [`SETTLED_TOLERANCE`] bisect their narrowed bracket
//!    to that width. Points whose coarse residual holds zero then run the fine march.
//!
//! Stage 1 alone would bisect the far bracket end through the fine noise band one bit per
//! round. Stage 2 replaces those rounds with two fine marches.

use super::Selected;
use super::inlet::any;
use super::lanes::Lanes;
use super::settle::settle;
use super::warm::Window;
use crate::structures::numerics::culvert::RowNames;
use crate::structures::numerics::culvert::outlet::{self, Accuracy};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{Bounds, TensorCulvertLaw};
use crate::structures::{CulvertClient, StructureError};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps, UnaryOps,
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
    let (bounds, _) = finish(law, rows, lanes, &enclosed, &found.bounds, zero)?;
    Ok(super::Outcome {
        enclosed: found.enclosed,
        absent: found.absent,
        bounds: [
            c.where_cond(&enclosed, &bounds[0], &found.bounds[0])?,
            c.where_cond(&enclosed, &bounds[1], &found.bounds[1])?,
        ],
    })
}

/// Encloses the root on the warm lanes of `window` inside their windows.
///
/// Stage 1 starts in the window. A window no wider than [`BRACKET_TOLERANCE`] skips stage 1,
/// and stage 2 certifies the signs at its ends. A warm lane whose window holds no certified root
/// stays outside `enclosed`.
pub(super) fn solve_warm<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    window: &Window<R>,
) -> Result<super::Outcome<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let zero = &barrel.zero;
    let one = &barrel.one;
    let warm = &window.warm;
    let low = c.where_cond(warm, &window.bounds[0], zero)?;
    let high = c.where_cond(warm, &window.bounds[1], zero)?;
    let narrow = c.mul(
        warm,
        &c.le(
            &c.sub(&high, &low)?,
            &c.mul_scalar(&c.abs(&high)?, BRACKET_TOLERANCE)?,
        )?,
    )?;
    let staged = c.mul(warm, &c.sub(one, &narrow)?)?;
    let (enclosed, bracket) = if any(c, &staged)? {
        let coarse = bisect(
            law,
            rows,
            lanes,
            &c.where_cond(&staged, &low, zero)?,
            &c.where_cond(&staged, &high, zero)?,
            BRACKET_TOLERANCE,
            Refine::Never,
        )?;
        let pick = |on: &Tensor<R>, off: &Tensor<R>| c.where_cond(&narrow, on, off);
        (
            c.maximum(&narrow, &c.mul(&staged, &coarse.enclosed)?)?,
            [
                pick(&low, &coarse.bounds[0])?,
                pick(&high, &coarse.bounds[1])?,
            ],
        )
    } else {
        (narrow.clone(), [low, high])
    };
    let (bounds, verified) = finish(law, rows, lanes, &enclosed, &bracket, &narrow)?;
    // A narrow window whose fine end residuals leave a sign open may miss the root.
    let held = c.mul(
        &enclosed,
        &c.sub(one, &c.mul(&narrow, &c.sub(one, &verified)?)?)?,
    )?;
    Ok(super::Outcome {
        absent: zero.clone(),
        bounds: [
            c.where_cond(&held, &bounds[0], &bracket[0])?,
            c.where_cond(&held, &bounds[1], &bracket[1])?,
        ],
        enclosed: held,
    })
}

/// Stages 2 and 3 on the `enclosed` lanes from `bracket`. Returns the bounds and the lanes whose
/// bracket ends certify their fine signs.
///
/// Stage 3 skips the `unchecked` lanes whose ends leave a sign open: their bracket may miss the
/// root.
fn finish<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    enclosed: &Tensor<R>,
    bracket: &Bounds<R>,
    unchecked: &Tensor<R>,
) -> Result<(Bounds<R>, Tensor<R>), StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let zero = &barrel.zero;
    let one = &barrel.one;
    let settled = settle(law, rows, lanes, enclosed, bracket)?;
    let narrowed = settled.bounds;
    let [low, high] = &narrowed;
    // A straddled bracket sits at the noise floor already. Another solve cannot narrow it.
    let scale = c.mul_scalar(&c.abs(high)?, SETTLED_TOLERANCE)?;
    let wide = c.gt(&c.sub(high, low)?, &scale)?;
    let open = c.sub(one, &settled.straddled)?;
    let doubtful = c.mul(unchecked, &c.sub(one, &settled.verified)?)?;
    let trusted = c.mul(enclosed, &c.sub(one, &doubtful)?)?;
    let loose = c.mul(&c.mul(&trusted, &wide)?, &open)?;
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
    Ok((bounds, settled.verified))
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
