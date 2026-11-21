//! Outlet-control discharge `Q_o(HW, TW)`, the inverse of the outlet march at fixed tailwater.
//!
//! - Drowned full barrel: with `TW >= D` and a pressure head that stays above the crown to the
//!   inlet, `HW_o = TW - S L + k V^2 / (2 g)` with `k = 1 + ke + 2 g n^2 L / R^(4/3)`. Then
//!   `HW - TW + S L` is the head difference `dH` of the two stages, so `Q_o = A sqrt(2 g dH / k)`
//!   in closed form. It keeps its relative precision as the heads approach each other.
//! - Every other lane: one certified bisection of `HW_o(Q) - HW` over `[0, 2 Q_i]`. A lane whose
//!   outlet headwater at `2 Q_i` stays below `HW` has `Q_o > Q_i`, so the inlet governs.
//!
//! `HW_o` rises with `Q`, but no positive slope floor is proven for the partly full profile.
//! The solver's floor is `2^-600`, and each residual endpoint of magnitude below `2^-600 U`
//! widens to that magnitude, with `U` the bracket width. The slope correction then cannot move
//! a bracket end, so every enclosure rests on certified residual signs alone.
//!
//! `HW_o(0)` is the still level `max(TW - S L, 0)`. A lane whose still level lies certainly above
//! `HW` carries exact zero under outlet control: the water stands below the outlet lip or the
//! tailwater. A lane whose still level is not certified below `HW` is level: the march cannot
//! separate its heads, so it carries exact zero too.

use super::Selected;
use super::inlet::{InletDischarge, any};
use super::lanes::Lanes;
use crate::structures::numerics::culvert::{RowNames, outlet};
use crate::structures::tensor::TensorStructureContext;
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

/// Outlet-control discharge per barrel on every lane.
pub(crate) struct OutletDischarge<R: Runtime<DType = DType>> {
    /// Enclosure of `Q_o` on `solved` lanes, `(+inf, +inf)` elsewhere.
    pub(crate) bounds: Bounds<R>,
    /// 1 where `Q_o` has an enclosure.
    pub(crate) solved: Tensor<R>,
    /// 1 where the drowned full-barrel closed form gave `Q_o`.
    pub(crate) full: Tensor<R>,
    /// 1 where the still level may reach the headwater, so the lane carries exact zero.
    pub(crate) level: Tensor<R>,
    /// 1 where the still level lies certainly above the headwater, so `Q_o = 0` exactly.
    pub(crate) dead: Tensor<R>,
}

/// Inverts the outlet curve of every flowing lane at its headwater and tailwater.
pub(crate) fn invert<R>(
    law: &TensorCulvertLaw<R>,
    context: &TensorStructureContext<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    inlet: &InletDischarge<R>,
) -> Result<OutletDischarge<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let outlet_rows = &rows.outlet;
    let c = &barrel.client;
    let one = &barrel.one;
    let flowing = c.mul(&lanes.active, &c.sub(one, &inlet.zero)?)?;
    let drowned = c.mul(&flowing, &c.ge(&lanes.tailwater, &barrel.rise)?)?;
    let (closed, full_head, still) =
        with_interval_arithmetic(c, DType::F64, "culvert outlet closed form", |a| {
            let c = a.client();
            let n = a.point(&outlet_rows.manning)?;
            let length = a.point(&outlet_rows.length)?;
            let ke = a.point(&outlet_rows.entrance)?;
            let g = a.point(&outlet_rows.gravity)?;
            let s = a.point(&outlet_rows.slope)?;
            let area = a.bounds(&barrel.full_area[0], &barrel.full_area[1])?;
            let radius = a.bounds(&barrel.full_radius[0], &barrel.full_radius[1])?;
            let four = a.point(&barrel.constant(4.0)?)?;
            let third = a.div_positive(&four, &barrel.constant(3.0)?)?;
            let ln_radius = a.log_positive(&radius)?;
            let scaled = a.mul(&third, &ln_radius)?;
            let power = super::exp(a, &scaled)?;
            let n_square = a.mul(&n, &n)?;
            let rough = a.mul(&n_square, &length)?;
            let per_head = a.div_positive_interval(&rough, &power)?;
            let two = a.point(&barrel.constant(2.0)?)?;
            let two_g = a.mul(&two, &g)?;
            let friction = a.mul(&two_g, &per_head)?;
            let unit = a.point(one)?;
            let entry = a.add(&unit, &ke)?;
            let k = a.add(&entry, &friction)?;
            // Lanes outside the closed form evaluate at a unit head difference.
            let gap = a.point(&c.where_cond(&drowned, &lanes.gap, one)?)?;
            let driving = a.mul(&two_g, &gap)?;
            let square = a.div_positive_interval(&driving, &k)?;
            let velocity = a.sqrt_nonnegative(&square)?;
            let discharge = a.mul(&area, &velocity)?;
            // Pressure head at the inlet: `TW - S L + n^2 V^2 L / R^(4/3)`.
            let tw = a.point(&lanes.tailwater)?;
            let drop = a.mul(&s, &length)?;
            let level = a.sub(&tw, &drop)?;
            let loss = a.mul(&per_head, &square)?;
            let head = a.add(&level, &loss)?;
            Ok((
                [discharge.lower().clone(), discharge.upper().clone()],
                head.lower().clone(),
                [
                    c.maximum(level.lower(), &barrel.zero)?,
                    c.maximum(level.upper(), &barrel.zero)?,
                ],
            ))
        })?;
    let full = c.mul(&drowned, &c.ge(&full_head, &barrel.rise)?)?;
    let open = c.mul(&flowing, &c.sub(one, &full)?)?;
    let dead = c.mul(&open, &c.gt(&still[0], &lanes.headwater)?)?;
    let level = c.mul(
        &c.mul(&open, &c.sub(one, &dead)?)?,
        &c.ge(&still[1], &lanes.headwater)?,
    )?;
    let candidates = c.mul(&c.mul(&open, &c.sub(one, &level)?)?, &c.sub(one, &dead)?)?;
    // One march at the inlet discharge's upper end. `HW_o(Q_i) < HW` there means
    // `Q_o > Q_i`, so the inlet governs and the bisection skips the lane.
    let search = if any(c, &candidates)? {
        let probe = c.where_cond(&candidates, &inlet.bounds[1], &barrel.zero)?;
        let lane: Vec<i64> = (0..barrel.width())
            .map(|index| i64::try_from(index).unwrap_or(i64::MAX))
            .collect();
        let owner = Tensor::<R>::from_slice(&lane, &[barrel.width()], &barrel.device)?;
        let [_, upper] = residual(law, rows, lanes, &barrel.zero, &owner, &probe)?;
        let inlet_side = c.lt(&upper, &barrel.zero)?;
        c.mul(&candidates, &c.sub(one, &inlet_side)?)?
    } else {
        candidates
    };
    let infinity = barrel.constant(f64::INFINITY)?;
    let mut bounds = [
        c.where_cond(&full, &closed[0], &infinity)?,
        c.where_cond(&full, &closed[1], &infinity)?,
    ];
    let mut solved = full.clone();
    if any(c, &search)? {
        let found = bisect(law, rows, lanes, inlet, &search)?;
        let unresolved = c.mul(
            &search,
            &c.sub(one, &c.maximum(&found.enclosed, &found.absent)?)?,
        )?;
        if let Some(lane) = super::first(&unresolved)? {
            return Err(StructureError::numerical(
                "culvert outlet headwater near the inlet discharge is unresolved or does not separate from the donor head, so the governing control is undetermined. The outlet march leaves a steep barrel unresolved where its critical depth reaches the crown. Lower the heads or supply a mild barrel",
            )
            .at(law.elements[lane]));
        }
        let take = c.mul(&search, &found.enclosed)?;
        bounds = [
            c.where_cond(&take, &found.bounds[0], &bounds[0])?,
            c.where_cond(&take, &found.bounds[1], &bounds[1])?,
        ];
        solved = c.maximum(&solved, &take)?;
    }
    let solved_upper = c.where_cond(&solved, &bounds[1], &barrel.zero)?;
    context.finite(
        &c.index_put(&context.zero, 0, &law.lanes, &solved_upper)?,
        "culvert outlet discharge enclosure",
    )?;
    Ok(OutletDischarge {
        bounds,
        solved,
        full,
        level,
        dead,
    })
}

/// Bisects `HW_o(Q) - HW` on `[0, 2 Q_i]` for the `search` lanes.
fn bisect<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    inlet: &InletDischarge<R>,
    search: &Tensor<R>,
) -> Result<super::Outcome<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let zero = &barrel.zero;
    let upper = c.where_cond(search, &c.mul_scalar(&inlet.bounds[1], 2.0)?, zero)?;
    let floor = barrel.constant(SLOPE_FLOOR)?;
    let widening = c.next_up(&c.mul(&upper, &floor)?)?;
    let cause = Cell::new(None);
    let solved = c.bisect_certified(
        |query| match residual(law, rows, lanes, &widening, &query.owner, &query.point) {
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
        zero,
        &upper,
        &floor,
        &super::options(barrel.width()),
    );
    let roots = match (solved, cause.take()) {
        (_, Some(error)) => return Err(error),
        (Ok(roots), None) => roots,
        (Err(error), None) => return Err(error.into()),
    };
    super::outcome(c, &roots, &[zero.clone(), upper])
}

/// `HW_o(Q) - HW` at the solver's query points, with tiny endpoints widened.
fn residual<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    widening: &Tensor<R>,
    owner: &Tensor<R>,
    point: &Tensor<R>,
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
    let tau = take(widening)?;
    let march = outlet::march(
        RowNames::mapped(&law.barrels, &oriented),
        &picked.barrel,
        &picked.unit,
        &picked.outlet,
        point,
        &tailwater,
    )?;
    // An unresolved march point, as at a discharge too small for the critical-depth floor, holds
    // only `HW_o >= 0`. Its residual `[-HW, UNKNOWN]` lets the solver cut nothing there.
    let barrel = &picked.barrel;
    let unresolved = c.eq(
        &march.headwater.status,
        &barrel.constant(codes::OUTLET_UNRESOLVED)?,
    )?;
    let [low, high] = &march.headwater.headwater;
    let low = c.where_cond(&unresolved, &barrel.zero, low)?;
    let high = c.where_cond(&unresolved, &barrel.constant(UNKNOWN)?, high)?;
    let value =
        with_interval_arithmetic(c, DType::F64, "culvert outlet inversion residual", |a| {
            let headwater_interval = a.bounds(&low, &high)?;
            let target = a.point(&headwater)?;
            let value = a.sub(&headwater_interval, &target)?;
            Ok([value.lower().clone(), value.upper().clone()])
        })?;
    let [lower, upper] = value;
    let zero = &picked.barrel.zero;
    let negative_tau = c.neg(&tau)?;
    let small_upper = c.mul(&c.gt(&upper, zero)?, &c.lt(&upper, &tau)?)?;
    let small_lower = c.mul(&c.lt(&lower, zero)?, &c.gt(&lower, &negative_tau)?)?;
    Ok([
        c.where_cond(&small_lower, &negative_tau, &lower)?,
        c.where_cond(&small_upper, &tau, &upper)?,
    ])
}
