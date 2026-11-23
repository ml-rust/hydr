//! Open-section geometry at represented depths above the bed.
//!
//! | Rows | Branch |
//! | --- | --- |
//! | `y = 0` | `A = P_w = M = 0`, `T = P = b`, `P_n = rho^(3/2) b` |
//! | box, `0 < y < D'` | `A = B y`, `T = B`, `P_w = 2 y`, `M = B y^2 / 2` |
//! | arc or circle, `0 < y < D'` | band formulas from the bed |
//! | `y >= D'` | full state, with moment `M(D') + A_full (y - D')` |
//!
//! `P`, `P_n` and `R` follow from `A`, `P_w` and the bed width on every row.

use super::frame;
use crate::structures::numerics::culvert::arcs::cap::Shared;
use crate::structures::numerics::culvert::section::{Layers, Quantities, band_at_depth, hull};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{Bounds, FloorRows, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Open-section enclosures at one depth per row.
pub(crate) struct FloorQuantities<R: Runtime<DType = DType>> {
    /// Flow area `A`, in square metres.
    pub(crate) area: Interval<R>,
    /// Free-surface width `T`, in metres. Zero in the full state.
    pub(crate) top_width: Interval<R>,
    /// Wall perimeter `P_w`, in metres.
    pub(crate) wall: Interval<R>,
    /// Bed width `b`, in metres.
    pub(crate) bed: Interval<R>,
    /// Wetted perimeter `P = P_w + b`, in metres.
    pub(crate) perimeter: Interval<R>,
    /// Weighted perimeter `P_n = P_w + rho^(3/2) b`, in metres.
    pub(crate) weighted: Interval<R>,
    /// Hydraulic radius `A / P`, in metres.
    pub(crate) radius: Interval<R>,
    /// Hydrostatic moment about the free surface, in cubic metres.
    pub(crate) moment: Interval<R>,
}

/// Open-section enclosures per row with the barrel state code of each row.
pub(crate) struct FloorSection<R: Runtime<DType = DType>> {
    /// `BARREL_DRY`, `BARREL_FREE` or `BARREL_FULL` per row.
    pub(crate) state: Tensor<R>,
    pub(crate) geometry: FloorQuantities<R>,
}

/// Area, width, wall perimeter and moment before the bed terms join.
pub(super) struct Core<R: Runtime<DType = DType>> {
    pub(super) area: Interval<R>,
    pub(super) top_width: Interval<R>,
    pub(super) wall: Interval<R>,
    pub(super) moment: Interval<R>,
}

/// Encloses the open-section geometry of `floor` at the represented depths `depth` above
/// each bed, in metres.
///
/// Rows with `depth >= D'` are full. A negative or NaN depth yields inverted bounds, which
/// reject the enclosing call at its completion check. A context with a frame runs in it.
pub(crate) fn at_depth<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    floor: &FloorRows<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<FloorSection<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let (state, core) = match &barrel.frame {
        Some(rows) => frame::core(a, barrel, floor, rows, depth)?,
        None => core(a, barrel, floor, depth)?,
    };
    Ok(FloorSection {
        state,
        geometry: finish(a, floor, core)?,
    })
}

/// `P`, `P_n` and `R` from the core quantities and the bed width.
fn finish<R>(
    a: &mut Arithmetic<'_, R>,
    floor: &FloorRows<R>,
    core: Core<R>,
) -> ArithmeticResult<FloorQuantities<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let enter = |a: &mut Arithmetic<'_, R>, [lower, upper]: &Bounds<R>| a.bounds(lower, upper);
    let bed = enter(a, &floor.bed_width)?;
    let weighted_bed = enter(a, &floor.weighted_bed)?;
    let perimeter = a.add(&core.wall, &bed)?;
    let weighted = a.add(&core.wall, &weighted_bed)?;
    let radius = a.div_positive_interval(&core.area, &perimeter)?;
    Ok(FloorQuantities {
        area: core.area,
        top_width: core.top_width,
        wall: core.wall,
        bed,
        perimeter,
        weighted,
        radius,
        moment: core.moment,
    })
}

/// [`at_depth`] before the bed terms, on the rows of `floor` as they are.
pub(super) fn core<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    floor: &FloorRows<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<(Tensor<R>, Core<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = &barrel.zero;
    let valid = c.ge(depth, zero)?;
    let dry = c.eq(depth, zero)?;
    let full_rows = c.ge(depth, &floor.top)?;
    let free = c.mul(&c.gt(depth, zero)?, &c.lt(depth, &floor.top)?)?;
    let box_rows = c.mul(&free, &floor.boxed)?;
    let band_rows = c.sub(&free, &box_rows)?;

    let rectangle = open_box(a, barrel, &c.where_cond(&box_rows, depth, &floor.half_top)?)?;
    let bands = banded(a, barrel, floor, depth, &band_rows)?;
    let full = full_state(a, floor, &c.where_cond(&full_rows, depth, &floor.top)?)?;

    let layers = [(&box_rows, &rectangle), (&full_rows, &full)];
    let rows = Selection {
        valid: &valid,
        dry: &dry,
        base: &bands,
        layers: &layers,
        one: &barrel.one,
        zero,
    };
    let nothing = [zero.clone(), zero.clone()];
    let geometry = Core {
        area: rows.select(a, |q| &q.area, &nothing)?,
        top_width: rows.select(a, |q| &q.top_width, &floor.bed_width)?,
        wall: rows.select(a, |q| &q.wall, &nothing)?,
        moment: rows.select(a, |q| &q.moment, &nothing)?,
    };
    let state = c.where_cond(
        &full_rows,
        &barrel.constant(codes::BARREL_FULL)?,
        &c.where_cond(
            &dry,
            &barrel.constant(codes::BARREL_DRY)?,
            &barrel.constant(codes::BARREL_FREE)?,
        )?,
    )?;
    Ok((state, geometry))
}

/// Per-row branch choice: later layers override the base, then dry and rejected rows.
struct Selection<'s, R: Runtime<DType = DType>> {
    valid: &'s Tensor<R>,
    dry: &'s Tensor<R>,
    base: &'s Core<R>,
    layers: &'s [(&'s Tensor<R>, &'s Core<R>)],
    one: &'s Tensor<R>,
    zero: &'s Tensor<R>,
}

impl<R: Runtime<DType = DType>> Selection<'_, R> {
    fn select(
        &self,
        a: &mut Arithmetic<'_, R>,
        pick: fn(&Core<R>) -> &Interval<R>,
        [dry_lower, dry_upper]: &Bounds<R>,
    ) -> ArithmeticResult<Interval<R>>
    where
        R::Client: BarrelClient<R>,
    {
        let c = a.client();
        let base = pick(self.base);
        let mut lower = base.lower().clone();
        let mut upper = base.upper().clone();
        for (rows, branch) in self.layers {
            let value = pick(branch);
            lower = c.where_cond(rows, value.lower(), &lower)?;
            upper = c.where_cond(rows, value.upper(), &upper)?;
        }
        let lower = c.where_cond(self.dry, dry_lower, &lower)?;
        let upper = c.where_cond(self.dry, dry_upper, &upper)?;
        // Inverted bounds on rejected rows fail the completion check.
        let lower = c.where_cond(self.valid, &lower, self.one)?;
        let upper = c.where_cond(self.valid, &upper, self.zero)?;
        a.bounds(&lower, &upper)
    }
}

/// Box closed forms above the bed at nonnegative depths `depth`.
fn open_box<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<Core<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let y = a.point(depth)?;
    let span = a.point(&barrel.span)?;
    let area = a.mul(&span, &y)?;
    let two = a.point(&barrel.constant(2.0)?)?;
    let wall = a.mul(&two, &y)?;
    let half = a.point(&barrel.constant(0.5)?)?;
    let first = a.mul(&area, &y)?;
    let moment = a.mul(&first, &half)?;
    Ok(Core {
        area,
        top_width: span,
        wall,
        moment,
    })
}

/// Band geometry of the free arc and circle rows `rows` (1 where `0 < y < D'`).
fn banded<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    floor: &FloorRows<R>,
    depth: &Tensor<R>,
    rows: &Tensor<R>,
) -> ArithmeticResult<Core<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let [_, corner, top] = &floor.bands;
    // A band serves a row whose depth may lie in its closed depth range.
    let to_corner = c.le(depth, &corner.start_depth[1])?;
    let from_corner = c.ge(depth, &corner.start_depth[0])?;
    let to_top = c.le(depth, &top.start_depth[1])?;
    let from_top = c.ge(depth, &top.start_depth[0])?;
    let members = [
        c.mul(rows, &to_corner)?,
        c.mul(rows, &c.mul(&from_corner, &to_top)?)?,
        c.mul(rows, &from_top)?,
    ];
    let shared = Shared::new(a, &barrel.one, &barrel.series)?;
    let mut serve = |k: usize| -> ArithmeticResult<Quantities<R>> {
        let operand = c.where_cond(&members[k], depth, &floor.middles[k])?;
        band_at_depth(
            a,
            &shared,
            &floor.bands[k],
            &floor.start_widths[k],
            &operand,
        )
    };
    let evaluated = [serve(0)?, serve(1)?, serve(2)?];
    let [m0, m1, m2] = &members;
    let [q0, q1, q2] = &evaluated;
    let layers: Layers<'_, R> = [(m0, q0), (m1, q1), (m2, q2)];
    // No width exceeds the widest width, so ranges across the widest depth hold every point.
    let width = hull(a, &layers, |q| &q.top_width)?;
    let capped = c.minimum(width.upper(), &floor.widest_width)?;
    Ok(Core {
        area: hull(a, &layers, |q| &q.area)?,
        top_width: a.bounds(width.lower(), &capped)?,
        // The band perimeter is the wall: every band starts with the wall at the bed.
        wall: hull(a, &layers, |q| &q.perimeter)?,
        moment: hull(a, &layers, |q| &q.moment)?,
    })
}

/// Full open-section state at depths `depth >= D'`.
pub(super) fn full_state<R>(
    a: &mut Arithmetic<'_, R>,
    floor: &FloorRows<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<Core<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let enter = |a: &mut Arithmetic<'_, R>, [lower, upper]: &Bounds<R>| a.bounds(lower, upper);
    let area = enter(a, &floor.full_area)?;
    let wall = enter(a, &floor.full_wall)?;
    let crown = enter(a, &floor.full_moment)?;
    let y = a.point(depth)?;
    let top = a.point(&floor.top)?;
    let above = a.sub(&y, &top)?;
    let pressed = a.mul(&area, &above)?;
    let moment = a.add(&crown, &pressed)?;
    let zero = Tensor::<R>::zeros(floor.top.shape(), DType::F64, floor.top.device())?;
    Ok(Core {
        area,
        top_width: a.point(&zero)?,
        wall,
        moment,
    })
}
