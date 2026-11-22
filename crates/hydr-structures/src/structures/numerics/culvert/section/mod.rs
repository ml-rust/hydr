//! Barrel geometry at represented depths and over depth ranges.
//!
//! Every branch runs on every row with a harmless depth on rows it does not serve, and
//! `where_cond` selects per row. No row takes a data-dependent kernel path.
//!
//! | Rows | Branch |
//! | --- | --- |
//! | `y = 0` | exact zeros, and `T = P = B` for a box and a flat-floor arch |
//! | circle, `0 < y < D/16` | shallow series in `s = y/D` with a certified tail |
//! | circle, `D/16 <= y < D` | `acos` closed forms |
//! | box, `0 < y < D` | rectangle closed forms |
//! | arc, `0 < y < D` | band formulas, with the circle series near a three-radius invert |
//! | `y >= D` | full-state constants, with moment `A_full (y - D/2)`, or `M(D) + A_full (y - D)` on arc rows |
//!
//! A context without arc rows skips the arc branch.
//!
//! A negative or NaN depth yields inverted bounds, which reject the enclosing call at its
//! completion check.

mod arcs;
mod circular;
mod full;
mod range;
mod rectangular;
mod shallow;

pub(crate) use full::full_constants;
pub(crate) use range::over_depth;
pub(crate) use shallow::sums;

use super::{Arithmetic, BarrelClient};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Area, top width, wetted perimeter, hydraulic radius and hydrostatic moment enclosures.
pub(crate) struct Quantities<R: Runtime<DType = DType>> {
    /// Flow area `A`, in square metres.
    pub(crate) area: Interval<R>,
    /// Free-surface width `T`, in metres. Zero in the full state.
    pub(crate) top_width: Interval<R>,
    /// Wetted perimeter `P`, in metres.
    pub(crate) perimeter: Interval<R>,
    /// Hydraulic radius `A / P`, in metres.
    pub(crate) radius: Interval<R>,
    /// Hydrostatic moment `integral of (y - eta) T(eta) d eta`, in cubic metres.
    pub(crate) moment: Interval<R>,
}

/// Geometry enclosures per row with the barrel state code of each row.
pub(crate) struct SectionIntervals<R: Runtime<DType = DType>> {
    /// `BARREL_DRY`, `BARREL_FREE` or `BARREL_FULL` per row.
    pub(crate) state: Tensor<R>,
    pub(crate) geometry: Quantities<R>,
}

/// Encloses the geometry at the represented depths `depth` (`[m]`, metres).
///
/// Rows with `depth >= D` are full. The caller checks that every depth is finite and nonnegative.
/// Other rows reject the enclosing call at its completion check.
pub(crate) fn at_depth<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<SectionIntervals<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let valid = c.ge(depth, &barrel.zero)?;
    let dry = c.eq(depth, &barrel.zero)?;
    let full_rows = c.ge(depth, &barrel.rise)?;
    let free = c.mul(&c.gt(depth, &barrel.zero)?, &c.lt(depth, &barrel.rise)?)?;
    let circle = c.mul(&free, &barrel.circular)?;
    let shallow_rows = c.mul(&circle, &c.lt(depth, &barrel.shallow_top)?)?;
    let trig_rows = c.sub(&circle, &shallow_rows)?;
    let box_rows = c.mul(&free, &barrel.boxed)?;

    let shallow = shallow::evaluate(
        a,
        barrel,
        &c.where_cond(&shallow_rows, depth, &barrel.shallow_operand)?,
    )?;
    let trig = circular::evaluate(
        a,
        barrel,
        &c.where_cond(&trig_rows, depth, &barrel.half_rise)?,
    )?;
    let rectangle = rectangular::evaluate(
        a,
        barrel,
        &c.where_cond(&box_rows, depth, &barrel.half_rise)?,
    )?;
    let full = full::at_depth(a, barrel, &c.where_cond(&full_rows, depth, &barrel.rise)?)?;

    let mut edge = c.mul(&barrel.boxed, &barrel.span)?;
    let mut layers = vec![(&trig_rows, &trig), (&shallow_rows, &shallow)];
    let arcs = match &barrel.arcs {
        Some(rows) => {
            let free_arcs = c.mul(&free, &rows.arced)?;
            edge = c.add(&edge, &rows.floor_width)?;
            Some(arcs::evaluate(a, barrel, rows, depth, &free_arcs)?)
        }
        None => None,
    };
    if let Some(arcs) = &arcs {
        layers.push((&arcs.banded, &arcs.bands));
        layers.push((&arcs.shallow_rows, &arcs.shallow));
    }
    layers.push((&full_rows, &full));
    let rows = Selection {
        valid: &valid,
        dry: &dry,
        base: &rectangle,
        layers: &layers,
    };
    let geometry = Quantities {
        area: rows.select(a, barrel, |q| &q.area, &barrel.zero)?,
        top_width: rows.select(a, barrel, |q| &q.top_width, &edge)?,
        perimeter: rows.select(a, barrel, |q| &q.perimeter, &edge)?,
        radius: rows.select(a, barrel, |q| &q.radius, &barrel.zero)?,
        moment: rows.select(a, barrel, |q| &q.moment, &barrel.zero)?,
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
    Ok(SectionIntervals { state, geometry })
}

/// Per-row branch choice: later layers override earlier ones, then dry and rejected rows.
struct Selection<'s, R: Runtime<DType = DType>> {
    valid: &'s Tensor<R>,
    dry: &'s Tensor<R>,
    base: &'s Quantities<R>,
    layers: &'s [(&'s Tensor<R>, &'s Quantities<R>)],
}

impl<R: Runtime<DType = DType>> Selection<'_, R> {
    fn select(
        &self,
        a: &mut Arithmetic<'_, R>,
        barrel: &TensorBarrelContext<R>,
        pick: fn(&Quantities<R>) -> &Interval<R>,
        dry_value: &Tensor<R>,
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
        let lower = c.where_cond(self.dry, dry_value, &lower)?;
        let upper = c.where_cond(self.dry, dry_value, &upper)?;
        // Inverted bounds on rejected rows fail the completion check.
        let lower = c.where_cond(self.valid, &lower, &barrel.one)?;
        let upper = c.where_cond(self.valid, &upper, &barrel.zero)?;
        a.bounds(&lower, &upper)
    }
}

/// Encloses `T = 2 sqrt(y (D - y))` for points `y` and `d` with `0 <= y <= D`.
fn top_width<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    y: &Interval<R>,
    d: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let gap = a.sub(d, y)?;
    let product = a.mul(y, &gap)?;
    let root = a.sqrt_nonnegative(&product)?;
    let two = a.point(&barrel.constant(2.0)?)?;
    a.mul(&two, &root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{CulvertBarrel, StructureError};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    type Ends = Vec<[Vec<f64>; 2]>;

    fn ends(q: &Quantities<CpuRuntime>) -> Result<Ends, StructureError> {
        [&q.area, &q.top_width, &q.perimeter, &q.radius, &q.moment]
            .into_iter()
            .map(|i| {
                Ok([
                    i.lower().try_to_vec::<f64>()?,
                    i.upper().try_to_vec::<f64>()?,
                ])
            })
            .collect()
    }

    /// Shallow and trigonometric branches at the same depths, one interval call.
    fn branches(rise: f64, depths: &[f64]) -> Result<(Ends, Ends), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let barrels = vec![CulvertBarrel::circular(rise)?; depths.len()];
        let barrel = TensorBarrelContext::<CpuRuntime>::new(client.clone(), &barrels)?;
        let depth = Tensor::<CpuRuntime>::from_slice(depths, &[depths.len()], &device)?;
        let pair = with_interval_arithmetic(&client, DType::F64, "seam", |a| {
            let shallow = shallow::evaluate(a, &barrel, &depth)?;
            let trig = circular::evaluate(a, &barrel, &depth)?;
            Ok((shallow, trig))
        })?;
        Ok((ends(&pair.0)?, ends(&pair.1)?))
    }

    #[test]
    fn shallow_and_trigonometric_enclosures_intersect_at_the_seam() -> Result<(), StructureError> {
        for rise in [0.3, 1.0, 3.7] {
            let seam: f64 = rise / 16.0;
            let depths = [seam.next_down(), seam, seam.next_up()];
            let (shallow, trig) = branches(rise, &depths)?;
            for (quantity, (series, closed)) in shallow.iter().zip(&trig).enumerate() {
                let rows = series[0]
                    .iter()
                    .zip(&series[1])
                    .zip(closed[0].iter().zip(&closed[1]));
                for (row, ((&sl, &su), (&tl, &tu))) in rows.enumerate() {
                    assert!(sl <= su && tl <= tu, "rise={rise} row={row} q={quantity}");
                    assert!(
                        sl <= tu && tl <= su,
                        "rise={rise} row={row} quantity={quantity}: [{sl:e}, {su:e}] and [{tl:e}, {tu:e}] are disjoint"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn shallow_branch_is_tight_where_the_closed_forms_cancel() -> Result<(), StructureError> {
        // At s = 2^-20 the closed-form area width is near 18 2^-52 / theta^2 relative.
        let depth = 2f64.powi(-20);
        let (shallow, trig) = branches(1.0, &[depth])?;
        let relative = |e: &Ends, q: usize| (e[q][1][0] - e[q][0][0]) / e[q][1][0];
        for quantity in [0, 4] {
            assert!(relative(&shallow, quantity) < 1e-14, "quantity={quantity}");
            assert!(relative(&shallow, quantity) < relative(&trig, quantity) / 1e3);
        }
        Ok(())
    }
}
