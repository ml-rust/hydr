//! Free-surface geometry of barrels built from tangent circular arcs.
//!
//! | Rows | Branch |
//! | --- | --- |
//! | three-radius, `0 < y < seam` | circle series on the bottom arc of diameter `2 r_b` |
//! | inside one band's depth range | that band's formulas |
//! | inside a junction depth enclosure | hull of both bands' enclosures |
//!
//! The seam is the largest exact `D 2^-j` at or below `min(2 r_b / 16, y_1)`, so it lies below
//! the first junction. A flat-floor arch repeats its arc in all three bands from the floor, so
//! every free arch row takes the top band.

mod band;

pub(crate) use band::evaluate as band_at_depth;

use super::{Quantities, shallow, top_width};
use crate::structures::numerics::culvert::arcs::cap::Shared;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{ArcRows, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Each band's serving mask with its enclosures.
pub(crate) type Layers<'q, R> = [(&'q Tensor<R>, &'q Quantities<R>); 3];

/// Arc rows served by the band formulas and by the bottom-circle series, with their geometry.
pub(super) struct ArcBranches<R: Runtime<DType = DType>> {
    /// 1 on free arc rows at or above the seam.
    pub(super) banded: Tensor<R>,
    pub(super) bands: Quantities<R>,
    /// 1 on free three-radius rows below the seam.
    pub(super) shallow_rows: Tensor<R>,
    pub(super) shallow: Quantities<R>,
}

/// Encloses the geometry of the free arc rows `free` (1 on arc rows with `0 < y < D`).
pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    rows: &ArcRows<R>,
    depth: &Tensor<R>,
    free: &Tensor<R>,
) -> ArithmeticResult<ArcBranches<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let shallow_rows = c.mul(free, &c.lt(depth, &rows.shallow_top)?)?;
    let banded = c.sub(free, &shallow_rows)?;
    let [_, corner, top] = &rows.bands;
    // A band serves a row whose depth may lie in its closed depth range.
    let to_corner = c.le(depth, &corner.start_depth[1])?;
    let from_corner = c.ge(depth, &corner.start_depth[0])?;
    let to_top = c.le(depth, &top.start_depth[1])?;
    let from_top = c.ge(depth, &top.start_depth[0])?;
    let members = [
        c.mul(&banded, &to_corner)?,
        c.mul(&banded, &c.mul(&from_corner, &to_top)?)?,
        c.mul(&banded, &from_top)?,
    ];
    let shared = Shared::new(a, &barrel.one, &barrel.series)?;
    let mut serve = |k: usize| -> ArithmeticResult<Quantities<R>> {
        let operand = c.where_cond(&members[k], depth, &rows.middles[k])?;
        band::evaluate(a, &shared, &rows.bands[k], &rows.start_widths[k], &operand)
    };
    let evaluated = [serve(0)?, serve(1)?, serve(2)?];
    let [m0, m1, m2] = &members;
    let [q0, q1, q2] = &evaluated;
    let layers = [(m0, q0), (m1, q1), (m2, q2)];
    // No width exceeds the widest width, so ranges across the widest depth hold every point.
    let width = hull(a, &layers, |q| &q.top_width)?;
    let capped = c.minimum(width.upper(), &rows.widest_width)?;
    let capped = c.where_cond(&rows.arced, &capped, width.upper())?;
    let bands = Quantities {
        area: hull(a, &layers, |q| &q.area)?,
        top_width: a.bounds(width.lower(), &capped)?,
        perimeter: hull(a, &layers, |q| &q.perimeter)?,
        radius: hull(a, &layers, |q| &q.radius)?,
        moment: hull(a, &layers, |q| &q.moment)?,
    };

    let [bottom_middle, _, _] = &rows.middles;
    let operand = c.where_cond(&shallow_rows, depth, bottom_middle)?;
    let shallow = bottom_series(a, barrel, rows, &operand)?;
    Ok(ArcBranches {
        banded,
        bands,
        shallow_rows,
        shallow,
    })
}

/// Encloses the bottom circle's series geometry at depths below its diameter.
fn bottom_series<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    rows: &ArcRows<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<Quantities<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let y = a.point(depth)?;
    let [lower, upper] = &rows.bottom_diameter;
    let d = a.bounds(lower, upper)?;
    let width = top_width(a, barrel, &y, &d)?;
    let s = a.div_positive_interval(&y, &d)?;
    shallow::from_fraction(a, barrel, width, &d, &s)
}

/// Hull of the enclosures of every band that serves a row.
///
/// Rows no band serves keep the last band's enclosure, which the caller does not select.
pub(crate) fn hull<R>(
    a: &mut Arithmetic<'_, R>,
    layers: &Layers<'_, R>,
    pick: fn(&Quantities<R>) -> &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let [.., (_, last)] = layers;
    let mut base_lower = pick(last).lower().clone();
    let mut base_upper = pick(last).upper().clone();
    for (member, quantities) in layers {
        let value = pick(quantities);
        base_lower = c.where_cond(member, value.lower(), &base_lower)?;
        base_upper = c.where_cond(member, value.upper(), &base_upper)?;
    }
    let mut lower = base_lower.clone();
    let mut upper = base_upper.clone();
    for (member, quantities) in layers {
        let value = pick(quantities);
        lower = c.minimum(&lower, &c.where_cond(member, value.lower(), &base_lower)?)?;
        upper = c.maximum(&upper, &c.where_cond(member, value.upper(), &base_upper)?)?;
    }
    a.bounds(&lower, &upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{CulvertBarrel, StructureError};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    type Cpu = CpuRuntime;
    type Ends = Vec<[Vec<f64>; 2]>;

    /// Three-radius barrels: an AISI pipe-arch, a fitted and a narrow ellipse.
    fn three_radius() -> Result<Vec<CulvertBarrel>, StructureError> {
        Ok(vec![
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            CulvertBarrel::horizontal_ellipse(1.524, 0.9652, 0.419_475_92)?,
            CulvertBarrel::vertical_ellipse(0.5, 2.0, 0.1)?,
        ])
    }

    fn ends(q: &Quantities<Cpu>) -> Result<Ends, StructureError> {
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

    /// Requires every row of every quantity of `first` and `second` to intersect.
    fn intersect(what: &str, first: &Ends, second: &Ends) {
        for (quantity, (x, y)) in first.iter().zip(second).enumerate() {
            let rows = x[0].iter().zip(&x[1]).zip(y[0].iter().zip(&y[1]));
            for (row, ((&xl, &xu), (&yl, &yu))) in rows.enumerate() {
                assert!(
                    xl <= yu && yl <= xu,
                    "{what} row {row} quantity {quantity}: [{xl:e}, {xu:e}] and [{yl:e}, {yu:e}] are disjoint"
                );
            }
        }
    }

    /// One row per barrel and depth, with each barrel repeated for its depths.
    fn context(barrels: &[CulvertBarrel]) -> Result<TensorBarrelContext<Cpu>, StructureError> {
        let device = CpuDevice::new();
        TensorBarrelContext::<Cpu>::new(Cpu::default_client(&device), barrels)
    }

    fn arc_rows(barrel: &TensorBarrelContext<Cpu>) -> Result<&ArcRows<Cpu>, StructureError> {
        barrel
            .arcs
            .as_ref()
            .ok_or_else(|| StructureError::definition("context holds no arc rows"))
    }

    #[test]
    fn seam_is_an_exact_rise_fraction_below_the_first_junction() -> Result<(), StructureError> {
        let barrels = three_radius()?;
        let barrel = context(&barrels)?;
        let rows = arc_rows(&barrel)?;
        let seams = rows.shallow_top.try_to_vec::<f64>()?;
        let junctions = rows.bands[1].start_depth[0].try_to_vec::<f64>()?;
        let diameters = rows.bottom_diameter[0].try_to_vec::<f64>()?;
        for (((culvert, seam), junction), diameter) in
            barrels.iter().zip(seams).zip(junctions).zip(diameters)
        {
            let ratio = culvert.rise_m() / seam;
            assert_eq!(ratio, 2f64.powi(ratio.log2().round() as i32), "{culvert:?}");
            assert!(seam <= junction && seam <= diameter / 16.0, "{culvert:?}");
            assert!(2.0 * seam > junction.min(diameter / 16.0), "{culvert:?}");
        }
        Ok(())
    }

    #[test]
    fn series_and_bottom_band_meet_at_the_seam() -> Result<(), StructureError> {
        let culverts = three_radius()?;
        let first = context(&culverts)?;
        let seams = arc_rows(&first)?.shallow_top.try_to_vec::<f64>()?;
        // Each barrel at its seam and both F64 neighbours.
        let mut barrels = Vec::new();
        let mut depths = Vec::new();
        for (culvert, seam) in culverts.iter().zip(seams) {
            for depth in [seam.next_down(), seam, seam.next_up()] {
                barrels.push(*culvert);
                depths.push(depth);
            }
        }
        let barrel = context(&barrels)?;
        let rows = arc_rows(&barrel)?;
        let depth = Tensor::<Cpu>::from_slice(&depths, &[depths.len()], &barrel.device)?;
        let (series, band) =
            with_interval_arithmetic(&barrel.client, DType::F64, "arc seam", |a| {
                let series = bottom_series(a, &barrel, rows, &depth)?;
                let shared = Shared::new(a, &barrel.one, &barrel.series)?;
                let band =
                    band::evaluate(a, &shared, &rows.bands[0], &rows.start_widths[0], &depth)?;
                Ok((series, band))
            })?;
        intersect("seam", &ends(&series)?, &ends(&band)?);
        Ok(())
    }

    #[test]
    fn neighbouring_bands_meet_across_each_junction() -> Result<(), StructureError> {
        let culverts = three_radius()?;
        let first = context(&culverts)?;
        let first_rows = arc_rows(&first)?;
        for (j, k) in [(0, 1), (1, 2)] {
            let lower = first_rows.bands[k].start_depth[0].try_to_vec::<f64>()?;
            let upper = first_rows.bands[k].start_depth[1].try_to_vec::<f64>()?;
            // Each barrel at both ends of its junction enclosure and one F64 step outside.
            let mut barrels = Vec::new();
            let mut depths = Vec::new();
            for ((culvert, low), high) in culverts.iter().zip(lower).zip(upper) {
                for depth in [low.next_down(), low, high, high.next_up()] {
                    barrels.push(*culvert);
                    depths.push(depth);
                }
            }
            let barrel = context(&barrels)?;
            let rows = arc_rows(&barrel)?;
            let depth = Tensor::<Cpu>::from_slice(&depths, &[depths.len()], &barrel.device)?;
            let (below, above) =
                with_interval_arithmetic(&barrel.client, DType::F64, "arc junction", |a| {
                    let shared = Shared::new(a, &barrel.one, &barrel.series)?;
                    let below =
                        band::evaluate(a, &shared, &rows.bands[j], &rows.start_widths[j], &depth)?;
                    let above =
                        band::evaluate(a, &shared, &rows.bands[k], &rows.start_widths[k], &depth)?;
                    Ok((below, above))
                })?;
            intersect(&format!("junction {k}"), &ends(&below)?, &ends(&above)?);
        }
        Ok(())
    }
}
