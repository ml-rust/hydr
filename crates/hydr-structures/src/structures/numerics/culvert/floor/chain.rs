//! Open sections of tangent-arc barrels: the bed band from the bed point, then the bands above.
//!
//! Band `j` holds the bed when `y_j <= h < y_(j+1)`. A bed inside the junction enclosure of
//! `y_j` takes band `j` with the hull of the bed points on both arcs that meet there. The arcs
//! are tangent at the junction, so both give the same point within that enclosure.
//!
//! Bands below the bed band repeat it with depth range `[0, 0]`, so no free depth reaches them.
//! A flat-floor arch at `h = 0` keeps its floor normal `psi_0` and width `B`.

use super::bed::{BedPoint, Circle, End, Start, advance, bed_point, hull_where, nonnegative};
use super::layout::{Candidate, copy_band, ends, pick, pick_band};
use crate::structures::numerics::culvert::arcs::ArcBand;
use crate::structures::numerics::culvert::arcs::cap::Shared;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose};
use crate::structures::tensor::culvert::{ArcRows, Bounds};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// One base band inside the interval call.
struct Entered<R: Runtime<DType = DType>> {
    centre_x: Interval<R>,
    centre_y: Interval<R>,
    radius: Interval<R>,
    angle: Interval<R>,
    sin: Interval<R>,
    cos: Interval<R>,
    depth: Interval<R>,
    low: Interval<R>,
    high: Interval<R>,
    width: Interval<R>,
}

impl<R: Runtime<DType = DType>> Entered<R> {
    fn circle(&self) -> Circle<'_, R> {
        Circle {
            centre_x: &self.centre_x,
            centre_y: &self.centre_y,
            radius: &self.radius,
            low: &self.low,
            high: &self.high,
        }
    }
}

/// Row masks that decide the bed band.
struct BedBands<'m, R: Runtime<DType = DType>> {
    /// Band `j` serves the bed where `selected[j]` is 1.
    selected: &'m [Tensor<R>; 3],
    /// Where 1, the bed lies inside the junction enclosure at the start of band `j`.
    junction: &'m [Tensor<R>; 3],
    /// 1 on flat-floor arch rows with the bed on the floor.
    on_floor: &'m Tensor<R>,
}

/// Encloses the open section of every arc row of `rows`, with the bed `bed` on `floored` rows.
///
/// Rows without an arc floor hold a harmless open section of their arc layout.
pub(super) fn arc_candidate<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    rows: &ArcRows<R>,
    floored: &Tensor<R>,
    bed: &Tensor<R>,
) -> ArithmeticResult<Candidate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let beds = c.mul(floored, &rows.arced)?;
    let [_, corner, top] = &rows.bands;
    let in_top = c.mul(&beds, &c.ge(bed, &top.start_depth[0])?)?;
    let from_corner = c.mul(&beds, &c.ge(bed, &corner.start_depth[0])?)?;
    let in_corner = c.sub(&from_corner, &c.mul(&from_corner, &in_top)?)?;
    let in_bottom = c.sub(&c.sub(&beds, &in_top)?, &in_corner)?;
    let junction = [
        shared.zero.clone(),
        c.mul(&in_corner, &c.le(bed, &corner.start_depth[1])?)?,
        c.mul(&in_top, &c.le(bed, &top.start_depth[1])?)?,
    ];
    let on_floor = c.mul(&c.mul(&beds, &rows.floored)?, &c.eq(bed, &shared.zero)?)?;
    let selected = [in_bottom, in_corner, in_top];
    let masks = BedBands {
        selected: &selected,
        junction: &junction,
        on_floor: &on_floor,
    };
    let first = chain(a, shared, rows, 0, bed, &masks)?;
    let second = chain(a, shared, rows, 1, bed, &masks)?;
    let third = chain(a, shared, rows, 2, bed, &masks)?;
    let [_, corner_rows, top_rows] = &selected;
    let upper = merge(a, top_rows, &third, &second)?;
    merge(a, &c.add(corner_rows, top_rows)?, &upper, &first)
}

/// `on` where `mask` is 1 and `off` elsewhere, field by field.
fn merge<R>(
    a: &Arithmetic<'_, R>,
    mask: &Tensor<R>,
    on: &Candidate<R>,
    off: &Candidate<R>,
) -> ArithmeticResult<Candidate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let [b0, b1, b2] = &on.bands;
    let [o0, o1, o2] = &off.bands;
    let [w0, w1, w2] = &on.widths;
    let [v0, v1, v2] = &off.widths;
    Ok(Candidate {
        bands: [
            pick_band(c, mask, b0, o0)?,
            pick_band(c, mask, b1, o1)?,
            pick_band(c, mask, b2, o2)?,
        ],
        widths: [
            pick(c, mask, w0, v0)?,
            pick(c, mask, w1, v1)?,
            pick(c, mask, w2, v2)?,
        ],
        bed_width: pick(c, mask, &on.bed_width, &off.bed_width)?,
        widest_depth: pick(c, mask, &on.widest_depth, &off.widest_depth)?,
        widest_width: c.where_cond(mask, &on.widest_width, &off.widest_width)?,
        full_area: pick(c, mask, &on.full_area, &off.full_area)?,
        full_wall: pick(c, mask, &on.full_wall, &off.full_wall)?,
        full_moment: pick(c, mask, &on.full_moment, &off.full_moment)?,
    })
}

/// The open section with the bed in band `j`. Rows whose bed lies elsewhere take a depth inside
/// band `j`.
fn chain<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    rows: &ArcRows<R>,
    j: usize,
    bed: &Tensor<R>,
    masks: &BedBands<'_, R>,
) -> ArithmeticResult<Candidate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let (Some(selected), Some(middle), Some(junction)) = (
        masks.selected.get(j),
        rows.middles.get(j),
        masks.junction.get(j),
    ) else {
        return Err(missing("a band past the crown band"));
    };
    let depth = c.where_cond(selected, bed, middle)?;
    let h = a.point(&depth)?;
    let zero = a.point(&shared.zero)?;
    let mut base = Vec::with_capacity(3);
    for (band, width) in rows.bands.iter().zip(&rows.start_widths) {
        base.push(enter(a, band, width)?);
    }
    let Some(holder) = base.get(j) else {
        return Err(missing("the bed band"));
    };
    let mut point = bed_point(a, shared, &holder.circle(), &h)?;
    if let Some(below) = j.checked_sub(1).and_then(|k| base.get(k)) {
        let other = bed_point(a, shared, &below.circle(), &h)?;
        point = hulled(a, junction, &holder.radius, &point, &other)?;
    }
    if j == 2 {
        point = floor_start(a, shared, masks.on_floor, holder, point)?;
    }

    let mut bands: Vec<ArcBand<R>> = Vec::with_capacity(3);
    let mut widths: Vec<Bounds<R>> = Vec::with_capacity(3);
    let mut area = a.point(&shared.zero)?;
    let mut wall = a.point(&shared.zero)?;
    let mut moment = a.point(&shared.zero)?;
    for (k, band) in base.iter().enumerate().skip(j) {
        let at_bed = k == j;
        let start_depth = if at_bed {
            a.point(&shared.zero)?
        } else {
            let lifted = a.sub(&band.depth, &h)?;
            nonnegative(a, shared, &lifted)?
        };
        let (sin, cos, angle, width) = if at_bed {
            (&point.sin, &point.cos, &point.angle, &point.width)
        } else {
            (&band.sin, &band.cos, &band.angle, &band.width)
        };
        let low = a.sub(&band.low, &h)?;
        let high = a.sub(&band.high, &h)?;
        let centre_y = a.sub(&band.centre_y, &h)?;
        bands.push(ArcBand {
            centre_x: ends(&band.centre_x),
            centre_y: ends(&centre_y),
            radius: ends(&band.radius),
            start_angle: ends(angle),
            start_sin: ends(sin),
            start_cos: ends(cos),
            start_depth: ends(&start_depth),
            start_area: ends(&area),
            start_perimeter: ends(&wall),
            start_moment: ends(&moment),
            circle_low: ends(&low),
            circle_high: ends(&high),
        });
        widths.push(ends(width));

        let from = if at_bed { &h } else { &band.depth };
        let (end_sin, end_width, rise) = match base.get(k + 1) {
            Some(next) => {
                let rise = a.sub(&next.depth, from)?;
                (&next.sin, &next.width, nonnegative(a, shared, &rise)?)
            }
            None => {
                let rise = a.sub(&band.high, from)?;
                (&zero, &zero, nonnegative(a, shared, &rise)?)
            }
        };
        let reach = if at_bed {
            a.bounds(point.reach.lower(), point.reach.upper())?
        } else {
            a.mul(&band.radius, &band.sin)?
        };
        let reached = advance(
            a,
            shared,
            &band.radius,
            &Start {
                sin,
                cos,
                reach: &reach,
                width,
                area: &area,
                wall: &wall,
                moment: &moment,
            },
            &End {
                sin: end_sin,
                width: end_width,
                rise: &rise,
            },
        )?;
        area = reached.area;
        wall = reached.wall;
        moment = reached.moment;
    }
    // Bands below the bed band repeat it.
    let (Some(bed_band), Some(bed_width)) = (bands.first(), widths.first()) else {
        return Err(missing("the bed band"));
    };
    let mut repeated: Vec<ArcBand<R>> = (0..j).map(|_| copy_band(bed_band)).collect();
    let mut repeated_widths = vec![bed_width.clone(); j];
    let bed_width = bed_width.clone();
    repeated.extend(bands);
    repeated_widths.extend(widths);
    let (Ok(bands), Ok(widths)) = (
        <[ArcBand<R>; 3]>::try_from(repeated),
        <[Bounds<R>; 3]>::try_from(repeated_widths),
    ) else {
        return Err(missing("a band of the open section"));
    };

    let [widest_low, widest_high] = &rows.widest_depth;
    let widest = a.bounds(widest_low, widest_high)?;
    let widest = a.sub(&widest, &h)?;
    let widest_depth = nonnegative(a, shared, &widest)?;
    // A bed at or above the widest depth is the widest surface.
    let above = c.ge(&depth, widest_high)?;
    let widest_width = c.where_cond(&above, point.width.upper(), &rows.widest_width)?;
    Ok(Candidate {
        bands,
        widths,
        bed_width,
        widest_depth: ends(&widest_depth),
        widest_width,
        full_area: ends(&area),
        full_wall: ends(&wall),
        full_moment: ends(&moment),
    })
}

/// An internal error for a band the three-band layout lacks.
fn missing(what: &str) -> solvr::common::arithmetic::ArithmeticError {
    numr::error::Error::InvalidArgument {
        arg: "band",
        reason: format!("{what} is missing from a three-band arc layout"),
    }
    .into()
}

/// One stored band and its start width inside the interval call.
fn enter<R>(
    a: &mut Arithmetic<'_, R>,
    band: &ArcBand<R>,
    width: &Bounds<R>,
) -> ArithmeticResult<Entered<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut take = |[lower, upper]: &Bounds<R>| a.bounds(lower, upper);
    Ok(Entered {
        centre_x: take(&band.centre_x)?,
        centre_y: take(&band.centre_y)?,
        radius: take(&band.radius)?,
        angle: take(&band.start_angle)?,
        sin: take(&band.start_sin)?,
        cos: take(&band.start_cos)?,
        depth: take(&band.start_depth)?,
        low: take(&band.circle_low)?,
        high: take(&band.circle_high)?,
        width: take(width)?,
    })
}

/// The hull of both bed points where `mask` is 1, with the reach on the bed band's `radius`.
fn hulled<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    radius: &Interval<R>,
    point: &BedPoint<R>,
    other: &BedPoint<R>,
) -> ArithmeticResult<BedPoint<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    // The reach is r sin psi_s of the bed band's own arc.
    let reach = a.mul(radius, &other.sin)?;
    Ok(BedPoint {
        sin: hull_where(a, mask, &point.sin, &other.sin)?,
        cos: hull_where(a, mask, &point.cos, &other.cos)?,
        angle: hull_where(a, mask, &point.angle, &other.angle)?,
        reach: hull_where(a, mask, &point.reach, &reach)?,
        width: hull_where(a, mask, &point.width, &other.width)?,
    })
}

/// The floor normal and width of a flat-floor arch where `mask` is 1.
fn floor_start<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    mask: &Tensor<R>,
    band: &Entered<R>,
    point: BedPoint<R>,
) -> ArithmeticResult<BedPoint<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    // The arch arc is centred on the axis, so its reach at the floor is B/2.
    let reach = a.mul(&band.width, &shared.half)?;
    Ok(BedPoint {
        sin: choose(a, mask, &band.sin, &point.sin)?,
        cos: choose(a, mask, &band.cos, &point.cos)?,
        angle: choose(a, mask, &band.angle, &point.angle)?,
        reach: choose(a, mask, &reach, &point.reach)?,
        width: choose(a, mask, &band.width, &point.width)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::numerics::culvert::arcs::meet;
    use crate::structures::tensor::culvert::{SeriesTensors, TensorBarrelContext};
    use crate::structures::{CulvertBarrel, StructureError};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    type Cpu = CpuRuntime;

    #[test]
    fn both_arcs_give_one_bed_point_inside_a_junction_enclosure() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = Cpu::default_client(&device);
        let pipe = CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?;
        let context = TensorBarrelContext::<Cpu>::new(client.clone(), &[pipe])?;
        let rows = context
            .arcs
            .as_ref()
            .ok_or_else(|| StructureError::definition("the pipe-arch lost its arc rows"))?;
        let [low, high] = &rows.bands[1].start_depth;
        // The frozen junction bed and each end of the first junction enclosure.
        let mut beds = vec![f64::from_bits(0x3fb9_a157_8b05_3020)];
        beds.extend(low.try_to_vec::<f64>()?);
        beds.extend(high.try_to_vec::<f64>()?);
        for bed in beds {
            let depth = Tensor::<Cpu>::from_slice(&[bed], &[1], &device)?;
            let gaps = with_interval_arithmetic(&client, DType::F64, "junction bed", |a| {
                let series = SeriesTensors::constant(1, &device)?;
                let shared = Shared::new(a, &context.one, &series)?;
                let h = a.point(&depth)?;
                let bottom = enter(a, &rows.bands[0], &rows.start_widths[0])?;
                let corner = enter(a, &rows.bands[1], &rows.start_widths[1])?;
                let below = bed_point(a, &shared, &bottom.circle(), &h)?;
                let above = bed_point(a, &shared, &corner.circle(), &h)?;
                // Disjoint enclosures give inverted bounds, which fail the call.
                Ok([
                    meet(a, &below.sin, &above.sin)?,
                    meet(a, &below.cos, &above.cos)?,
                    meet(a, &below.width, &above.width)?,
                ])
            });
            assert!(gaps.is_ok(), "bed {bed:e}: the bed points are disjoint");
        }
        Ok(())
    }
}
