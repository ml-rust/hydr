//! Open sections above a floor as march bands, in units of the open rise `D'`.
//!
//! The bed band starts at the bed normal `psi_s` with zero area and the weighted bed
//! `rho^(3/2) b` as its start perimeter, so `p` is `P_n / D'`. The bands above it chain on, and
//! bands below it repeat it with a zero depth range. Depths run from the bed. The bands come
//! from each row's frame, divided by the frame open rise, so no value underflows.
//!
//! A floored circle or arch repeats one arc in every band, so it takes the top band, as an arch
//! does. Every open row takes its harmless coordinate in the middle of its top band, which lies
//! above the bed normal.

use super::{ArcShape, MarchBand};
use crate::structures::numerics::culvert::arcs::ArcBand;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

impl<R: Runtime<DType = DType>> ArcShape<R> {
    /// The march shape of every row of `barrel`: `base` on its arc rows, and the open section
    /// on each floored circle and arc row. `barrel` must hold floor rows.
    pub(crate) fn with_floors(
        a: &mut Arithmetic<'_, R>,
        barrel: &TensorBarrelContext<R>,
        base: Option<Self>,
    ) -> ArithmeticResult<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let c = a.client();
        let framed = barrel.frame.as_ref().map_or(barrel, |frame| &frame.context);
        let (Some(rows), Some(frame_rows)) = (barrel.floor.as_deref(), framed.floor.as_deref())
        else {
            return Err(numr::error::Error::InvalidArgument {
                arg: "barrel",
                reason: "an open march shape needs the floor rows of the barrel and its frame"
                    .into(),
            }
            .into());
        };
        let open = c.sub(&rows.floored, &rows.boxed)?;
        let arch = match &barrel.arcs {
            Some(arcs) => arcs.floored.clone(),
            None => barrel.zero.clone(),
        };
        let repeated = c.mul(&open, &c.maximum(&barrel.circular, &arch)?)?;
        let top = &frame_rows.top;
        let weighted = a.bounds(&frame_rows.weighted_bed[0], &frame_rows.weighted_bed[1])?;
        let [b0, b1, b2] = &frame_rows.bands;
        let bands = [
            unit_band(a, b0, top, &weighted)?,
            unit_band(a, b1, top, &weighted)?,
            unit_band(a, b2, top, &weighted)?,
        ];
        let shape = Self::from_bands(a, &bands, [&open, &repeated, &open])?;
        let Some(base) = base else {
            return Ok(shape);
        };
        let bands = base
            .bands
            .iter()
            .zip(&shape.bands)
            .map(|(old, new)| MarchBand::choose(c, &open, new, old))
            .collect::<numr::error::Result<Vec<_>>>()?;
        Ok(Self {
            arced: c.maximum(&base.arced, &open)?,
            floored: c.where_cond(&open, &shape.floored, &base.floored)?,
            bands,
            least_radius: c.where_cond(&open, &shape.least_radius, &base.least_radius)?,
            harmless: c.where_cond(&open, &shape.harmless, &base.harmless)?,
        })
    }
}

/// One open band of a frame in units of its open rise `top`, with `weighted` added to its start
/// perimeter.
fn unit_band<R>(
    a: &mut Arithmetic<'_, R>,
    band: &ArcBand<R>,
    top: &Tensor<R>,
    weighted: &Interval<R>,
) -> ArithmeticResult<ArcBand<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    // A quantity of length dimension `power`, divided by `top` that many times.
    let unit = |a: &mut Arithmetic<'_, R>, [lower, upper]: &Bounds<R>, power: usize| {
        let mut value = a.bounds(lower, upper)?;
        for _ in 0..power {
            value = a.div_positive(&value, top)?;
        }
        ArithmeticResult::Ok([value.lower().clone(), value.upper().clone()])
    };
    let length = |a: &mut Arithmetic<'_, R>, bounds: &Bounds<R>| unit(a, bounds, 1);
    let wall = a.bounds(&band.start_perimeter[0], &band.start_perimeter[1])?;
    let perimeter = a.add(&wall, weighted)?;
    let perimeter = a.div_positive(&perimeter, top)?;
    Ok(ArcBand {
        centre_x: length(a, &band.centre_x)?,
        centre_y: length(a, &band.centre_y)?,
        radius: length(a, &band.radius)?,
        start_angle: band.start_angle.clone(),
        start_sin: band.start_sin.clone(),
        start_cos: band.start_cos.clone(),
        start_depth: length(a, &band.start_depth)?,
        start_area: unit(a, &band.start_area, 2)?,
        start_perimeter: [perimeter.lower().clone(), perimeter.upper().clone()],
        start_moment: unit(a, &band.start_moment, 3)?,
        circle_low: length(a, &band.circle_low)?,
        circle_high: length(a, &band.circle_high)?,
    })
}
