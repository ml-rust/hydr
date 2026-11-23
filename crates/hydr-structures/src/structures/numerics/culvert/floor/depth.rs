//! Open-section terms of the critical and normal depth residuals in `s = y/D'`.
//!
//! A floored circle or arc row reads its open section in its frame, the barrel scaled by
//! `2^-e` to a rise in `[1, 2)`. The frame depth of `s` is `D'_f s`, enclosed by its outward
//! rounding, so range geometry over that enclosure holds the exact point. A metre area is
//! `2^2e` times the frame area and a metre perimeter or width `2^e` times the frame one. The
//! residuals add `e ln 2` terms to their targets instead, so no metre product underflows.
//!
//! A floored box takes the box closed forms on `D'` with the bed term `rho^(3/2) B` in `P_n`.

use super::over_depth;
use super::point::FloorQuantities;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{Bounds, FloorRows, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Frame geometry at one depth fraction per row, with the rows it serves.
pub(crate) struct OpenDepth<R: Runtime<DType = DType>> {
    /// 1 on floored circle and arc rows.
    pub(crate) rows: Tensor<R>,
    /// Frame geometry at `s` on those rows and at `s = 1/2` elsewhere.
    pub(crate) geometry: FloorQuantities<R>,
}

/// Encloses the frame geometry of the open sections of `barrel` at the fractions `fraction`.
///
/// Every fraction of a floored circle or arc row must lie in `[2^-30, 1 - 2^-52]`. Other rows
/// read the harmless fraction `1/2`.
pub(crate) fn at_fraction<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    floor: &FloorRows<R>,
    fraction: &Tensor<R>,
) -> ArithmeticResult<OpenDepth<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let rows = c.sub(&floor.floored, &floor.boxed)?;
    let framed = barrel.frame.as_ref().map_or(barrel, |frame| &frame.context);
    let Some(open) = framed.floor.as_deref() else {
        return Err(numr::error::Error::InvalidArgument {
            arg: "barrel",
            reason: "the barrel frame lost its floor rows".into(),
        }
        .into());
    };
    let operand = c.where_cond(&rows, fraction, &barrel.constant(0.5)?)?;
    let [lower, upper] = frame_depths(a, open, &operand, &operand)?;
    let geometry = over_depth(a, framed, open, &lower, &upper)?.geometry;
    Ok(OpenDepth { rows, geometry })
}

/// Frame depths `[D'_f lower, D'_f upper]` of the fractions `[lower, upper]` of the open
/// sections `open`, rounded outward.
///
/// Every fraction must lie in `[0, 1 - 2^-52]`. The exact product `D'_f (1 - 2^-52)` lies at or
/// below the F64 below `D'_f`, because `D'_f 2^-52` is at least one F64 step of `D'_f`. So that
/// F64 caps the upper depth, which keeps the range below the crown.
pub(crate) fn frame_depths<R>(
    a: &mut Arithmetic<'_, R>,
    open: &FloorRows<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<Bounds<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let top = a.point(&open.top)?;
    let low = a.point(lower)?;
    let high = a.point(upper)?;
    let from = a.mul(&low, &top)?;
    let to = a.mul(&high, &top)?;
    let below_crown = c.next_down(&open.top)?;
    Ok([from.lower().clone(), c.minimum(to.upper(), &below_crown)?])
}

/// `ln 2^e` per row, the log of the frame scale. Exactly zero on rows of scale 1, so a row
/// answers alike with or without a frame beside it.
pub(crate) fn log_scale<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let Some(frame) = &barrel.frame else {
        return a.point(&barrel.zero);
    };
    let c = a.client();
    let scale = a.point(&frame.scale)?;
    let log = a.log_positive(&scale)?;
    let unit = c.eq(&frame.scale, &barrel.one)?;
    let lower = c.where_cond(&unit, &barrel.zero, log.lower())?;
    let upper = c.where_cond(&unit, &barrel.zero, log.upper())?;
    a.bounds(&lower, &upper)
}
