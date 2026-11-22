//! The supercritical profile S2 of each inlet-critical row, and the momentum function.

use super::super::March;
use super::super::coordinates::coordinate;
use super::super::integrand::{ProfileParams, integrand};
use super::super::jet;
use super::super::partition::{ProfileSpec, guarded};
use super::super::screen::with_peak;
use super::super::trig::fill;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, section};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorOutletContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// S2 profile inputs and the rows each later stage serves.
pub(super) struct Steep<R: Runtime<DType = DType>> {
    pub(super) spec: ProfileSpec<R>,
    /// Inlet-critical rows with an S2 profile.
    pub(super) supercritical: Tensor<R>,
    /// Rows with an S1 profile from a free outlet down to `yc`.
    pub(super) candidates: Tensor<R>,
    /// `Q^2 / g`.
    pub(super) momentum_scale: Bounds<R>,
}

pub(super) fn pair<R: Runtime<DType = DType>>(value: &Interval<R>) -> Bounds<R> {
    [value.lower().clone(), value.upper().clone()]
}

/// Describes the S2 profile from `yc` at the inlet toward `yn`, on rows with `yn < yc`.
pub(super) fn steep<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    outlet: &TensorOutletContext<R>,
    march: &March<R>,
    discharge: &Tensor<R>,
) -> ArithmeticResult<Steep<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let one = &barrel.one;
    let rise = &barrel.rise;
    let classes = &march.classes;
    let normal = &march.normal;
    let critical = c.eq(
        &march.headwater.status,
        &fill(rise, codes::OUTLET_INLET_CRITICAL)?,
    )?;
    let solved = c.eq(&normal.status, &fill(rise, codes::NORMAL_SOLVED)?)?;
    let yn = [
        c.where_cond(&solved, &normal.lower, &barrel.half_rise)?,
        c.where_cond(&solved, &normal.upper, &barrel.half_rise)?,
    ];
    let below = c.mul(&solved, &c.lt(&yn[1], &classes.critical[0])?)?;
    let wanted = c.mul(&critical, &below)?;

    let outline = outlet.outline(barrel);
    let critical_depth = a.bounds(&classes.critical[0], &classes.critical[1])?;
    let v_c = coordinate(a, &outline, rise, &critical_depth)?;
    let normal_depth = a.bounds(&yn[0], &yn[1])?;
    let v_n = coordinate(a, &outline, rise, &normal_depth)?;
    let minus = fill(rise, -1.0)?;
    let params = ProfileParams {
        numerator: minus.clone(),
        denominator: one.clone(),
        ..march.profile.spec.params.clone()
    };
    let params = with_peak(a, params, barrel)?;
    let at_start = integrand(a, &params, &v_c, 1, None)?;
    let value = jet::coefficient(a, &at_start.value, 0, rise)?;
    let start = a.point(v_c.lower())?;
    let width = a.sub(&v_c, &start)?;
    let w = a.point(width.upper())?;
    let scaled = a.mul(&w, &value)?;
    let zero = &barrel.zero;
    let error = [
        c.minimum(scaled.lower(), zero)?,
        c.maximum(scaled.upper(), zero)?,
    ];
    let room = guarded(c, &minus, v_c.lower(), v_n.upper())?;
    let active = c.mul(&c.mul(&wanted, &room)?, &at_start.valid)?;
    let harmless = outline.harmless()?;
    let keep = |value: &Tensor<R>| c.where_cond(&active, value, &harmless);
    let spec = ProfileSpec {
        params,
        active: active.clone(),
        asymptotic: active.clone(),
        sign: minus,
        start: keep(v_c.lower())?,
        start_far: keep(v_c.upper())?,
        start_error: [
            c.where_cond(&active, &error[0], zero)?,
            c.where_cond(&active, &error[1], zero)?,
        ],
        limit: keep(v_n.upper())?,
        pole_far: keep(v_n.lower())?,
        scale: c.where_cond(&active, &outlet.length, one)?,
    };
    let falling = &classes.falling;
    let free = c.sub(one, falling)?;
    let candidates = c.mul(&c.mul(&march.reach.critical, &free)?, &active)?;
    let flow = a.point(&c.where_cond(&c.eq(discharge, zero)?, one, discharge)?)?;
    let square = a.mul(&flow, &flow)?;
    let scale = a.div_positive(&square, &outlet.gravity)?;
    Ok(Steep {
        spec,
        supercritical: active,
        candidates,
        momentum_scale: pair(&scale),
    })
}

/// Encloses `M = Q^2 / (g A) + m` over a depth enclosure below the crown, from the barrel
/// geometry at both ends. `A` and `m` both rise with depth.
pub(super) fn momentum<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    scale: &Interval<R>,
    depth: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let low = c.maximum(depth.lower(), &barrel.zero)?;
    let high = c.maximum(depth.upper(), &barrel.zero)?;
    let bottom = section::at_depth(a, barrel, &low)?.geometry;
    let top = section::at_depth(a, barrel, &high)?.geometry;
    let area = a.bounds(bottom.area.lower(), top.area.upper())?;
    let moment = a.bounds(bottom.moment.lower(), top.moment.upper())?;
    let force = a.div_positive_interval(scale, &area)?;
    a.add(&force, &moment)
}

/// Row index of every lane when each row owns `per_row` consecutive lanes.
pub(super) fn owners<R>(
    device: &R::Device,
    rows: usize,
    per_row: usize,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
{
    let values: Vec<i64> = (0..rows * per_row)
        .map(|lane| i64::try_from(lane / per_row).unwrap_or(i64::MAX))
        .collect();
    Tensor::<R>::from_slice(&values, &[rows * per_row], device)
}

/// Entry of every row of `table` at the per-row column index `index`.
pub(super) fn column<R>(
    c: &R::Client,
    table: &Tensor<R>,
    index: &Tensor<R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let lanes = index.numel();
    let index = c.cast(index, DType::I64)?.reshape(&[lanes, 1])?;
    c.gather(table, 1, &index)?.reshape(&[lanes])
}
