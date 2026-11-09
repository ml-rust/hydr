//! Complete pinned native Modified Green–Ampt interval branches.

use super::integration::{integral, native_min};
use crate::cpu::CpuTensor;
use crate::cpu::soil_storage::{CpuSoilContext, CpuSoilState};
use crate::soil::SoilError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps};

pub(crate) fn advance(
    context: &CpuSoilContext,
    old: &CpuSoilState,
    dt: f64,
    rain: &CpuTensor,
    depth: &CpuTensor,
) -> Result<(CpuSoilState, CpuTensor), SoilError> {
    let c = &context.client;
    let zero = &context.zero;
    let one = &context.one;
    let duration = c.mul_scalar(one, dt)?;
    let raw_ia = c.add(rain, &c.div_scalar(depth, dt)?)?;
    context.finite(&raw_ia, "native effective availability (ft/s)")?;
    let ia = c.where_cond(&c.lt(&raw_ia, &c.mul_scalar(one, 1e-10)?)?, zero, &raw_ia)?;
    let unsaturated = c.sub(one, &old.sat)?;
    let dry = c.mul(&unsaturated, &c.eq(&ia, zero)?)?;
    let wet = c.gt(&ia, zero)?;
    let low = c.mul(&unsaturated, &c.mul(&wet, &c.le(&ia, &context.ks)?)?)?;
    let high = c.mul(&unsaturated, &c.gt(&ia, &context.ks)?)?;
    let saturation_wet = c.mul(&old.sat, &wet)?;
    let mut next = old.clone();
    next.t = c.sub_scalar(&old.t, dt)?;

    // Unsaturated effective-zero recovery, including tiny positive ponding.
    let recovering = c.mul(&dry, &c.gt(&old.fu, zero)?)?;
    let recovered_amount = c.mul_scalar(
        &c.mul(&c.div_scalar(&context.lu, 90000.0)?, &context.capacity)?,
        dt,
    )?;
    let recovered_f = c.sub(&old.f, &recovered_amount)?;
    let recovered_fu = c.sub(&old.fu, &recovered_amount)?;
    let empty = c.mul(&recovering, &c.le(&recovered_fu, zero)?)?;
    let expired = c.mul(
        &recovering,
        &c.mul(&c.gt(&recovered_fu, zero)?, &c.le(&next.t, zero)?)?,
    )?;
    next.f = c.where_cond(&recovering, &recovered_f, &next.f)?;
    next.fu = c.where_cond(&recovering, &recovered_fu, &next.fu)?;
    next.f = c.where_cond(&c.maximum(&empty, &expired)?, zero, &next.f)?;
    next.fu = c.where_cond(&empty, zero, &next.fu)?;
    let recovered_imd = c.div(&c.sub(&context.capacity, &recovered_fu)?, &context.lu)?;
    next.imd = c.where_cond(&expired, &recovered_imd, &next.imd)?;
    next.imd = c.where_cond(&empty, &context.imdmax, &next.imd)?;

    // Native Fs multiplication order differs from ks*c1.
    let sum = c.add(&context.suction, depth)?;
    let high_sum = c.where_cond(&high, &sum, zero)?;
    let denominator = c.where_cond(&high, &c.sub(&ia, &context.ks)?, one)?;
    let fs = c.div(
        &c.mul(&c.mul(&context.ks, &high_sum)?, &old.imd)?,
        &denominator,
    )?;
    let already = c.mul(&high, &c.gt(&old.f, &fs)?)?;
    let supply = c.mul_scalar(&ia, dt)?;
    let remaining = c.mul(
        &high,
        &c.mul(
            &c.sub(one, &already)?,
            &c.lt(&c.add(&old.f, &supply)?, &fs)?,
        )?,
    )?;
    let crossing = c.mul(&high, &c.sub(one, &c.maximum(&already, &remaining)?)?)?;
    let saturated = c.maximum(&saturation_wet, &already)?;
    let integrating = c.maximum(&crossing, &saturated)?;
    let crossing_ia = c.where_cond(&crossing, &ia, one)?;
    let ts_raw = c.sub(&duration, &c.div(&c.sub(&fs, &old.f)?, &crossing_ia)?)?;
    let ts = c.where_cond(&c.le(&ts_raw, zero)?, zero, &ts_raw)?;
    let ts = c.where_cond(&crossing, &ts, &duration)?;
    let f1 = c.where_cond(&crossing, &fs, &old.f)?;
    let safe_sum = c.where_cond(&integrating, &sum, zero)?;
    let coefficient = c.mul(&safe_sum, &old.imd)?;
    let f2 = integral(context, &f1, &coefficient, &ts, &integrating)?;

    let crossing_limit = c.add(&fs, &c.mul(&ia, &ts)?)?;
    let crossing_f2 = c.where_cond(&c.gt(&f2, &crossing_limit)?, &crossing_limit, &f2)?;
    let crossing_df = c.sub(&crossing_f2, &old.f)?;
    let saturated_df = c.sub(&f2, &old.f)?;
    let supply_limited = c.mul(&saturated, &c.gt(&saturated_df, &supply)?)?;
    let saturated_df = c.where_cond(&supply_limited, &supply, &saturated_df)?;
    let simple = c.maximum(&low, &remaining)?;
    let mut rate = c.where_cond(&simple, &ia, zero)?;
    rate = c.where_cond(&crossing, &c.div_scalar(&crossing_df, dt)?, &rate)?;
    rate = c.where_cond(&saturated, &c.div_scalar(&saturated_df, dt)?, &rate)?;
    let df = c.where_cond(&crossing, &crossing_df, &saturated_df)?;
    let df = c.where_cond(&simple, &supply, &df)?;
    let updating = c.maximum(&simple, &integrating)?;
    let updated_f = c.add(&old.f, &df)?;
    next.f = c.where_cond(&updating, &updated_f, &next.f)?;
    // The unsaturated crossing branch assigns F2 directly.
    next.f = c.where_cond(&crossing, &crossing_f2, &next.f)?;
    let updated_fu = native_min(context, &c.add(&old.fu, &df)?, &context.capacity)?;
    next.fu = c.where_cond(&updating, &updated_fu, &next.fu)?;
    next.sat = c.where_cond(&integrating, one, &next.sat)?;
    next.sat = c.where_cond(&supply_limited, zero, &next.sat)?;
    let renewed = c.maximum(&high, &saturation_wet)?;
    next.t = c.where_cond(
        &renewed,
        &c.div(&c.mul_scalar(one, 5400.0)?, &context.lu)?,
        &next.t,
    )?;
    next.check(context)?;
    context.finite(&rate, "native infiltration rate (ft/s)")?;
    context.reject_mask(
        &c.lt(&rate, zero)?,
        "native infiltration rate (ft/s)",
        &[
            ("value", &rate),
            ("incoming rate (ft/s)", rain),
            ("ponding (ft)", depth),
        ],
        &[("dt (s)", dt)],
        "Native interval rate is negative. Inspect state and input scales",
    )?;
    Ok((next, rate))
}
