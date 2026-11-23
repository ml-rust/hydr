//! Form-1 unsubmerged inlet inversion in the critical-depth fraction `s = dc / D`.
//!
//! The residual is `f(s) = D h(s) + D (K X(s)^M + Ks S) - HW`. `h(s) = Hc / D` is the unit
//! critical head and `X(s) = Q(s) / scale`, with `Q(s)^2 = g A^3 / T` at depth `s D`.
//! `Hc = dc + A / (2 T)` rises at least as fast as `dc`, because `A / T` never falls with depth.
//! `X` rises with `s` too. So `f(t) - f(s) >= D (t - s)`, the solver's slope floor.
//! One certified bisection in `s` replaces a critical-depth solve inside a bisection in `Q`.
//!
//! The domain is `[2^-30, top]`. A root below the floor or next to it lies in `[0, past]`,
//! where `past` follows from the slope floor.

use super::Selected;
use crate::structures::numerics::culvert::constants::CRITICAL_FLOOR;
use crate::structures::numerics::culvert::critical::residual::{residual as critical, target};
use crate::structures::numerics::culvert::inlet::head::section_unit_head;
use crate::structures::numerics::culvert::{Arithmetic, roots};
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorInletContext};
use crate::structures::{CulvertClient, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, with_interval_arithmetic};
use solvr::optimize::{CertifiedBisectionAlgorithms, CertifiedRootValues};
use std::cell::Cell;

/// Encloses `ln Q(s)` at the depth fractions `s`, each in `[2^-30, top]`.
fn log_discharge<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    inlet: &TensorInletContext<R>,
    s: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    // With a unit discharge the critical residual is `ln(g A^3 / T) = 2 ln Q(s)`.
    let unit_target = target(a, barrel, &barrel.one, &inlet.gravity)?;
    let twice = critical(a, barrel, unit, &unit_target, s)?;
    let half = a.point(&barrel.constant(0.5)?)?;
    a.mul(&half, &twice)
}

/// Encloses `f(s)` at the depth fractions `s`, each in `[2^-30, top]`, against `headwater`.
fn residual<R>(
    a: &mut Arithmetic<'_, R>,
    rows: &Selected<R>,
    headwater: &Tensor<R>,
    s: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let Selected {
        barrel,
        unit,
        inlet,
        ..
    } = rows;
    let ln_q = log_discharge(a, barrel, unit, inlet, s)?;
    let scale = a.bounds(&inlet.scale[0], &inlet.scale[1])?;
    let ln_scale = a.log_positive(&scale)?;
    let ln_x = a.sub(&ln_q, &ln_scale)?;
    let m = a.point(&inlet.m)?;
    let ln_power = a.mul(&m, &ln_x)?;
    let power = super::exp(a, &ln_power)?;
    let k = a.point(&inlet.k)?;
    let term = a.mul(&k, &power)?;
    let slope_term = a.bounds(&inlet.slope_term[0], &inlet.slope_term[1])?;
    let inner = a.add(&term, &slope_term)?;
    let head = section_unit_head(a, barrel, unit, s)?;
    let sum = a.add(&head, &inner)?;
    let d = a.point(barrel.open_rise())?;
    let total = a.mul(&d, &sum)?;
    let h = a.point(headwater)?;
    a.sub(&total, &h)
}

/// Encloses the per-barrel discharge of every `wanted` row (mask 1) from its inlet headwater.
///
/// Rows outside `wanted` and rows whose root lies above the domain get the empty pair
/// `(+inf, -inf)`. The caller excludes rows at or below the zero-discharge headwater.
pub(crate) fn invert<R>(
    rows: &Selected<R>,
    headwater: &Tensor<R>,
    wanted: &Tensor<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let floor = barrel.constant(CRITICAL_FLOOR)?;
    let top = &barrel.critical_top;
    let signs = with_interval_arithmetic(c, DType::F64, "culvert form-1 inversion ends", |a| {
        let bottom = residual(a, rows, headwater, &floor)?;
        let crown = residual(a, rows, headwater, top)?;
        roots::classify(
            a,
            barrel,
            wanted,
            [&bottom, &crown],
            top,
            barrel.open_rise(),
        )
    })?;
    let upper = c.where_cond(&signs.solved, top, &floor)?;
    let cause = Cell::new(None);
    let solved = c.bisect_certified(
        |query| {
            let outcome = query_residual(rows, headwater, &query.owner, &query.point);
            match outcome {
                Ok([lower, upper]) => Ok(CertifiedRootValues { lower, upper }),
                Err(error) => {
                    let reason = error.to_string();
                    cause.set(Some(error));
                    Err(numr::error::Error::InvalidArgument {
                        arg: "culvert form-1 inversion residual",
                        reason,
                    })
                }
            }
        },
        &floor,
        &upper,
        barrel.open_rise(),
        &super::options(barrel.width()),
    );
    let roots = match (solved, cause.take()) {
        (_, Some(error)) => return Err(error),
        (Ok(roots), None) => roots,
        (Err(error), None) => return Err(error.into()),
    };
    let found = super::outcome(c, &roots, &[floor.clone(), upper.clone()])?;
    let open = c.mul(&signs.solved, &c.sub(&barrel.one, &found.enclosed)?)?;
    if let Some(lane) = super::first(&open)? {
        return Err(StructureError::numerical(
            "culvert form-1 inlet inversion left a certified bracket open. Report this lane",
        )
        .at(lane));
    }
    let [root_low, root_high] = &found.bounds;
    let zero = &barrel.zero;
    let past = c.minimum(&signs.past, top)?;
    let low = c.where_cond(
        &signs.solved,
        root_low,
        &c.where_cond(&signs.near_top, &signs.start, zero)?,
    )?;
    let high = c.where_cond(
        &signs.solved,
        root_high,
        &c.where_cond(
            &signs.below,
            &floor,
            &c.where_cond(&signs.straddle, &past, top)?,
        )?,
    )?;
    let [lower, upper] =
        with_interval_arithmetic(c, DType::F64, "culvert form-1 discharge", |a| {
            let at = |a: &mut Arithmetic<'_, R>, s: &Tensor<R>| -> ArithmeticResult<Interval<R>> {
                let ln_q = log_discharge(a, barrel, &rows.unit, &rows.inlet, s)?;
                super::exp(a, &ln_q)
            };
            let c = a.client();
            let positive = c.gt(&low, zero)?;
            let bottom = at(a, &c.where_cond(&positive, &low, &floor)?)?;
            let crown = at(a, &high)?;
            Ok([
                c.where_cond(&positive, bottom.lower(), zero)?,
                crown.upper().clone(),
            ])
        })?;
    let kept = c.mul(wanted, &c.sub(&barrel.one, &signs.above)?)?;
    Ok([
        c.where_cond(&kept, &lower, &barrel.constant(f64::INFINITY)?)?,
        c.where_cond(&kept, &upper, &barrel.constant(f64::NEG_INFINITY)?)?,
    ])
}

/// Residual bounds at the solver's query points, on the lanes each point belongs to.
fn query_residual<R>(
    rows: &Selected<R>,
    headwater: &Tensor<R>,
    owner: &Tensor<R>,
    point: &Tensor<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let c = &rows.barrel.client;
    let picked = rows.select(owner)?;
    let h = c.index_select(headwater, 0, owner)?;
    let value =
        with_interval_arithmetic(c, DType::F64, "culvert form-1 inversion residual", |a| {
            let value = residual(a, &picked, &h, point)?;
            Ok([value.lower().clone(), value.upper().clone()])
        })?;
    Ok(value)
}
