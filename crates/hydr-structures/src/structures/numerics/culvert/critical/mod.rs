//! Certified critical depth `Q^2 T = g A^3` for circle and box barrels.
//!
//! Hydr writes the log residual in `s = y/D` and solvr's certified bisection finds its root on
//! `[2^-30, top]`: `top = 1 - 2^-53` for a circle and `1` for a box. The slope floors are
//! `5.4` for a circle and `3` for a box, in units of `1/D`. End classification runs first, so
//! only bracketed rows iterate. Depth enclosures are `D` times the enclosure in `s`, rounded
//! outward.

mod ends;
mod residual;

use residual::residual;

use super::BarrelClient;
use super::constants::{CIRCLE_CRITICAL_TOP, CRITICAL_FLOOR};
use crate::structures::tensor::culvert::TensorBarrelContext;
use crate::structures::{CulvertBarrel, CulvertBarrelShape, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};
use solvr::optimize::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootProbe,
    CertifiedRootQuery, CertifiedRootResult, CertifiedRootStatus, CertifiedRootValues,
};
use std::cell::Cell;

/// Probe rounds allowed per solve. Bisection needs about 64 plus the floor span.
const MAX_ITERATIONS: usize = 256;

/// Relative width at which a solved enclosure stops contracting.
const RELATIVE_TOLERANCE: f64 = 1.0 / 1_125_899_906_842_624.0;

/// Critical status codes per row with lower and upper depths in metres.
///
/// Rows without an enclosure hold NaN depths.
pub(crate) struct CriticalDepths<R: Runtime<DType = DType>> {
    pub(crate) status: Tensor<R>,
    pub(crate) lower: Tensor<R>,
    pub(crate) upper: Tensor<R>,
}

/// Encloses the critical depth of every row for discharges `discharge` and gravities `gravity`.
///
/// Both are `[m]` tensors. The caller checks that every discharge is finite and nonnegative and
/// every gravity finite and positive. `unit` is [`TensorBarrelContext::unit_circles`] of `barrel`.
/// `barrels` is the host list of `barrel`, read only to name a row in an error.
pub(crate) fn solve<R>(
    barrels: &[CulvertBarrel],
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    discharge: &Tensor<R>,
    gravity: &Tensor<R>,
) -> Result<CriticalDepths<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R>,
{
    let c = &barrel.client;
    let ends = with_interval_arithmetic(c, DType::F64, "culvert critical depth ends", |a| {
        ends::classify(a, barrel, unit, discharge, gravity)
    })?;
    let floor = barrel.constant(CRITICAL_FLOOR)?;
    let options = CertifiedBisectionOptions {
        atol: 0.0,
        rtol: RELATIVE_TOLERANCE,
        max_iterations: MAX_ITERATIONS,
        max_points: barrel.width().saturating_mul(2 * (MAX_ITERATIONS + 1)),
        probe: CertifiedRootProbe::SlackSecant,
    };
    let cause = Cell::new(None);
    let solved = c.bisect_certified(
        |query| evaluate(barrel, unit, &ends.target, query, &cause),
        &floor,
        &ends.solver_top,
        &barrel.critical_slope,
        &options,
    );
    let roots = match (solved, cause.take()) {
        (_, Some(error)) => return Err(error),
        (Ok(roots), None) => roots,
        (Err(error), None) => return Err(error.into()),
    };
    let depth = with_interval_arithmetic(c, DType::F64, "culvert critical depth", |a| {
        let c = a.client();
        let [lower, upper] = scatter(barrel, &roots, &ends.fraction)?;
        let status = c.cast(roots.status(), DType::F64)?;
        let code = |value: CertifiedRootStatus| barrel.constant(value as i64 as f64);
        let converged = c.eq(&status, &code(CertifiedRootStatus::Converged)?)?;
        let floored = c.eq(&status, &code(CertifiedRootStatus::AccuracyFloor)?)?;
        let accepted = c.maximum(&converged, &floored)?;
        let failed = c.mul(&ends.solved, &c.sub(&barrel.one, &accepted)?)?;
        // Inverted bounds on a bracketed row the solver left open fail the completion check.
        let lower = c.where_cond(&failed, &barrel.one, &lower)?;
        let upper = c.where_cond(&failed, &barrel.zero, &upper)?;
        let fraction = a.bounds(&lower, &upper)?;
        let rise = a.point(&barrel.rise)?;
        let depth = a.mul(&rise, &fraction)?;
        // Every enclosure lies in s <= 1, so the exact crown D caps the rounded product.
        Ok([
            depth.lower().clone(),
            c.minimum(depth.upper(), &barrel.rise)?,
        ])
    });
    let [lower, upper] = match depth {
        Ok(depth) => depth,
        Err(error) => return Err(unresolved(barrels, discharge, &ends.solved, &roots, error)),
    };
    let missing = barrel.constant(f64::NAN)?;
    Ok(CriticalDepths {
        lower: c.where_cond(&ends.enclosed, &lower, &missing)?,
        upper: c.where_cond(&ends.enclosed, &upper, &missing)?,
        status: ends.status,
    })
}

/// Residual bounds at the solver's query points, on the rows each point belongs to.
///
/// A round can query one row at two points, so the rows are gathered, never scattered.
fn evaluate<R>(
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    target: &[Tensor<R>; 2],
    query: &CertifiedRootQuery<R>,
    cause: &Cell<Option<StructureError>>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let outcome = with_interval_arithmetic(
        &barrel.client,
        DType::F64,
        "culvert critical residual",
        |a| {
            let c = a.client();
            let rows = barrel.select(&query.owner)?;
            let unit_rows = unit.select(&query.owner)?;
            let lower = c.index_select(&target[0], 0, &query.owner)?;
            let upper = c.index_select(&target[1], 0, &query.owner)?;
            let row_target = a.bounds(&lower, &upper)?;
            let value = residual(a, &rows, &unit_rows, &row_target, &query.point)?;
            Ok([value.lower().clone(), value.upper().clone()])
        },
    );
    match outcome {
        Ok([lower, upper]) => Ok(CertifiedRootValues { lower, upper }),
        Err(error) => {
            let reason = error.to_string();
            cause.set(Some(error.into()));
            Err(numr::error::Error::InvalidArgument {
                arg: "culvert critical residual",
                reason,
            })
        }
    }
}

/// Writes the solver's compact root enclosures over the per-row enclosures in `s`.
fn scatter<R>(
    barrel: &TensorBarrelContext<R>,
    roots: &CertifiedRootResult<R>,
    fraction: &[Tensor<R>; 2],
) -> numr::error::Result<[Tensor<R>; 2]>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let [lower, upper] = fraction;
    let owner = roots.root_owner();
    if owner.numel() == 0 {
        return Ok([lower.clone(), upper.clone()]);
    }
    let c = &barrel.client;
    Ok([
        c.index_put(lower, 0, owner, roots.root_lower())?,
        c.index_put(upper, 0, owner, roots.root_upper())?,
    ])
}

/// Names the first bracketed row the solver did not resolve. Runs only after a rejected call.
fn unresolved<R>(
    barrels: &[CulvertBarrel],
    discharge: &Tensor<R>,
    solved: &Tensor<R>,
    roots: &CertifiedRootResult<R>,
    error: ArithmeticError,
) -> StructureError
where
    R: Runtime<DType = DType>,
{
    let read = || -> Result<_, StructureError> {
        Ok((
            solved.try_to_vec::<f64>()?,
            roots.status().try_to_vec::<i64>()?,
            discharge.try_to_vec::<f64>()?,
        ))
    };
    let (solved, status, discharge) = match read() {
        Ok(values) => values,
        Err(read_error) => return read_error,
    };
    let accepted = |code: i64| {
        matches!(
            CertifiedRootStatus::from_code(code),
            Some(CertifiedRootStatus::Converged | CertifiedRootStatus::AccuracyFloor)
        )
    };
    let failed = solved
        .iter()
        .zip(&status)
        .position(|(&bracketed, &code)| bracketed == 1.0 && !accepted(code));
    let Some((row, entry)) = failed.and_then(|row| Some((row, barrels.get(row)?))) else {
        return error.into();
    };
    let rise = entry.rise_m();
    let top = match entry.shape() {
        CulvertBarrelShape::Circular => rise * CIRCLE_CRITICAL_TOP,
        _ => rise,
    };
    StructureError::numerical(format!(
        "critical depth shape={:?},rise_m={rise:?},span_m={:?},discharge_m3_per_s={:?},bracket_m=[{:?}, {top:?}]: the solver returned status {:?} on a certified bracket. Report this row",
        entry.shape(),
        entry.span_m(),
        discharge[row],
        rise * CRITICAL_FLOOR,
        CertifiedRootStatus::from_code(status[row]),
    ))
    .at(row)
}
