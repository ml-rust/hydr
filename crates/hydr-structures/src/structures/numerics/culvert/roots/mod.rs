//! One certified bisection call over every barrel row, and the depth enclosures it yields.
//!
//! Each depth query writes an increasing log residual in `s = y/D` and classifies its rows from
//! the residual at both domain ends first. Bracketed rows get the domain `[2^-30, top]`. Every
//! other row gets `[2^-30, 2^-30]`, so the solver classifies it from one endpoint. Depth
//! enclosures are `D` times the enclosure in `s`, rounded outward and capped at `D`.

mod signs;

pub(super) use signs::{Signs, classify};

use super::constants::CRITICAL_FLOOR;
use super::{Arithmetic, BarrelClient};
use crate::structures::StructureError;
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, with_interval_arithmetic};
use solvr::optimize::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootProbe,
    CertifiedRootQuery, CertifiedRootResult, CertifiedRootStatus, CertifiedRootValues,
};
use std::cell::Cell;

/// Probe rounds allowed per solve. Bisection needs about 64 plus the floor span.
const MAX_ITERATIONS: usize = 256;

/// Default relative width `2^-50` at which a solved enclosure stops contracting.
pub(crate) const RELATIVE_TOLERANCE: f64 = 1.0 / 1_125_899_906_842_624.0;

/// Depth status codes per row with lower and upper depths in metres.
///
/// Rows without an enclosure hold NaN depths.
pub(crate) struct BarrelDepths<R: Runtime<DType = DType>> {
    pub(crate) status: Tensor<R>,
    pub(crate) lower: Tensor<R>,
    pub(crate) upper: Tensor<R>,
}

/// Per-row classification and the enclosures that need no solver call.
pub(super) struct Ends<R: Runtime<DType = DType>> {
    /// Endpoints of the row target, re-entered by the solver callback.
    pub(super) target: Bounds<R>,
    /// Depth status code per row. Bracketed rows hold the solved code.
    pub(super) status: Tensor<R>,
    /// 1 on bracketed rows, 0 elsewhere.
    pub(super) solved: Tensor<R>,
    /// 1 on rows with a depth enclosure, 0 elsewhere.
    pub(super) enclosed: Tensor<R>,
    /// Enclosure in `s` of every row the solver does not serve. Bracketed rows hold the domain.
    pub(super) fraction: Bounds<R>,
    /// Solver upper end: `top` on bracketed rows, the floor elsewhere, so other rows stop at once.
    pub(super) solver_top: Tensor<R>,
}

/// Encloses a residual at the depth fractions `fraction` of gathered `rows`.
///
/// Arguments: the interval context, the barrel rows, the unit-circle rows, the row target and
/// the fractions.
pub(super) trait Residual<R: Runtime<DType = DType>>:
    Fn(
    &mut Arithmetic<'_, R>,
    &TensorBarrelContext<R>,
    &TensorBarrelContext<R>,
    &Interval<R>,
    &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
{
}

impl<R, F> Residual<R> for F
where
    R: Runtime<DType = DType>,
    F: Fn(
        &mut Arithmetic<'_, R>,
        &TensorBarrelContext<R>,
        &TensorBarrelContext<R>,
        &Interval<R>,
        &Tensor<R>,
    ) -> ArithmeticResult<Interval<R>>,
{
}

/// How one residual solve stops and names itself.
pub(super) struct Stopping<'a, R: Runtime<DType = DType>> {
    /// Row slope floors in `s`.
    pub(super) slope: &'a Tensor<R>,
    /// Relative width that stops a row.
    pub(super) rtol: f64,
    /// Residual name used in errors.
    pub(super) stage: &'static str,
}

/// Encloses the root of every bracketed row and assembles the depths of every row.
///
/// `describe` names a bracketed row the solver left open. It runs only after a rejected call.
pub(super) fn solve<R>(
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    ends: Ends<R>,
    stopping: Stopping<'_, R>,
    residual: impl Residual<R>,
    describe: impl FnOnce(usize) -> Result<String, StructureError>,
) -> Result<BarrelDepths<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R>,
{
    let Stopping { slope, rtol, stage } = stopping;
    let c = &barrel.client;
    let floor = barrel.constant(CRITICAL_FLOOR)?;
    let options = CertifiedBisectionOptions {
        atol: 0.0,
        rtol,
        max_iterations: MAX_ITERATIONS,
        max_points: barrel.width().saturating_mul(2 * (MAX_ITERATIONS + 1)),
        probe: CertifiedRootProbe::SlackSecant,
    };
    let cause = Cell::new(None);
    let solved = c.bisect_certified(
        |query| evaluate(barrel, unit, &ends.target, query, stage, &residual, &cause),
        &floor,
        &ends.solver_top,
        slope,
        &options,
    );
    let roots = match (solved, cause.take()) {
        (_, Some(error)) => return Err(error),
        (Ok(roots), None) => roots,
        (Err(error), None) => return Err(error.into()),
    };
    let depth = with_interval_arithmetic(c, DType::F64, stage, |a| {
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
        Err(error) => {
            return Err(match unresolved(&ends.solved, &roots) {
                Ok(Some((row, status))) => match describe(row) {
                    Ok(text) => StructureError::numerical(format!(
                        "{text}: the solver returned status {status:?} on a certified bracket. Report this row"
                    ))
                    .at(row),
                    Err(read_error) => read_error,
                },
                Ok(None) => error.into(),
                Err(read_error) => read_error,
            });
        }
    };
    let missing = barrel.constant(f64::NAN)?;
    Ok(BarrelDepths {
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
    target: &Bounds<R>,
    query: &CertifiedRootQuery<R>,
    stage: &'static str,
    residual: &impl Residual<R>,
    cause: &Cell<Option<StructureError>>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let outcome = with_interval_arithmetic(&barrel.client, DType::F64, stage, |a| {
        let c = a.client();
        let rows = barrel.select(&query.owner)?;
        let unit_rows = unit.select(&query.owner)?;
        let lower = c.index_select(&target[0], 0, &query.owner)?;
        let upper = c.index_select(&target[1], 0, &query.owner)?;
        let row_target = a.bounds(&lower, &upper)?;
        let value = residual(a, &rows, &unit_rows, &row_target, &query.point)?;
        Ok([value.lower().clone(), value.upper().clone()])
    });
    match outcome {
        Ok([lower, upper]) => Ok(CertifiedRootValues { lower, upper }),
        Err(error) => {
            let reason = error.to_string();
            cause.set(Some(error.into()));
            Err(numr::error::Error::InvalidArgument { arg: stage, reason })
        }
    }
}

/// Writes the solver's compact root enclosures over the per-row enclosures in `s`.
fn scatter<R>(
    barrel: &TensorBarrelContext<R>,
    roots: &CertifiedRootResult<R>,
    fraction: &Bounds<R>,
) -> numr::error::Result<Bounds<R>>
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

/// The first bracketed row the solver did not resolve, with its status. Reads to the host.
fn unresolved<R>(
    solved: &Tensor<R>,
    roots: &CertifiedRootResult<R>,
) -> Result<Option<(usize, Option<CertifiedRootStatus>)>, StructureError>
where
    R: Runtime<DType = DType>,
{
    let solved = solved.try_to_vec::<f64>()?;
    let status = roots.status().try_to_vec::<i64>()?;
    let accepted = |code: i64| {
        matches!(
            CertifiedRootStatus::from_code(code),
            Some(CertifiedRootStatus::Converged | CertifiedRootStatus::AccuracyFloor)
        )
    };
    Ok(solved
        .iter()
        .zip(&status)
        .position(|(&bracketed, &code)| bracketed == 1.0 && !accepted(code))
        .map(|row| (row, CertifiedRootStatus::from_code(status[row]))))
}
