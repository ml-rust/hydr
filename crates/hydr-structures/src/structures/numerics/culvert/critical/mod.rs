//! Certified critical depth `Q^2 T = g A^3` for circle, box and arc barrels.
//!
//! Hydr writes the log residual in `s = y/D` and solvr's certified bisection finds its root on
//! `[2^-30, top]`: `top = 1 - 2^-53` for a circle and an arc barrel, `1` for a box. The slope
//! floors are `5.4` for a circle, `3` for a box and the row's certified floor for an arc
//! barrel, in units of `1/D`. End classification runs first, so only bracketed rows iterate.
//! Depth enclosures are `D` times the enclosure in `s`, rounded outward.

mod ends;
pub(crate) mod residual;

use super::constants::CRITICAL_FLOOR;
use super::roots::{self, BarrelDepths};
use super::{BarrelClient, RowNames};
use crate::structures::StructureError;
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::optimize::CertifiedBisectionAlgorithms;

/// Encloses the critical depth of every row for discharges `discharge` and gravities `gravity`.
///
/// Both are `[m]` tensors. The caller checks that every discharge is finite and nonnegative and
/// every gravity finite and positive. `unit` is [`TensorBarrelContext::unit`] of `barrel`.
/// `names` names the rows of `barrel` in an error.
pub(crate) fn solve<R>(
    names: RowNames<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    discharge: &Tensor<R>,
    gravity: &Tensor<R>,
) -> Result<BarrelDepths<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R>,
{
    solve_to(
        names,
        barrel,
        unit,
        discharge,
        gravity,
        roots::RELATIVE_TOLERANCE,
    )
}

/// [`solve`] with the relative enclosure width `rtol` in `s`.
pub(crate) fn solve_to<R>(
    names: RowNames<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    discharge: &Tensor<R>,
    gravity: &Tensor<R>,
    rtol: f64,
) -> Result<BarrelDepths<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R>,
{
    let c = &barrel.client;
    let ends = with_interval_arithmetic(c, DType::F64, "culvert critical depth ends", |a| {
        ends::classify(a, barrel, unit, discharge, gravity)
    })?;
    roots::solve(
        barrel,
        unit,
        ends,
        roots::Stopping {
            slope: &barrel.critical_slope,
            rtol,
            stage: "culvert critical residual",
        },
        residual::residual,
        |row| describe(names, [discharge, &barrel.critical_top], row),
    )
}

/// Names a bracketed row the solver left open. Reads the discharges and domain tops to the host.
fn describe<R>(
    names: RowNames<'_, R>,
    [discharge, tops]: [&Tensor<R>; 2],
    row: usize,
) -> Result<String, StructureError>
where
    R: Runtime<DType = DType>,
{
    let discharge = discharge.try_to_vec::<f64>()?;
    let tops = tops.try_to_vec::<f64>()?;
    let (Some(entry), Some(flow), Some(fraction)) =
        (names.barrel(row)?, discharge.get(row), tops.get(row))
    else {
        return Err(StructureError::numerical(format!(
            "critical depth row={row}: the solver named a row past the barrel list. Report this runtime"
        )));
    };
    let rise = entry.rise_m();
    let top = rise * fraction;
    Ok(format!(
        "critical depth shape={:?},rise_m={rise:?},span_m={:?},discharge_m3_per_s={flow:?},bracket_m=[{:?}, {top:?}]",
        entry.shape(),
        entry.span_m(),
        rise * CRITICAL_FLOOR,
    ))
}
