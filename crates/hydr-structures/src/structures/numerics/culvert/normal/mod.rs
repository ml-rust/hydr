//! Certified Manning normal depth `Q = (1/n) A R^(2/3) sqrt(S)` for circle, box and arc
//! barrels.
//!
//! Hydr writes the log residual in `s = y/D` and solvr's certified bisection finds its root on
//! `[2^-30, top]`: `top = s_N = 0.93818` for a circle, `1` for a box and the row's certified
//! rising-limb top for an arc barrel. The slope floors are `5.42e-5` for a circle, `3` for a
//! box and the row's certified floor for an arc barrel, in units of `1/D`. End classification
//! runs first, so only bracketed rows iterate. Depth enclosures are `D` times the enclosure in
//! `s`, rounded outward.
//!
//! - Circle and arc conveyance rises to a peak at `s_K`, just above `s_N`, and falls to the
//!   crown. The normal depth is the root on the rising limb. A target above the peak has no
//!   free-surface uniform flow.
//! - Box conveyance rises up to the crown, where its open-section limit `K(D^-)` exceeds the
//!   full-barrel conveyance. The domain ends at that limit. A target above it has no
//!   free-surface uniform flow. Full-barrel friction above the crown is the caller's branch.
//!
//! Above a floor the domain is `s = y/D'` from the bed and the conveyance is
//! `A^(5/3) / P_n^(2/3)` with the wall `n`. A floored circle or arc barrel takes its certified
//! rising-limb top, slope floor and peak. A floored box keeps the box domain and floor.

mod ends;
mod residual;

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

/// Encloses the rising-limb normal depth of every row.
///
/// `discharge`, `roughness` (Manning `n`) and `slope` are `[m]` tensors. The caller checks that
/// every discharge is finite and nonnegative, every roughness finite and positive and every
/// slope finite. `unit` is [`TensorBarrelContext::unit`] of `barrel`.
/// `names` names the rows of `barrel` in an error.
pub(crate) fn solve<R>(
    names: RowNames<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    discharge: &Tensor<R>,
    roughness: &Tensor<R>,
    slope: &Tensor<R>,
) -> Result<BarrelDepths<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R>,
{
    solve_to(
        names,
        barrel,
        unit,
        [discharge, roughness, slope],
        roots::RELATIVE_TOLERANCE,
    )
}

/// [`solve`] on `[discharge, roughness, slope]` with the relative enclosure width `rtol` in `s`.
pub(crate) fn solve_to<R>(
    names: RowNames<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    [discharge, roughness, slope]: [&Tensor<R>; 3],
    rtol: f64,
) -> Result<BarrelDepths<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R>,
{
    let c = &barrel.client;
    let (ends, floors) =
        with_interval_arithmetic(c, DType::F64, "culvert normal depth ends", |a| {
            ends::classify(a, barrel, unit, discharge, roughness, slope)
        })?;
    roots::solve(
        barrel,
        unit,
        ends,
        roots::Stopping {
            slope: &floors,
            rtol,
            stage: "culvert normal residual",
        },
        residual::residual,
        |row| {
            describe(
                names,
                [discharge, roughness, slope, &barrel.normal_top],
                row,
            )
        },
    )
}

/// Names a bracketed row the solver left open. Reads the row inputs and domain tops to the host.
fn describe<R>(
    names: RowNames<'_, R>,
    inputs: [&Tensor<R>; 4],
    row: usize,
) -> Result<String, StructureError>
where
    R: Runtime<DType = DType>,
{
    let mut values = [f64::NAN; 4];
    for (value, input) in values.iter_mut().zip(inputs) {
        if let Some(read) = input.try_to_vec::<f64>()?.get(row) {
            *value = *read;
        }
    }
    let [discharge, roughness, slope, fraction] = values;
    let Some(entry) = names.barrel(row)? else {
        return Err(StructureError::numerical(format!(
            "normal depth row={row}: the solver named a row past the barrel list. Report this runtime"
        )));
    };
    let rise = entry.open_rise_m();
    let top = rise * fraction;
    Ok(format!(
        "normal depth shape={:?},rise_m={:?},open_rise_m={rise:?},span_m={:?},discharge_m3_per_s={discharge:?},manning_n={roughness:?},slope={slope:?},bracket_m=[{:?}, {top:?}]",
        entry.shape(),
        entry.rise_m(),
        entry.span_m(),
        rise * CRITICAL_FLOOR,
    ))
}
