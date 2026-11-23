//! Certified HDS-5 inlet-control headwater `HW_i(Q)` for barrels of every shape.
//!
//! `X = Ku Q / (A D^0.5)`, with the exact SI conversion `Ku = 0.3048^(-1/2)`. `X` then equals
//! its value in foot units, so HDS-5's US limits and coefficients apply unchanged.
//!
//! | Band | Headwater above the inlet invert |
//! | --- | --- |
//! | `X <= 3.5`, form 1 | `Hc + D (K X^M + Ks S)` |
//! | `X <= 3.5`, form 2 | `D K X^M` |
//! | `3.5 < X < 4` | Linear in `X` from the unsubmerged headwater at 3.5 to the submerged one at 4 |
//! | `X >= 4` | `D (c X^2 + Y + Ks S)` |
//!
//! The transition is continuous at both limits by construction, and admission certifies that it
//! rises. Every public query runs one discharge check, one critical-depth solve over the form-1
//! rows that may be unsubmerged, and one interval call.
//!
//! Above a floor the open section takes the coefficients: `D'` for `D`, the open full area for
//! `A`, and the critical head of the open section. The headwater runs up from the bed.
//!
//! A face of blocked share `f` passes `(1 - f)` times the unblocked discharge at each
//! headwater, so `HW_f(Q) = HW_i(Q / (1 - f))`. `HW_i` rises with `Q`, so a query encloses
//! `Q / (1 - f)` and evaluates the bands over that enclosure, with critical depth solved at both
//! of its ends. A batch without a face blockage runs at `Q` with one solve.

pub(crate) mod anchors;
pub(crate) mod bands;
pub(crate) mod head;
mod power;

use super::{BarrelClient, BarrelDepths, RowNames, critical};
use crate::structures::StructureError;
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorInletContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::optimize::CertifiedBisectionAlgorithms;

/// Inlet headwater per row: status codes with lower and upper headwater in metres.
pub(crate) struct InletHeadwater<R: Runtime<DType = DType>> {
    pub(crate) status: Tensor<R>,
    pub(crate) lower: Tensor<R>,
    pub(crate) upper: Tensor<R>,
}

/// Encloses the inlet-control headwater of every row at the discharges `discharge` (`[m]`).
///
/// The caller checks that every discharge is finite and nonnegative. `names` names the rows of
/// `barrel` in an error.
pub(crate) fn evaluate<R>(
    names: RowNames<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    inlet: &TensorInletContext<R>,
    discharge: &Tensor<R>,
) -> Result<InletHeadwater<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R>,
{
    let c = &barrel.client;
    let flows = unblocked(barrel, inlet, discharge)?;
    let bands = bands::classify(inlet, barrel, &flows)?;
    let needs = c.mul(&inlet.form_one, &bands.unsubmerged)?;
    let solve = |end: &Tensor<R>| {
        let flow = c.where_cond(&needs, end, &barrel.zero)?;
        critical::solve(names, barrel, unit, &flow, &inlet.gravity)
    };
    let [low, high] = &flows;
    let critical = if inlet.faced {
        // Critical depth rises with discharge, so the two ends enclose it.
        let below = solve(low)?;
        let above = solve(high)?;
        BarrelDepths {
            status: c.maximum(&below.status, &above.status)?,
            lower: below.lower,
            upper: above.upper,
        }
    } else {
        solve(low)?
    };
    let operands = bands::Operands {
        barrel,
        unit,
        inlet,
        discharge: &flows,
        bands: &bands,
        critical: &critical,
    };
    let headwater = with_interval_arithmetic(c, DType::F64, "culvert inlet headwater", |a| {
        bands::headwater(a, &operands)
    })?;
    Ok(InletHeadwater {
        status: bands::status(barrel, &bands)?,
        lower: headwater.lower().clone(),
        upper: headwater.upper().clone(),
    })
}

/// Encloses the unblocked discharge `Q / (1 - f)` that gives the headwater of `Q` through a
/// face of blocked share `f`. Both ends are `Q` itself when no row has a face blockage, and on
/// each row with `f = 0`.
fn unblocked<R>(
    barrel: &TensorBarrelContext<R>,
    inlet: &TensorInletContext<R>,
    discharge: &Tensor<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    if !inlet.faced {
        return Ok([discharge.clone(), discharge.clone()]);
    }
    let c = &barrel.client;
    let flows = with_interval_arithmetic(c, DType::F64, "culvert inlet face blockage", |a| {
        let q = a.point(discharge)?;
        let one = a.point(&barrel.one)?;
        let blocked = a.point(&inlet.face)?;
        let open = a.sub(&one, &blocked)?;
        let flow = a.div_positive_interval(&q, &open)?;
        Ok([flow.lower().clone(), flow.upper().clone()])
    })?;
    Ok(flows)
}
