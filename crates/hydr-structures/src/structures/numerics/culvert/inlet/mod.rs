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

pub(crate) mod anchors;
pub(crate) mod bands;
pub(crate) mod head;
mod power;

use super::{BarrelClient, RowNames, critical};
use crate::structures::StructureError;
use crate::structures::tensor::culvert::{TensorBarrelContext, TensorInletContext};
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
    let bands = bands::classify(inlet, barrel, discharge)?;
    let needs = c.mul(&inlet.form_one, &bands.unsubmerged)?;
    let flow = c.where_cond(&needs, discharge, &barrel.zero)?;
    let critical = critical::solve(names, barrel, unit, &flow, &inlet.gravity)?;
    let operands = bands::Operands {
        barrel,
        unit,
        inlet,
        discharge,
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
