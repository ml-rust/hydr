//! Culvert discharge as the inverse of the larger control headwater.
//!
//! `HW(Q) = max(HW_i(Q), HW_o(Q, TW))` and both controls rise with `Q`, so `HW(Q) <= H` holds
//! exactly when `Q <= Q_i(H)` and `Q <= Q_o(H, TW)`. The discharge is `min(Q_i, Q_o)`:
//!
//! - [`inlet`] inverts the inlet curve in closed form per band. Form-1 unsubmerged rows solve
//!   one certified bisection in the critical-depth fraction ([`form_one`]).
//! - [`outlet`] inverts the outlet curve at fixed tailwater on `[0, 2 Q_i]`. A drowned full
//!   barrel takes its closed form. Every other row runs one certified root search over the
//!   outlet march ([`outlet_root`]), and a row whose outlet headwater at `Q_i` stays below `H`
//!   is inlet-governed. An input hint starts each search near an earlier root ([`warm`]).
//!
//! The inversion runs in `Q` at fixed tailwater. `HW_o` is not monotone in tailwater, which
//! the inversion never needs.

mod assemble;
mod form_one;
mod inlet;
mod lanes;
mod outlet;
mod outlet_root;
mod sensitivity;
mod settle;
mod warm;

use crate::structures::numerics::culvert::Arithmetic;
use crate::structures::tensor::culvert::{
    Bounds, CulvertLaw, TensorBarrelContext, TensorCulvertLaw, TensorInletContext,
    TensorOutletContext,
};
use crate::structures::tensor::{
    TensorStructureContext, TensorStructureInput, TensorStructureResult,
};
use crate::structures::{CulvertClient, StructureError};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, CorrectlyRoundedOps, IndexingOps, NextRepresentableOps,
    TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};
use solvr::optimize::{
    CertifiedBisectionOptions, CertifiedRootProbe, CertifiedRootResult, CertifiedRootStatus,
};

/// Probe rounds allowed per solve.
const MAX_ITERATIONS: usize = 256;

/// Relative width at which a solved enclosure stops contracting.
const RELATIVE_TOLERANCE: f64 = 1.0 / 1_125_899_906_842_624.0;

impl<R> CulvertLaw<R> for TensorCulvertLaw<R>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    fn evaluate(
        &self,
        context: &TensorStructureContext<R>,
        input: &TensorStructureInput<R>,
        result: TensorStructureResult<R>,
    ) -> Result<TensorStructureResult<R>, StructureError> {
        let lanes = lanes::classify(self, context, input, &self.lane)?;
        let selected = Selected::new(self, &lanes.rows)?;
        let inlet = inlet::invert(self, &selected, &lanes)?;
        let hint = input.culvert_hint.as_ref();
        let outlet = outlet::invert(self, context, &selected, &lanes, &inlet, hint)?;
        let mut result =
            assemble::assemble(self, context, &selected, &lanes, &inlet, &outlet, result)?;
        if let (Some(older), Some(culvert)) = (hint, result.culvert.as_mut()) {
            culvert.hint = culvert.hint.clone().over(older, &self.barrel.client)?;
        }
        Ok(result)
    }
}

/// Barrel, unit-circle, inlet and outlet rows gathered at one row index per lane.
pub(crate) struct Selected<R: Runtime<DType = DType>> {
    pub(crate) barrel: TensorBarrelContext<R>,
    pub(crate) unit: TensorBarrelContext<R>,
    pub(crate) inlet: TensorInletContext<R>,
    pub(crate) outlet: TensorOutletContext<R>,
}

impl<R: Runtime<DType = DType>> Selected<R> {
    fn new(law: &TensorCulvertLaw<R>, rows: &Tensor<R>) -> Result<Self, StructureError>
    where
        R::Client: CulvertClient<R>,
    {
        let c = &law.barrel.client;
        Ok(Self {
            barrel: law.barrel.select(rows)?,
            unit: law.unit.select(rows)?,
            inlet: law.inlet.select(c, rows)?,
            outlet: law.outlet.select(c, rows)?,
        })
    }

    /// The rows at the I64 lane indices `owner`.
    pub(crate) fn select(&self, owner: &Tensor<R>) -> numr::error::Result<Self>
    where
        R::Client: CulvertClient<R>,
    {
        let c = &self.barrel.client;
        Ok(Self {
            barrel: self.barrel.select(owner)?,
            unit: self.unit.select(owner)?,
            inlet: self.inlet.select(c, owner)?,
            outlet: self.outlet.select(c, owner)?,
        })
    }
}

/// Encloses `exp(x)`. `cr_exp` rounds correctly, so one step outward holds the exact value.
pub(crate) fn exp<R>(a: &mut Arithmetic<'_, R>, x: &Interval<R>) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let c = a.client();
    let zero = Tensor::<R>::zeros(x.lower().shape(), DType::F64, c.device())?;
    let lower = c.maximum(&c.next_down(&c.cr_exp(x.lower())?)?, &zero)?;
    let upper = c.next_up(&c.cr_exp(x.upper())?)?;
    a.bounds(&lower, &upper)
}

/// Solver options for `width` rows.
pub(crate) fn options(width: usize) -> CertifiedBisectionOptions {
    options_with(width, RELATIVE_TOLERANCE)
}

/// Solver options for `width` rows with relative width `rtol`.
pub(crate) fn options_with(width: usize, rtol: f64) -> CertifiedBisectionOptions {
    CertifiedBisectionOptions {
        atol: 0.0,
        rtol,
        max_iterations: MAX_ITERATIONS,
        max_points: width.saturating_mul(2 * (MAX_ITERATIONS + 1)),
        probe: CertifiedRootProbe::SlackSecant,
    }
}

/// Status masks of a solve: rows with a root enclosure, and rows with no root in the domain.
pub(crate) struct Outcome<R: Runtime<DType = DType>> {
    /// `Converged`, `AccuracyFloor` or `WorkLimit`: the root lies in the returned enclosure.
    pub(crate) enclosed: Tensor<R>,
    /// `NoRoot`.
    pub(crate) absent: Tensor<R>,
    /// Root enclosures scattered over `base`.
    pub(crate) bounds: Bounds<R>,
}

/// Reads the solver outcome per row, with root enclosures written over `base`.
pub(crate) fn outcome<R>(
    c: &R::Client,
    roots: &CertifiedRootResult<R>,
    base: &Bounds<R>,
) -> Result<Outcome<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let status = c.cast(roots.status(), DType::F64)?;
    let like = |value: CertifiedRootStatus| -> numr::error::Result<Tensor<R>> {
        Tensor::<R>::full_scalar(status.shape(), DType::F64, value as i64 as f64, c.device())
    };
    let is = |value| -> Result<Tensor<R>, StructureError> { Ok(c.eq(&status, &like(value)?)?) };
    let enclosed = c.maximum(
        &c.maximum(
            &is(CertifiedRootStatus::Converged)?,
            &is(CertifiedRootStatus::AccuracyFloor)?,
        )?,
        &is(CertifiedRootStatus::WorkLimit)?,
    )?;
    let absent = is(CertifiedRootStatus::NoRoot)?;
    let [lower, upper] = base;
    let owner = roots.root_owner();
    let bounds = if owner.numel() == 0 {
        [lower.clone(), upper.clone()]
    } else {
        [
            c.index_put(lower, 0, owner, roots.root_lower())?,
            c.index_put(upper, 0, owner, roots.root_upper())?,
        ]
    };
    Ok(Outcome {
        enclosed,
        absent,
        bounds,
    })
}

/// The first lane where `mask` is 1, read to the host. Only error paths call it.
pub(crate) fn first<R>(mask: &Tensor<R>) -> Result<Option<usize>, StructureError>
where
    R: Runtime<DType = DType>,
{
    Ok(mask
        .try_to_vec::<f64>()?
        .iter()
        .position(|&value| value != 0.0))
}
