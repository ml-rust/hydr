//! Certified outlet-control headwater `HW_o(Q, TW)` for circle and box barrels.
//!
//! The march starts at the outlet. A full outlet segment carries head at the full-barrel friction
//! rate `Sf_full - S` per metre upstream, with `Sf_full = n^2 V^2 / R^(4/3)`. A free segment
//! follows the direct-step profile `dx/dy = (1 - Fr^2) / (S - Sf)` from `max(TW, yc)`, which is
//! the zero-tail-velocity exit loss. Solvr certified Simpson integrates the profile over a graded
//! partition in the smooth coordinate of [`integrand`]. Solvr certified bisection finds the depth
//! where the profile has covered the barrel. A steep barrel whose subcritical profile reaches
//! critical depth takes critical depth at the inlet: `HW_o = yc + (1 + ke) Vc^2 / (2 g)`.
//!
//! Each query runs the critical and normal depth solves, one classification call, one profile
//! integral, one depth solve and one assembly call. An [`Accuracy`] sets the quadrature
//! budget of the profile and the widths that stop the depth solves.

mod assemble;
mod classes;
mod coordinates;
mod depth;
mod integrand;
mod jet;
pub(crate) mod jump;
mod levels;
mod partition;
mod quadrature;
mod screen;
mod slope;
mod trig;
mod trig_ends;

pub(crate) use levels::Accuracy;
pub(crate) use slope::{SlopeOperands, head_slopes};

use super::{BarrelClient, BarrelDepths, RowNames, critical, normal};
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorOutletContext};
use crate::structures::{CulvertOutletRow, StructureError};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedBisectionAlgorithms;

/// Outlet headwater per row: status codes with inlet head and headwater bounds in metres.
pub(crate) struct OutletHeadwater<R: Runtime<DType = DType>> {
    pub(crate) status: Tensor<R>,
    pub(crate) inlet: Bounds<R>,
    pub(crate) headwater: Bounds<R>,
}

/// Everything one headwater query computed, for the jump query to reuse.
pub(crate) struct March<R: Runtime<DType = DType>> {
    pub(crate) headwater: OutletHeadwater<R>,
    pub(crate) classes: classes::Classes<R>,
    pub(crate) reach: assemble::Reach<R>,
    pub(crate) profile: partition::Profile<R>,
    pub(crate) normal: BarrelDepths<R>,
}

/// Uploads the outlet rows.
pub(crate) fn prepare<R>(
    barrel: &TensorBarrelContext<R>,
    rows: &[CulvertOutletRow],
) -> Result<TensorOutletContext<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let device = &barrel.device;
    let m = rows.len();
    let column = |f: &dyn Fn(&CulvertOutletRow) -> f64| -> numr::error::Result<Tensor<R>> {
        let values: Vec<f64> = rows.iter().map(f).collect();
        Tensor::<R>::from_slice(&values, &[m], device)
    };
    let beta = with_interval_arithmetic(&barrel.client, DType::F64, "culvert outlet shape", |a| {
        let span = a.point(&barrel.span)?;
        let ratio = a.div_positive(&span, &barrel.rise)?;
        Ok([ratio.lower().clone(), ratio.upper().clone()])
    })?;
    Ok(TensorOutletContext {
        length: column(&|r| r.length_m)?,
        manning: column(&|r| r.manning_n)?,
        entrance: column(&|r| r.entrance_loss)?,
        slope: column(&|r| r.slope)?,
        gravity: column(&|r| r.gravity_m_per_s2)?,
        beta,
    })
}

/// Runs the outlet march for every row at the discharges and tailwaters given.
///
/// The caller checks that every discharge and tailwater is finite and nonnegative. `names`
/// names rows in errors.
pub(crate) fn march<R>(
    names: RowNames<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    outlet: &TensorOutletContext<R>,
    discharge: &Tensor<R>,
    tailwater: &Tensor<R>,
    accuracy: Accuracy,
) -> Result<March<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R> + CertifiedSimpsonAlgorithms<R>,
{
    let c = &barrel.client;
    let rtol = accuracy.root_tolerance();
    let critical = critical::solve_to(names, barrel, unit, discharge, &outlet.gravity, rtol)?;
    let normal = normal::solve_to(
        names,
        barrel,
        unit,
        [discharge, &outlet.manning, &outlet.slope],
        rtol,
    )?;
    let operands = classes::Operands {
        barrel,
        outlet,
        discharge,
        tailwater,
        critical: &critical,
        normal: &normal,
    };
    let classes = with_interval_arithmetic(c, DType::F64, "culvert outlet classes", |a| {
        classes::classify(a, &operands)
    })?;
    let profile = partition::build(c, classes.spec.clone(), accuracy)?;
    let reach = with_interval_arithmetic(c, DType::F64, "culvert outlet reach", |a| {
        assemble::reach(a, barrel, &classes, &profile)
    })?;
    let rows: Vec<i64> = (0..barrel.width())
        .map(|row| i64::try_from(row).unwrap_or(i64::MAX))
        .collect();
    let rows = Tensor::<R>::from_slice(&rows, &[barrel.width()], &barrel.device)?;
    let inlet = depth::depth_at(
        c,
        &profile,
        &rows,
        &classes.target,
        &reach.solve,
        accuracy.depth_tolerance(),
    )?;
    let parts = assemble::Parts {
        barrel,
        outlet,
        classes: &classes,
        reach: &reach,
        inlet: &inlet,
    };
    let assembled = with_interval_arithmetic(c, DType::F64, "culvert outlet headwater", |a| {
        assemble::assemble(a, &parts)
    })?;
    Ok(March {
        headwater: OutletHeadwater {
            status: assembled.status,
            inlet: assembled.inlet,
            headwater: assembled.headwater,
        },
        classes,
        reach,
        profile,
        normal,
    })
}
