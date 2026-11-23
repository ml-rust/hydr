//! Certified outlet-control headwater `HW_o(Q, TW)` for circle, box and arc barrels.
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
//!
//! Above a floor the march runs on the open section, with depths, tailwater and heads from the
//! bed. Friction takes the weighted perimeter `P_n = P_w + rho^(3/2) b` with the wall `n`, so
//! `Sf = n^2 Q^2 P_n^(4/3) / A^(10/3)`, and full flow takes `R = A'_full / P_n,full`. A box keeps
//! its crown `B` in `P_n,full`. Critical and normal depth solve on the declared rows, which
//! carry the open-section certificates. A face of blocked share `f` takes the entrance loss
//! `ke_f = ke / (1 - f)^2 + (f / (1 - f))^2`.

mod assemble;
mod bands;
mod classes;
mod coordinates;
mod depth;
mod integrand;
mod jet;
pub(crate) mod jump;
mod junctions;
mod levels;
mod partition;
mod quadrature;
mod screen;
mod slope;
mod trig;
mod trig_ends;
mod zone;

pub(crate) use bands::{ArcShape, Outline};
pub(crate) use levels::Accuracy;
pub(crate) use slope::{SlopeOperands, head_slopes};

use super::{Arithmetic, BarrelClient, BarrelDepths, RowNames, critical, normal};
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorOutletContext};
use crate::structures::{CulvertBarrelShape, CulvertOutletRow, StructureError};
use numr::dtype::DType;
use numr::ops::ConditionalOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, with_interval_arithmetic};
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
    let open_rows = rows
        .iter()
        .any(|r| r.barrel.floor().is_some() && r.barrel.shape() != CulvertBarrelShape::Box);
    let entrance = column(&|r| r.entrance_loss)?;
    let face = column(&|r| r.face_blockage.blocked_share())?;
    let c = &barrel.client;
    let ends = |x: &Interval<R>| -> Bounds<R> { [x.lower().clone(), x.upper().clone()] };
    let (beta, bed, entrance, arcs) =
        with_interval_arithmetic(c, DType::F64, "culvert outlet shape", |a| {
            let span = a.point(&barrel.span)?;
            let ratio = a.div_positive(&span, barrel.open_rise())?;
            let beta = ends(&ratio);
            let bed = match barrel.floor.as_deref() {
                Some(rows) => {
                    let weight = a.bounds(&rows.weight[0], &rows.weight[1])?;
                    let weighted = a.mul(&weight, &ratio)?;
                    let [lower, upper] = ends(&weighted);
                    [
                        c.where_cond(&rows.boxed, &lower, &beta[0])?,
                        c.where_cond(&rows.boxed, &upper, &beta[1])?,
                    ]
                }
                None => beta.clone(),
            };
            let base = match &barrel.unit_arcs {
                Some(rows) => Some(ArcShape::new(a, rows)?),
                None => None,
            };
            let arcs = if open_rows {
                Some(ArcShape::with_floors(a, barrel, base)?)
            } else {
                base
            };
            Ok((
                beta,
                bed,
                ends(&face_loss(a, barrel, &entrance, &face)?),
                arcs,
            ))
        })?;
    Ok(TensorOutletContext {
        length: column(&|r| r.length_m)?,
        manning: column(&|r| r.manning_n)?,
        entrance,
        slope: column(&|r| r.slope)?,
        gravity: column(&|r| r.gravity_m_per_s2)?,
        beta,
        bed,
        arcs,
        open: barrel.march_view()?.map(Box::new),
    })
}

/// Encloses the face entrance loss `ke / (1 - f)^2 + (f / (1 - f))^2` per row: the entrance
/// loss at the face velocity `V / (1 - f)` plus the Borda-Carnot expansion
/// `(V / (1 - f) - V)^2 / (2 g)` back to the barrel velocity. Rows with `f = 0` hold `ke`
/// exactly.
fn face_loss<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    entrance: &Tensor<R>,
    face: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let one = a.point(&barrel.one)?;
    let blocked = a.point(face)?;
    let open = a.sub(&one, &blocked)?;
    let square = a.mul(&open, &open)?;
    let ke = a.point(entrance)?;
    let loss = a.div_positive_interval(&ke, &square)?;
    let ratio = a.div_positive_interval(&blocked, &open)?;
    let expansion = a.mul(&ratio, &ratio)?;
    a.add(&loss, &expansion)
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
    // Depths solve on the declared rows. The profile runs on each open section.
    let barrel = outlet.view(barrel);
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
