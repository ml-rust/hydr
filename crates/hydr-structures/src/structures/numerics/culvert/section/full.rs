//! Full state from the crown upward: depth above the crown changes no geometry.
//!
//! | Quantity | Circle | Box | Arc |
//! | --- | --- | --- | --- |
//! | `A_full` | `pi D^2 / 4` | `B D` | band chain to the crown |
//! | `T` | `0` | `0` | `0` |
//! | `P_full` | `pi D` | `2 (B + D)` | band chain to the crown |
//! | `R_full` | `D / 4` | `B D / (2 (B + D))` | `A_full / P_full` |
//! | `M` | `A_full (y - D/2)` | `A_full (y - D/2)` | `M(D) + A_full (y - D)` |
//!
//! A pipe-arch or an arch has its centroid off `D/2`, so arc rows add the head above the crown
//! to the crown moment `M(D)`.

use super::Quantities;
use crate::structures::numerics::culvert::constants::PI;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose};
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::ConditionalOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Full-state area, wetted perimeter and hydraulic radius enclosures.
pub(crate) struct FullIntervals<R: Runtime<DType = DType>> {
    /// `A_full`, in square metres.
    pub(crate) area: Interval<R>,
    /// `P_full`, in metres.
    pub(crate) perimeter: Interval<R>,
    /// `R_full = A_full / P_full`, in metres.
    pub(crate) radius: Interval<R>,
}

/// Encloses `[A_full, P_full, R_full]` for rows of shape mask `circular`, span and rise.
///
/// The barrel context stores the endpoints. It calls this once, at construction.
pub(crate) fn full_constants<R>(
    a: &mut Arithmetic<'_, R>,
    circular: &Tensor<R>,
    span: &Tensor<R>,
    rise: &Tensor<R>,
) -> ArithmeticResult<[Bounds<R>; 3]>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let shape = rise.shape();
    let device = c.device();
    let constant = |value: f64| Tensor::<R>::full_scalar(shape, DType::F64, value, device);
    let pi = a.bounds(
        &constant(f64::from_bits(PI[0]))?,
        &constant(f64::from_bits(PI[1]))?,
    )?;
    let d = a.point(rise)?;
    let b = a.point(span)?;
    let quarter = a.point(&constant(0.25)?)?;
    let two = a.point(&constant(2.0)?)?;

    let square = a.mul(&d, &d)?;
    let pi_square = a.mul(&pi, &square)?;
    let circle_area = a.mul(&pi_square, &quarter)?;
    let circle_perimeter = a.mul(&pi, &d)?;
    // D/4 is exact, because D 2^-30 is normal.
    let circle_radius = a.mul(&d, &quarter)?;

    let box_area = a.mul(&b, &d)?;
    let sum = a.add(&b, &d)?;
    let box_perimeter = a.mul(&two, &sum)?;
    let box_radius = a.div_positive_interval(&box_area, &box_perimeter)?;

    let choose = |circle: &Interval<R>, rectangle: &Interval<R>| -> ArithmeticResult<Bounds<R>> {
        Ok([
            c.where_cond(circular, circle.lower(), rectangle.lower())?,
            c.where_cond(circular, circle.upper(), rectangle.upper())?,
        ])
    };
    Ok([
        choose(&circle_area, &box_area)?,
        choose(&circle_perimeter, &box_perimeter)?,
        choose(&circle_radius, &box_radius)?,
    ])
}

/// Re-enters the stored full-state enclosures into the current interval call.
pub(crate) fn full<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
) -> ArithmeticResult<FullIntervals<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let enter = |a: &mut Arithmetic<'_, R>, [lower, upper]: &Bounds<R>| a.bounds(lower, upper);
    Ok(FullIntervals {
        area: enter(a, &barrel.full_area)?,
        perimeter: enter(a, &barrel.full_perimeter)?,
        radius: enter(a, &barrel.full_radius)?,
    })
}

/// Encloses the full-state geometry at depths `depth >= D`.
pub(super) fn at_depth<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<Quantities<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let FullIntervals {
        area,
        perimeter,
        radius,
    } = full(a, barrel)?;
    let y = a.point(depth)?;
    let centroid = a.point(&barrel.half_rise)?;
    let head = a.sub(&y, &centroid)?;
    let mut moment = a.mul(&area, &head)?;
    if let Some(rows) = &barrel.arcs {
        let [lower, upper] = &rows.full_moment;
        let crown = a.bounds(lower, upper)?;
        let d = a.point(&barrel.rise)?;
        let above = a.sub(&y, &d)?;
        let pressed = a.mul(&area, &above)?;
        let arc = a.add(&crown, &pressed)?;
        moment = choose(a, &rows.arced, &arc, &moment)?;
    }
    let top_width = a.point(&barrel.zero)?;
    Ok(Quantities {
        area,
        top_width,
        perimeter,
        radius,
        moment,
    })
}
