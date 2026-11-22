//! Construction-time layout of barrels built from tangent circular arcs.
//!
//! Notation: `y` up from the invert, `D` rise, `B` span. Arc `k` has centre `(x_k, y_k)` and
//! radius `r_k`. The normal angle `psi` runs from the downward vertical. The right wall point at
//! `psi` is `(x_k + r_k sin psi, y_k - r_k cos psi)`.
//!
//! Every layout has three bands from the invert: bottom, corner and top. A flat-floor arch
//! repeats its one arc in all three bands, each starting at the floor normal `psi_0` with the
//! floor wetted. Each band records its arc, its start angle, sine and cosine, and its start
//! depth, area, wetted perimeter and moment. The layout also records the full-state constants
//! and the widest depth.
//!
//! | Shape | Bottom arc | Corner arcs | Top arc |
//! | --- | --- | --- | --- |
//! | Pipe-arch | `r_b` given | `r_c` given | `r_t` derived |
//! | Horizontal ellipse | `R` derived | side radius given | `R` |
//! | Vertical ellipse | end radius given | `r_c` derived | end radius |
//! | Flat-floor arch | `r` derived | `r` | `r` |
//!
//! Every family runs on every row with a harmless admitted barrel on rows of another shape.

mod cap;
mod inputs;
mod layout;
#[cfg(test)]
mod layout_rows;
mod normals;

use super::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// One band: its arc and its start values. Lengths in metres, areas in square metres, moments in
/// cubic metres, angles in radians.
pub(crate) struct ArcBand<R: Runtime<DType = DType>> {
    pub(crate) centre_x: Bounds<R>,
    pub(crate) centre_y: Bounds<R>,
    pub(crate) radius: Bounds<R>,
    pub(crate) start_angle: Bounds<R>,
    pub(crate) start_sin: Bounds<R>,
    pub(crate) start_cos: Bounds<R>,
    pub(crate) start_depth: Bounds<R>,
    pub(crate) start_area: Bounds<R>,
    pub(crate) start_perimeter: Bounds<R>,
    pub(crate) start_moment: Bounds<R>,
}

/// Bands from the invert, full-state constants with the moment `M(D)`, and the widest depth.
pub(crate) struct ArcLayout<R: Runtime<DType = DType>> {
    pub(crate) bands: [ArcBand<R>; 3],
    pub(crate) full_area: Bounds<R>,
    pub(crate) full_perimeter: Bounds<R>,
    pub(crate) full_radius: Bounds<R>,
    pub(crate) full_moment: Bounds<R>,
    pub(crate) widest_depth: Bounds<R>,
}

/// Intersection of two enclosures of one value.
fn meet<R>(
    a: &mut Arithmetic<'_, R>,
    x: &Interval<R>,
    y: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.maximum(x.lower(), y.lower())?;
    let upper = c.minimum(x.upper(), y.upper())?;
    a.bounds(&lower, &upper)
}

/// Endpoints of `on` where `mask` is 1 and of `off` where it is 0.
fn pick<R>(
    a: &Arithmetic<'_, R>,
    mask: &Tensor<R>,
    on: &Interval<R>,
    off: &Interval<R>,
) -> ArithmeticResult<Bounds<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    Ok([
        c.where_cond(mask, on.lower(), off.lower())?,
        c.where_cond(mask, on.upper(), off.upper())?,
    ])
}
