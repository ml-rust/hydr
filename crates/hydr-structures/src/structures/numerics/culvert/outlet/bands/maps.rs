//! Maps between the depth fraction `s = y/D` and the normal angle `psi` on arc rows.
//!
//! On band `k`, `s = y_k - r_k cos psi` in units of the rise. Both directions are monotone, so
//! a depth enclosure maps through its two ends. A point inside a junction depth or angle
//! enclosure takes the hull of both bands.

use super::{ArcShape, hull};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, clip};
use numr::dtype::DType;
use numr::runtime::Runtime;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Normal angle of the depth fraction point `s`, clipped to `[0, pi]`.
fn angle_at<R>(
    a: &mut Arithmetic<'_, R>,
    arcs: &ArcShape<R>,
    s: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut angles = Vec::with_capacity(3);
    for band in &arcs.bands {
        let centre = a.bounds(&band.centre_y[0], &band.centre_y[1])?;
        let radius = a.bounds(&band.radius[0], &band.radius[1])?;
        let drop = a.sub(&centre, s)?;
        let cosine = a.div_positive_interval(&drop, &radius)?;
        // The exact cosine lies in [-1, 1]. Outward rounding may step past either end.
        let cosine = clip(a, &cosine, -1.0, 1.0)?;
        angles.push(a.acos(&cosine)?);
    }
    match angles.as_slice() {
        [first, second, third] => {
            let members = arcs.depth_members(a.client(), s.lower())?;
            hull(a, &members, [first, second, third])
        }
        [only] => a.bounds(only.lower(), only.upper()),
        _ => Err(numr::error::Error::InvalidArgument {
            arg: "culvert outlet arc bands",
            reason: "no band to map: report this runtime".to_string(),
        }
        .into()),
    }
}

/// Normal angle enclosure of a depth fraction enclosure `s` inside `[0, 1]`.
pub(crate) fn angle_of<R>(
    a: &mut Arithmetic<'_, R>,
    arcs: &ArcShape<R>,
    s: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let low = a.point(s.lower())?;
    let high = a.point(s.upper())?;
    let at_low = angle_at(a, arcs, &low)?;
    let at_high = angle_at(a, arcs, &high)?;
    a.bounds(at_low.lower(), at_high.upper())
}

/// Depth fraction enclosure of the angle enclosure `psi`, whose cosine is `cosine`.
pub(crate) fn fraction_of<R>(
    a: &mut Arithmetic<'_, R>,
    arcs: &ArcShape<R>,
    psi: &Interval<R>,
    cosine: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut depths = Vec::with_capacity(3);
    for band in &arcs.bands {
        let centre = a.bounds(&band.centre_y[0], &band.centre_y[1])?;
        let radius = a.bounds(&band.radius[0], &band.radius[1])?;
        let drop = a.mul(&radius, cosine)?;
        depths.push(a.sub(&centre, &drop)?);
    }
    let s = match depths.as_slice() {
        [first, second, third] => {
            let members = arcs.members(a.client(), psi.lower(), psi.upper())?;
            hull(a, &members, [first, second, third])?
        }
        [only] => a.bounds(only.lower(), only.upper())?,
        _ => {
            return Err(numr::error::Error::InvalidArgument {
                arg: "culvert outlet arc bands",
                reason: "no band to map: report this runtime".to_string(),
            }
            .into());
        }
    };
    clip(a, &s, 0.0, 1.0)
}
