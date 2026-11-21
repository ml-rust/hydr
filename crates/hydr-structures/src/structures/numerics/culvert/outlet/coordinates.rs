//! Maps between depth and the profile coordinate `v`, with the flow area at a coordinate.
//!
//! A circle uses the half angle `v = phi`, with `y = D (1 - cos phi) / 2`. A box uses `v = y/D`.

use super::integrand::{part, shape};
use super::jet;
use super::trig::{fill, sin_cos};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose};
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Coordinate `v` of a depth enclosure `[lower, upper]` in metres, clipped to the barrel.
pub(crate) fn coordinate<R>(
    a: &mut Arithmetic<'_, R>,
    circular: &Tensor<R>,
    rise: &Tensor<R>,
    depth: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let s = a.div_positive(depth, rise)?;
    fraction_coordinate(a, circular, &s)
}

/// Coordinate `v` of a depth fraction enclosure `s = y/D`, clipped to the barrel.
pub(crate) fn fraction_coordinate<R>(
    a: &mut Arithmetic<'_, R>,
    circular: &Tensor<R>,
    s: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let like = circular;
    let s = clip(a, s, 0.0, 1.0)?;
    let one = a.point(&fill(like, 1.0)?)?;
    let two = a.point(&fill(like, 2.0)?)?;
    let doubled = a.mul(&two, &s)?;
    let cosine = a.sub(&one, &doubled)?;
    // The exact cosine lies in [-1, 1]. Outward rounding may step one F64 past either end.
    let cosine = clip(a, &cosine, -1.0, 1.0)?;
    let angle = a.acos(&cosine)?;
    choose(a, circular, &angle, &s)
}

/// Intersects `x` with `[low, high]`, collapsing an interval wholly outside to the nearer end.
fn clip<R>(
    a: &mut Arithmetic<'_, R>,
    x: &Interval<R>,
    low: f64,
    high: f64,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = x.lower();
    let floor = fill(like, low)?;
    let ceiling = fill(like, high)?;
    let lower = c.minimum(&c.maximum(x.lower(), &floor)?, &ceiling)?;
    let upper = c.maximum(&c.minimum(x.upper(), &ceiling)?, &floor)?;
    a.bounds(&lower, &upper)
}

/// Depth in metres of a coordinate enclosure `v`.
pub(crate) fn depth<R>(
    a: &mut Arithmetic<'_, R>,
    circular: &Tensor<R>,
    rise: &Tensor<R>,
    v: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let like = rise;
    let [_, cosine] = sin_cos(a, v)?;
    let one = a.point(&fill(like, 1.0)?)?;
    let gap = a.sub(&one, &cosine)?;
    let circle = part(a, &gap, 2.0)?;
    let s = choose(a, circular, &circle, v)?;
    let s = clip(a, &s, 0.0, 1.0)?;
    let d = a.point(rise)?;
    a.mul(&d, &s)
}

/// Flow area in square metres of a coordinate enclosure `v` below the crown.
pub(crate) fn area<R>(
    a: &mut Arithmetic<'_, R>,
    circular: &Tensor<R>,
    beta: &Bounds<R>,
    rise: &Tensor<R>,
    v: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let section = shape(a, circular, beta, v, 1)?;
    let unit = jet::coefficient(a, &section.area, 0, rise)?;
    let d = a.point(rise)?;
    let square = a.mul(&d, &d)?;
    a.mul(&square, &unit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureError;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    #[test]
    fn coordinates_round_trip_and_areas_match_closed_forms() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let tensor =
            |values: &[f64]| Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &device);
        let circular = tensor(&[1.0, 0.0])?;
        let rise = tensor(&[2.0, 1.5])?;
        let beta = [tensor(&[1.0, 4.0 / 3.0])?, tensor(&[1.0, 4.0 / 3.0])?];
        let depths = tensor(&[1.0, 0.6])?;
        let [back, square] = with_interval_arithmetic(&client, DType::F64, "coordinates", |a| {
            let y = a.point(&depths)?;
            let v = coordinate(a, &circular, &rise, &y)?;
            let back = depth(a, &circular, &rise, &v)?;
            let square = area(a, &circular, &beta, &rise, &v)?;
            Ok([
                [
                    back.lower().try_to_vec::<f64>()?,
                    back.upper().try_to_vec::<f64>()?,
                ],
                [
                    square.lower().try_to_vec::<f64>()?,
                    square.upper().try_to_vec::<f64>()?,
                ],
            ])
        })?;
        // A half-full circle of diameter 2 holds pi/2. The box holds B y = 2 x 0.6.
        let exact = [(1.0, std::f64::consts::FRAC_PI_2), (0.6, 1.2)];
        for (row, (y, area)) in exact.into_iter().enumerate() {
            assert!(back[0][row] <= y && y <= back[1][row], "row={row}");
            assert!(back[1][row] - back[0][row] < 1e-14, "row={row}");
            assert!(square[0][row] <= area * (1.0 + 1e-15), "row={row}");
            assert!(area * (1.0 - 1e-15) <= square[1][row], "row={row}");
        }
        Ok(())
    }
}
