use super::equation_ops::{choose, scalar, scale};
use super::error::{Arithmetic, HydraulicResult};
use crate::channel::section::{Constitutive, GatheredBands, HydraulicGeometry};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) fn pressure<R>(
    a: &mut Arithmetic<'_, R>,
    state: &Constitutive<R>,
    gravity: f64,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    scale(a, &state.integral, gravity)
}

pub(crate) fn pressure_secant<R>(
    a: &mut Arithmetic<'_, R>,
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    first: &Constitutive<R>,
    second: &Constitutive<R>,
    gravity: f64,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let count = first.depth.lower().numel();
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let low = a.bounds(
        &c.minimum(first.depth.lower(), second.depth.lower())?,
        &c.minimum(first.depth.upper(), second.depth.upper())?,
    )?;
    let high = a.bounds(
        &c.maximum(first.depth.lower(), second.depth.lower())?,
        &c.maximum(first.depth.upper(), second.depth.upper())?,
    )?;
    let span = a.bounds(low.lower(), high.upper())?;
    // Convexity places every pressure secant between the endpoint pressure derivatives, including equality.
    let connected = geometry.at_depth(a, bands, &span, None)?;
    let mut result = scale(a, &connected.hydraulic_depth, gravity)?;
    let segments = super::pressure_segments::local_segments(a, bands, &low, &high, &zero)?;
    let area_secant = segments.area_secant;
    let integral_secant = segments.integral_secant;
    let delta = segments.depth_increment;
    let area_increment = a.mul(&delta, &area_secant)?;
    let integral_increment = a.mul(&delta, &integral_secant)?;
    let area_increment = choose(a, &bands.valid, &area_increment, &zero)?;
    let integral_increment = choose(a, &bands.valid, &integral_increment, &zero)?;
    let mut total_area = a.point(&Tensor::zeros(&[count], DType::F64, c.device())?)?;
    let mut total_integral = a.point(&Tensor::zeros(&[count], DType::F64, c.device())?)?;
    for band in 0..bands.band_count {
        let area = column(a, &area_increment, band)?;
        let integral = column(a, &integral_increment, band)?;
        total_area = a.add(&total_area, &area)?;
        total_integral = a.add(&total_integral, &integral)?;
    }
    let positive = c.gt(total_area.lower(), zero.lower())?;
    let denominator = choose(a, &positive, &total_area, &one)?;
    let quotient = a.div_positive_interval(&total_integral, &denominator)?;
    let direct = scale(a, &quotient, gravity)?;
    result = choose(a, &positive, &direct, &result)?;
    // Within a common certified band, cancel the depth difference before interval evaluation.
    for band in 0..bands.band_count {
        let start = column(a, &bands.start_depth, band)?;
        let end = column(a, &bands.end_depth, band)?;
        let valid = bands.valid.narrow(1, band, 1)?.squeeze(Some(1));
        let common = c.mul(
            &valid,
            &c.mul(
                &c.ge(low.lower(), start.upper())?,
                &c.le(high.upper(), end.lower())?,
            )?,
        )?;
        let denominator = column(a, &area_secant, band)?;
        let positive = c.mul(&common, &c.gt(denominator.lower(), zero.lower())?)?;
        let denominator = choose(a, &positive, &denominator, &one)?;
        let numerator = column(a, &integral_secant, band)?;
        let quotient = a.div_positive_interval(&numerator, &denominator)?;
        let local = scale(a, &quotient, gravity)?;
        result = choose(a, &positive, &local, &result)?;
    }
    Ok(result)
}

fn column<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    band: usize,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.bounds(
        &value.lower().narrow(1, band, 1)?.squeeze(Some(1)),
        &value.upper().narrow(1, band, 1)?.squeeze(Some(1)),
    )?)
}
