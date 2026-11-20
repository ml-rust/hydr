use super::super::super::hydraulic_geometry::Constitutive;
use super::{bottom, weights};
use crate::channel::{
    flow::surveyed::{
        equation_ops::{choose, scalar, scale},
        error::{Arithmetic, HydraulicResult},
        pressure_segments,
    },
    section::{GatheredBands, HydraulicGeometry, PrismaticAreaPointRecovery},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) fn secant<R>(
    a: &mut Arithmetic<'_, R>,
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    source: &PrismaticAreaPointRecovery<R>,
    target: &Constitutive<R>,
    delta: &CertifiedExpansion<R>,
    gravity: f64,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let first = source.depth_m();
    let second = &target.depth;
    let low = a.bounds(
        &c.minimum(first.lower(), second.lower())?,
        &c.minimum(first.upper(), second.upper())?,
    )?;
    let high = a.bounds(
        &c.maximum(first.lower(), second.lower())?,
        &c.maximum(first.upper(), second.upper())?,
    )?;
    let span = a.bounds(low.lower(), high.upper())?;
    let connected = geometry.at_depth(a, bands, &span, None)?;
    let fallback = scale(a, &connected.hydraulic_depth, gravity)?;
    let beta = a.point(&c.midpoint(fallback.lower(), fallback.upper())?)?;
    let delta = a.expansion_interval(delta)?;
    let positive = c.gt(delta.lower(), zero.lower())?;
    let negative = c.lt(delta.upper(), zero.upper())?;
    let separated = c.maximum(&positive, &negative)?;
    let minus = scale(a, &delta, -1.0)?;
    let distance = choose(a, &positive, &delta, &minus)?;
    let denominator = choose(a, &separated, &distance, &one)?;
    let segments = pressure_segments::local_segments(a, bands, &low, &high, &zero)?;
    let valid_width = c.mul(
        &bands.valid,
        &c.gt(segments.area_secant.lower(), zero.lower())?,
    )?;
    let divisor = choose(a, &valid_width, &segments.area_secant, &one)?;
    let numerator = choose(a, &valid_width, &segments.integral_secant, &zero)?;
    let local = a.div_positive_interval(&numerator, &divisor)?;
    let local = scale(a, &local, gravity)?;
    let count = first.lower().numel();
    let derivative = a.bounds(
        &fallback.lower().reshape(&[count, 1])?,
        &fallback.upper().reshape(&[count, 1])?,
    )?;
    let local = choose(a, &valid_width, &local, &derivative)?;
    let segment_area = a.mul(&segments.depth_increment, &segments.area_secant)?;
    let active = c.mul(&bands.valid, &separated.reshape(&[count, 1])?)?;
    let segment_area = choose(a, &active, &segment_area, &zero)?;
    let denominator = a.bounds(
        &denominator.lower().reshape(&[count, 1])?,
        &denominator.upper().reshape(&[count, 1])?,
    )?;
    let weight = weights::band_weights(a, &segment_area, &denominator, &zero, &one)?;
    let anchor = a.bounds(
        &beta.lower().reshape(&[count, 1])?,
        &beta.upper().reshape(&[count, 1])?,
    )?;
    let difference = a.sub(&local, &anchor)?;
    let terms = a.mul(&weight, &difference)?;
    // The fixed conditioner cancels because the physical band weights sum to one.
    let mut result = beta;
    for band in 0..bands.band_count {
        let term = a.bounds(
            &terms.lower().narrow(1, band, 1)?.squeeze(Some(1)),
            &terms.upper().narrow(1, band, 1)?.squeeze(Some(1)),
        )?;
        result = a.add(&result, &term)?;
    }
    let result = choose(a, &separated, &result, &fallback)?;
    let result = a.bounds(
        &c.maximum(result.lower(), fallback.lower())?,
        &c.minimum(result.upper(), fallback.upper())?,
    )?;
    bottom::refine(a, bands, source, target, &result, gravity)
}
