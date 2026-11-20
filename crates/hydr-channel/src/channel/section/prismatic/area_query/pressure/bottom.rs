use super::super::super::hydraulic_geometry::Constitutive;
use crate::channel::{
    flow::surveyed::{
        analytic_bottom,
        equation_ops::{choose, scalar, scale},
        error::{Arithmetic, HydraulicResult},
    },
    section::{GatheredBands, PrismaticAreaPointRecovery},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
};
use solvr::common::arithmetic::Interval;

pub(super) fn refine<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    source: &PrismaticAreaPointRecovery<R>,
    target: &Constitutive<R>,
    current: &Interval<R>,
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
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let width = a.bounds(
        &bands.start_width.lower().narrow(1, 0, 1)?.squeeze(Some(1)),
        &bands.start_width.upper().narrow(1, 0, 1)?.squeeze(Some(1)),
    )?;
    let slope = a.bounds(
        &bands.slope.lower().narrow(1, 0, 1)?.squeeze(Some(1)),
        &bands.slope.upper().narrow(1, 0, 1)?.squeeze(Some(1)),
    )?;
    let valid = bands.valid.narrow(1, 0, 1)?.squeeze(Some(1));
    let pointed = bands.pointed.narrow(1, 0, 1)?.squeeze(Some(1));
    let (rectangular, pointed) = analytic_bottom::kinds(a, &valid, &pointed, &width, &slope)?;
    let end = bands.end_area.lower().narrow(1, 0, 1)?.squeeze(Some(1));
    let end_point = a.expansion_point(&end)?;
    let source_difference = a.expansion_sub(source.requested_areas_m2(), &end_point)?;
    let source_difference = a.expansion_interval(&source_difference)?;
    let supported = c.mul(
        &c.le(source_difference.upper(), zero.upper())?,
        &c.le(target.area.upper(), &end)?,
    )?;
    let rectangular = c.mul(&rectangular, &supported)?;
    let pointed = c.mul(&pointed, &supported)?;
    let source_area = a.expansion_interval(source.requested_areas_m2())?;
    let first = choose(a, &rectangular, &source_area, &zero)?;
    let second = choose(a, &rectangular, &target.area, &zero)?;
    let first = scale(a, &first, 0.5)?;
    let second = scale(a, &second, 0.5)?;
    let mean = a.add(&first, &second)?;
    let width = choose(a, &rectangular, &width, &one)?;
    let rectangular_k = a.div_positive_interval(&mean, &width)?;
    let rectangular_k = scale(a, &rectangular_k, gravity)?;

    let first = choose(a, &pointed, source.depth_m(), &zero)?;
    let second = choose(a, &pointed, &target.depth, &zero)?;
    let maximum = a.bounds(
        &c.maximum(first.lower(), second.lower())?,
        &c.maximum(first.upper(), second.upper())?,
    )?;
    let minimum = a.bounds(
        &c.minimum(first.lower(), second.lower())?,
        &c.minimum(first.upper(), second.upper())?,
    )?;
    let positive = c.gt(maximum.lower(), zero.lower())?;
    let denominator = choose(a, &positive, &maximum, &one)?;
    let numerator = choose(a, &positive, &minimum, &zero)?;
    let ratio = a.div_positive_interval(&numerator, &denominator)?;
    let uncertain = c.mul(
        &c.eq(&positive, zero.lower())?,
        &c.gt(minimum.upper(), zero.upper())?,
    )?;
    let ratio = a.bounds(
        &c.minimum(&c.maximum(ratio.lower(), zero.lower())?, one.upper())?,
        &c.where_cond(
            &positive,
            &c.minimum(&c.maximum(ratio.upper(), zero.lower())?, one.upper())?,
            &uncertain,
        )?,
    )?;
    let square = a.mul(&ratio, &ratio)?;
    let sum = a.add(&one, &ratio)?;
    let numerator = a.add(&sum, &square)?;
    let denominator = scale(a, &sum, 3.0)?;
    let factor = a.div_positive_interval(&numerator, &denominator)?;
    let pointed_k = a.mul(&maximum, &factor)?;
    let pointed_k = scale(a, &pointed_k, gravity)?;
    let result = choose(a, &rectangular, &rectangular_k, current)?;
    choose(a, &pointed, &pointed_k, &result)
}
