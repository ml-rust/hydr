use crate::channel::flow::surveyed::{
    analytic_bottom,
    error::{Arithmetic, HydraulicResult},
};
use crate::channel::section::GatheredBands;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

use super::analytic_bottom_arithmetic::{AnalyticOperands, evaluate};

/// Only eligible rows contain available analytic source certificates.
pub(crate) struct AnalyticSource<R: Runtime<DType = DType>> {
    pub(crate) eligible: Tensor<R>,
    pub(crate) rectangular: Tensor<R>,
    pub(crate) pointed: Tensor<R>,
    /// Ineligible rows contain unavailable zero placeholders.
    pub(crate) potential: CertifiedExpansion<R>,
    /// Ineligible rows contain unavailable zero placeholders.
    pub(crate) sound: CertifiedExpansion<R>,
    /// Retains the pre-root radicand relation, including its complete arithmetic radius.
    pub(crate) sound_squared: CertifiedExpansion<R>,
}

/// Use captured source area and its complete recovered-depth certificate.
/// Gravity comes from the checked Riemann config. Noneligible rows require the complete general branch.
pub(crate) fn source<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    exact_area: &CertifiedExpansion<R>,
    complete_depth: &Interval<R>,
    gravity: f64,
) -> HydraulicResult<AnalyticSource<R>>
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
    let zero = Tensor::zeros(exact_area.high().shape(), DType::F64, c.device())?;
    let one = Tensor::ones(exact_area.high().shape(), DType::F64, c.device())?;
    let column = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        Ok(value.narrow(1, 0, 1)?.squeeze(Some(1)))
    };
    let width = a.bounds(
        &column(bands.start_width.lower())?,
        &column(bands.start_width.upper())?,
    )?;
    let slope = a.bounds(&column(bands.slope.lower())?, &column(bands.slope.upper())?)?;
    let (rectangular, pointed) = analytic_bottom::kinds(
        a,
        &column(&bands.valid)?,
        &column(&bands.pointed)?,
        &width,
        &slope,
    )?;
    let end_area = a.expansion_point(&column(bands.end_area.lower())?)?;
    let difference = a.expansion_sub(exact_area, &end_area)?;
    let difference = a.expansion_interval(&difference)?;
    let supported = c.le(difference.upper(), &zero)?;
    let rectangular = c.mul(&rectangular, &supported)?;
    let pointed = c.mul(&pointed, &supported)?;
    let eligible = c.maximum(&rectangular, &pointed)?;

    evaluate(
        a,
        AnalyticOperands {
            eligible,
            rectangular,
            pointed,
            exact_area,
            complete_depth,
            width: &width,
            zero: &zero,
            one: &one,
            gravity,
        },
    )
}
