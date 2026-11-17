use super::error::{Arithmetic, HydraulicResult, checked};
use super::integration::{complete_integration, retain_cause};
use super::work::{QueryWork, WorkLedger};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;
use solvr::integrate::{
    CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, CertifiedSimpsonResult,
    CertifiedSimpsonValues,
};
use std::cell::Cell;

pub(crate) fn numerical_contributions<R, F, D>(
    c: &R::Client,
    left: &Tensor<R>,
    right: &Tensor<R>,
    options: CertifiedSimpsonOptions,
    work: &WorkLedger,
    mut value_callback: F,
    mut fourth_callback: D,
) -> HydraulicResult<CertifiedSimpsonResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
    F: FnMut(&mut Arithmetic<'_, R>, &Tensor<R>, &Interval<R>) -> HydraulicResult<Interval<R>>,
    D: FnMut(&mut Arithmetic<'_, R>, &Tensor<R>, &Interval<R>) -> HydraulicResult<Tensor<R>>,
{
    let cause = Cell::new(None);
    let result = c.certified_simpson(
        left,
        right,
        options,
        |query| {
            let result = work
                .charge(QueryWork::SimpsonValue, query.owner.numel())
                .and_then(|()| {
                    checked(c, "surveyed Simpson values", |a| {
                        let x = a.bounds(&query.lower, &query.upper)?;
                        let value = value_callback(a, &query.owner, &x)?;
                        let point = a.point(&query.point)?;
                        let nominal_bounds = value_callback(a, &query.owner, &point)?;
                        let half_width = c.mul_scalar(
                            &c.sub(nominal_bounds.upper(), nominal_bounds.lower())?,
                            0.5,
                        )?;
                        let nominal = c.add(nominal_bounds.lower(), &half_width)?;
                        let nominal =
                            c.maximum(value.lower(), &c.minimum(&nominal, value.upper())?)?;
                        Ok(CertifiedSimpsonValues {
                            value: nominal,
                            lower: value.lower().clone(),
                            upper: value.upper().clone(),
                        })
                    })
                });
            result.map_err(|error| retain_cause(&cause, error))
        },
        |panel| {
            let result = work
                .charge(QueryWork::SimpsonDerivative, panel.owner.numel())
                .and_then(|()| {
                    checked(c, "surveyed Simpson fourth derivatives", |a| {
                        let x = a.bounds(&panel.lower, &panel.upper)?;
                        fourth_callback(a, &panel.owner, &x)
                    })
                });
            result.map_err(|error| retain_cause(&cause, error))
        },
    );
    complete_integration(result, cause.take())
}
