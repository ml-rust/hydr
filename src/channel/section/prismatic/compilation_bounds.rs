use super::inputs::{BankInputs, column};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, IntervalContext};

pub(super) fn select<R>(
    arithmetic: &mut IntervalContext<'_, R, R::Client>,
    mask: &Tensor<R>,
    selected: &Interval<R>,
    otherwise: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let client = arithmetic.client();
    let lower = client.where_cond(mask, selected.lower(), otherwise.lower())?;
    let upper = client.where_cond(mask, selected.upper(), otherwise.upper())?;
    arithmetic.bounds(&lower, &upper)
}

pub(super) fn nonnegative<R>(
    arithmetic: &mut IntervalContext<'_, R, R::Client>,
    value: Interval<R>,
    zero: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    // Survey topology proves these physical quantities are nonnegative.
    let lower = arithmetic.client().maximum(value.lower(), zero)?;
    arithmetic.bounds(&lower, value.upper())
}

pub(super) fn bank_slope<R>(
    arithmetic: &mut IntervalContext<'_, R, R::Client>,
    bank: &BankInputs<R>,
    index: usize,
    rows: usize,
    zero: &Interval<R>,
    one: &Interval<R>,
) -> ArithmeticResult<(Interval<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let client = arithmetic.client();
    let vertical = column(&bank.vertical, index, rows)?;
    let left = arithmetic.point(&column(&bank.left_station, index, rows)?)?;
    let right = arithmetic.point(&column(&bank.right_station, index, rows)?)?;
    let low = arithmetic.point(&column(&bank.low_elevation, index, rows)?)?;
    let high = arithmetic.point(&column(&bank.high_elevation, index, rows)?)?;
    let extent = arithmetic.sub(&right, &left)?;
    let extent = nonnegative(arithmetic, extent, zero.lower())?;
    let span = arithmetic.sub(&high, &low)?;
    let positive_span = client.gt(span.lower(), zero.lower())?;
    let resolved = client.maximum(&vertical, &positive_span)?;
    let sloping = client.mul(&client.eq(&vertical, zero.lower())?, &positive_span)?;
    // Vertical walls need no quotient. Unresolved spans use a finite inactive branch.
    let numerator = select(arithmetic, &sloping, &extent, zero)?;
    let denominator = select(arithmetic, &sloping, &span, one)?;
    let slope = arithmetic.div_positive_interval(&numerator, &denominator)?;
    Ok((nonnegative(arithmetic, slope, zero.lower())?, resolved))
}

pub(super) fn stack<R>(
    arithmetic: &mut IntervalContext<'_, R, R::Client>,
    columns: &[Interval<R>],
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let lower: Vec<_> = columns.iter().map(Interval::lower).collect();
    let upper: Vec<_> = columns.iter().map(Interval::upper).collect();
    let lower = arithmetic.client().stack(&lower, 1)?;
    let upper = arithmetic.client().stack(&upper, 1)?;
    arithmetic.bounds(&lower, &upper)
}
