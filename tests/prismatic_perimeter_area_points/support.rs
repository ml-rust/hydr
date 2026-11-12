pub(crate) use super::point_support::{Client, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use solvr::common::arithmetic::Interval;

pub(crate) fn endpoints<R: Runtime<DType = DType>>(
    interval: &Interval<R>,
    count: usize,
) -> Result<Vec<[f64; 2]>> {
    super::point_support::interval(interval, &[count]);
    Ok(interval
        .lower()
        .try_to_vec::<f64>()?
        .into_iter()
        .zip(interval.upper().try_to_vec::<f64>()?)
        .map(|(lower, upper)| [lower, upper])
        .collect())
}

pub(crate) fn contact(actual: [f64; 2], truth: [u64; 2], allowance: u64, label: &str) {
    super::point_support::enclosed(actual, truth, label);
    let truth = truth.map(f64::from_bits);
    let error = (actual[0] - truth[1])
        .abs()
        .max((actual[1] - truth[0]).abs());
    assert!(
        error <= f64::from_bits(allowance),
        "{label}: {actual:?}, error={error}"
    );
}
