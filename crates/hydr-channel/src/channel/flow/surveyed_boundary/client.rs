//! Client operations of certified surveyed trace solves.

use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::roots::{BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms},
};

/// Interval arithmetic, certified Simpson integration and certified bisection.
pub(in crate::channel::flow) trait TraceClient<R: Runtime<DType = DType>>:
    TensorOps<R>
    + CompareOps<R>
    + NextRepresentableOps<R>
    + MidpointOps<R>
    + ScalarOps<R>
    + TwoSumOps<R>
    + RuntimeClient<R>
    + CertifiedSimpsonAlgorithms<R>
    + CertifiedBisectionAlgorithms<R>
    + BatchedBisectionAlgorithms<R>
{
}

impl<R, C> TraceClient<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
}
