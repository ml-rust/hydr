//! The client operations of one common surface/channel interval.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};

/// Every client operation a common surface/channel interval uses.
pub(super) trait Ops<R: Runtime<DType = DType>>:
    BinaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + CorrectlyRoundedOps<R>
    + CumulativeOps<R>
    + HypotOps<R>
    + IndexingOps<R>
    + MidpointOps<R>
    + NextRepresentableOps<R>
    + ReduceOps<R>
    + ScalarOps<R>
    + ShapeOps<R>
    + TensorOps<R>
    + TwoSumOps<R>
    + TypeConversionOps<R>
    + UnaryOps<R>
    + BatchedBisectionAlgorithms<R>
    + BatchedNewtonAlgorithms<R>
    + Interp1dAlgorithms<R>
    + AndersonAlgorithms<R>
    + solvr::integrate::CertifiedSimpsonAlgorithms<R>
    + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
    + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
    + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>
{
}

impl<R: Runtime<DType = DType>, T> Ops<R> for T where
    T: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>
{
}
