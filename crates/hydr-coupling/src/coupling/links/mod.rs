//! Link numerics shared by the coupling coordinators.
//!
//! Each function keeps the operation order of the coordinator it came from. Callers supply their
//! own rejection texts where the coordinators word them differently.

pub(crate) mod culvert_roots;
pub(crate) mod cutoffs;
pub(crate) mod cuts;
pub(crate) mod donors;
pub(crate) mod error;
pub(crate) mod pool_ends;
pub(crate) mod pool_heads;
pub(crate) mod pool_plan;

pub(crate) use error::LinkError;

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::Runtime;

/// Every client operation the shared link numerics use.
pub(crate) trait LinkOps<R: Runtime<DType = DType>>:
    BinaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + IndexingOps<R>
    + ReduceOps<R>
    + ScalarOps<R>
    + ShapeOps<R>
    + TwoSumOps<R>
    + UnaryOps<R>
{
}

impl<R: Runtime<DType = DType>, T> LinkOps<R> for T where
    T: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
{
}
