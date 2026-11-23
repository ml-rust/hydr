//! Shape view of the rows of a march, and the hull of band enclosures.

use super::ArcShape;
use crate::structures::numerics::culvert::outlet::trig::fill;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Shape view of the rows of a march: shape masks, `B / D`, the box bed term and the arc bands.
#[derive(Clone, Copy)]
pub(crate) struct Outline<'s, R: Runtime<DType = DType>> {
    /// 1 on circle rows.
    pub(crate) circular: &'s Tensor<R>,
    pub(crate) beta: &'s Bounds<R>,
    /// Bed term of the box perimeter `p = bed + 2 s`: `beta`, or `rho^(3/2) beta` above a floor.
    pub(crate) bed: &'s Bounds<R>,
    pub(crate) arcs: Option<&'s ArcShape<R>>,
}

impl<R: Runtime<DType = DType>> Outline<'_, R> {
    /// 1 on rows whose coordinate is an angle: circle and arc rows.
    pub(crate) fn angled(&self, c: &R::Client) -> numr::error::Result<Tensor<R>>
    where
        R::Client: BarrelClient<R>,
    {
        match self.arcs {
            None => Ok(self.circular.clone()),
            Some(arcs) => c.add(self.circular, &arcs.arced),
        }
    }

    /// A coordinate inside every row's domain.
    pub(crate) fn harmless(&self) -> numr::error::Result<Tensor<R>> {
        match self.arcs {
            None => fill(self.circular, 0.5),
            Some(arcs) => Ok(arcs.harmless.clone()),
        }
    }
}

/// Hull of the enclosures of the bands in `members`, per row.
///
/// Rows of no member keep the last band's enclosure.
pub(crate) fn hull<R>(
    a: &mut Arithmetic<'_, R>,
    members: &[Tensor<R>; 3],
    values: [&Interval<R>; 3],
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let [.., last] = values;
    let mut base_lower = last.lower().clone();
    let mut base_upper = last.upper().clone();
    for (member, value) in members.iter().zip(values) {
        base_lower = c.where_cond(member, value.lower(), &base_lower)?;
        base_upper = c.where_cond(member, value.upper(), &base_upper)?;
    }
    let mut lower = base_lower.clone();
    let mut upper = base_upper.clone();
    for (member, value) in members.iter().zip(values) {
        lower = c.minimum(&lower, &c.where_cond(member, value.lower(), &base_lower)?)?;
        upper = c.maximum(&upper, &c.where_cond(member, value.upper(), &base_upper)?)?;
    }
    a.bounds(&lower, &upper)
}
