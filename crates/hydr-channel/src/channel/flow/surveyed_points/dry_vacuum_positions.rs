use super::dry_vacuum_types::{DryVacuumEdges, DryVacuumPossibilities, DryVacuumRegime};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    regime: &Tensor<R>,
    edges: &[DryVacuumEdges<R>; 2],
) -> HydraulicResult<DryVacuumPossibilities<R>>
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
    let zero = Tensor::zeros(regime.shape(), a.dtype(), c.device())?;
    let one = Tensor::full_scalar(regime.shape(), a.dtype(), 1., c.device())?;
    let is = |value: DryVacuumRegime| -> HydraulicResult<Tensor<R>> {
        Ok(c.where_cond(
            &c.eq(
                regime,
                &Tensor::full_scalar(&[], DType::I64, value as i64 as f64, c.device())?,
            )?,
            &one,
            &zero,
        )?)
    };
    let both = c.maximum(
        &is(DryVacuumRegime::Vacuum)?,
        &is(DryVacuumRegime::Touching)?,
    )?;
    let left = c.maximum(&is(DryVacuumRegime::LeftWet)?, &both)?;
    let right = c.maximum(&is(DryVacuumRegime::RightWet)?, &both)?;
    // Missing endpoints preserve alternatives and keep their required fan certificate pending.
    let possible = |value: &Tensor<R>, available: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        Ok(c.where_cond(available, value, &one)?)
    };
    let l = &edges[0];
    let r = &edges[1];
    let le = c.mul(
        &left,
        &possible(&c.ge(l.head.physical.upper(), &zero)?, &l.head.available)?,
    )?;
    let re = c.mul(
        &right,
        &possible(&c.le(r.head.physical.lower(), &zero)?, &r.head.available)?,
    )?;
    let lf = c.mul(
        &left,
        &c.mul(
            &possible(&c.le(l.head.physical.lower(), &zero)?, &l.head.available)?,
            &possible(&c.ge(l.front.physical.upper(), &zero)?, &l.front.available)?,
        )?,
    )?;
    let rf = c.mul(
        &right,
        &c.mul(
            &possible(&c.le(r.front.physical.lower(), &zero)?, &r.front.available)?,
            &possible(&c.ge(r.head.physical.upper(), &zero)?, &r.head.available)?,
        )?,
    )?;
    let ld = possible(&c.le(l.front.physical.lower(), &zero)?, &l.front.available)?;
    let rd = possible(&c.ge(r.front.physical.upper(), &zero)?, &r.front.available)?;
    let dry = c.maximum(
        &is(DryVacuumRegime::BothDry)?,
        &c.maximum(
            &c.mul(&is(DryVacuumRegime::LeftWet)?, &ld)?,
            &c.maximum(
                &c.mul(&is(DryVacuumRegime::RightWet)?, &rd)?,
                &c.mul(&both, &c.mul(&ld, &rd)?)?,
            )?,
        )?,
    )?;
    Ok(DryVacuumPossibilities {
        left_exterior: le,
        left_fan: lf,
        dry,
        right_fan: rf,
        right_exterior: re,
    })
}
