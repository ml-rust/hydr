use super::wet_wave_types::{WetWaveRegions, WetWaveSide};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn evaluate<R>(
    a: &Arithmetic<'_, R>,
    sides: &[WetWaveSide<R>; 2],
) -> HydraulicResult<WetWaveRegions<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let (left_out, left_in, left_fan) = side(a, &sides[0], true)?;
    let (right_out, right_in, right_fan) = side(a, &sides[1], false)?;
    let c = a.client();
    Ok(WetWaveRegions {
        left_exterior: c.mul(&left_out, &right_in)?,
        left_fan: c.mul(&left_fan, &right_in)?,
        star: c.mul(&left_in, &right_in)?,
        right_fan: c.mul(&left_in, &right_fan)?,
        right_exterior: c.mul(&left_in, &right_out)?,
    })
}

fn side<R>(
    a: &Arithmetic<'_, R>,
    side: &WetWaveSide<R>,
    left: bool,
) -> HydraulicResult<(Tensor<R>, Tensor<R>, Tensor<R>)>
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
    let zero = Tensor::zeros(side.valid.shape(), a.dtype(), c.device())?;
    let s = &side.shock_speed.physical;
    let h = &side.head.physical;
    let t = &side.tail.physical;
    let shock = c.mul(&side.shock_possible, &side.shock_speed.available)?;
    let rare = c.mul(
        &side.rarefaction_possible,
        &c.mul(&side.head.available, &side.tail.available)?,
    )?;
    let equal = c.mul(&side.equality_possible, &side.head.available)?;
    let (shock_out, shock_in, rare_out, rare_in, fan, equal_out, equal_in) = if left {
        (
            c.ge(s.upper(), &zero)?,
            c.le(s.lower(), &zero)?,
            c.ge(h.upper(), &zero)?,
            c.le(t.lower(), &zero)?,
            c.mul(&c.le(h.lower(), &zero)?, &c.ge(t.upper(), &zero)?)?,
            c.ge(h.upper(), &zero)?,
            c.le(h.lower(), &zero)?,
        )
    } else {
        (
            c.le(s.lower(), &zero)?,
            c.ge(s.upper(), &zero)?,
            c.le(h.lower(), &zero)?,
            c.ge(t.upper(), &zero)?,
            c.mul(&c.le(t.lower(), &zero)?, &c.ge(h.upper(), &zero)?)?,
            c.le(h.lower(), &zero)?,
            c.ge(h.upper(), &zero)?,
        )
    };
    let outside = c.maximum(
        &c.mul(&shock, &shock_out)?,
        &c.maximum(&c.mul(&rare, &rare_out)?, &c.mul(&equal, &equal_out)?)?,
    )?;
    let inside = c.maximum(
        &c.mul(&shock, &shock_in)?,
        &c.maximum(&c.mul(&rare, &rare_in)?, &c.mul(&equal, &equal_in)?)?,
    )?;
    Ok((outside, inside, c.mul(&rare, &fan)?))
}
