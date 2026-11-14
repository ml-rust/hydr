use super::point_root_session::{PointRootSession, invalid};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn admit<R>(
    a: &mut Arithmetic<'_, R>,
    session: &PointRootSession<'_, '_, R>,
    mask: &Tensor<R>,
) -> HydraulicResult<()>
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
    if mask.shape() != session.invocation_rows.shape()
        || mask.dtype() != a.dtype()
        || !mask.device().is_same(c.device())
    {
        return Err(invalid(
            "point candidate demand needs invocation-sized caller-device working-dtype metadata",
        ));
    }
    let zero = Tensor::zeros(&[], a.dtype(), c.device())?;
    let one = Tensor::full_scalar(&[], a.dtype(), 1., c.device())?;
    if c.nonzero(&c.mul(&c.ne(mask, &zero)?, &c.ne(mask, &one)?)?)?
        .numel()
        != 0
    {
        return Err(invalid(
            "point candidate demand contains a value outside exact zero or one",
        ));
    }
    Ok(())
}
