use crate::channel::flow::{ChannelError, output::ChannelFrictionGeometryDiagnostics};
use numr::dtype::DType;
use numr::ops::ReduceOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct FrictionGeometryState<R: Runtime<DType = DType>> {
    pub momentum_abs_bound: Tensor<R>,
}

impl<R: Runtime<DType = DType>> FrictionGeometryState<R>
where
    R::Client: ReduceOps<R>,
{
    pub fn project(
        &self,
        client: &R::Client,
    ) -> Result<ChannelFrictionGeometryDiagnostics, ChannelError> {
        Ok(ChannelFrictionGeometryDiagnostics {
            momentum_abs_bound_m4_per_s: self.momentum_abs_bound.try_to_vec::<f64>()?,
            max_momentum_abs_bound_m4_per_s: client
                .max(&self.momentum_abs_bound, &[0], false)?
                .item::<f64>()?,
        })
    }
}
