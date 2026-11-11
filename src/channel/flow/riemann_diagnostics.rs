use super::{ChannelError, ChannelRiemannDiagnostics};
use numr::dtype::DType;
use numr::ops::ReduceOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct RiemannDiagnosticState<R: Runtime<DType = DType>> {
    pub water_flux_width: Tensor<R>,
    pub momentum_flux_width: Tensor<R>,
    pub root_iterations: usize,
}

impl<R: Runtime<DType = DType>> RiemannDiagnosticState<R>
where
    R::Client: ReduceOps<R>,
{
    pub fn project(&self, client: &R::Client) -> Result<ChannelRiemannDiagnostics, ChannelError> {
        Ok(ChannelRiemannDiagnostics {
            water_flux_width_m3_per_s: self.water_flux_width.try_to_vec::<f64>()?,
            momentum_flux_width_m4_per_s2: self.momentum_flux_width.try_to_vec::<f64>()?,
            max_water_flux_width_m3_per_s: client
                .max(&self.water_flux_width, &[0], false)?
                .item::<f64>()?,
            max_momentum_flux_width_m4_per_s2: client
                .max(&self.momentum_flux_width, &[0], false)?
                .item::<f64>()?,
            root_iterations: self.root_iterations,
        })
    }
}
