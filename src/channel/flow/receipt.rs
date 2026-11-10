use super::{ChannelBoundaryInterval, ChannelError, ChannelStepDiagnostics, ChannelWaterLedger};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(super) struct LastInterval<R: Runtime<DType = DType>> {
    pub previous_volume: Tensor<R>,
    pub previous_momentum: Tensor<R>,
    pub previous_ledger: ChannelWaterLedger,
    pub previous_time: f64,
    pub previous_steps: usize,
    pub forcing: ChannelBoundaryInterval,
    pub requested_end: f64,
    pub scalar: ChannelStepDiagnostics,
    pub face_volume: Tensor<R>,
    pub face_momentum: Tensor<R>,
    pub left_pressure: Tensor<R>,
    pub right_pressure: Tensor<R>,
    pub friction: Tensor<R>,
    pub momentum_underflow_bound: Tensor<R>,
    pub transfer_underflow_bound: Tensor<R>,
}

impl<R: Runtime<DType = DType>> LastInterval<R> {
    pub fn project(&self) -> Result<ChannelStepDiagnostics, ChannelError> {
        let mut result = self.scalar.clone();
        result.face_volume_m3 = self.face_volume.try_to_vec::<f64>()?;
        result.face_momentum_m4_per_s = self.face_momentum.try_to_vec::<f64>()?;
        result.left_pressure_impulse_m4_per_s = self.left_pressure.try_to_vec::<f64>()?;
        result.right_pressure_impulse_m4_per_s = self.right_pressure.try_to_vec::<f64>()?;
        result.friction_impulse_m4_per_s = self.friction.try_to_vec::<f64>()?;
        result.momentum_underflow_bound_m4_per_s =
            self.momentum_underflow_bound.try_to_vec::<f64>()?;
        result.face_transfer_underflow_bound_m3 =
            self.transfer_underflow_bound.try_to_vec::<f64>()?;
        Ok(result)
    }
}
