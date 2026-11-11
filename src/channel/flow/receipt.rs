use super::{
    ChannelError, ChannelStepDiagnostics, ChannelWaterLedger,
    forcing::BoundChannelForcing,
    riemann_diagnostics::RiemannDiagnosticState,
    sources::{LateralContext, LateralReceipt, LateralState},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(super) struct LastInterval<R: Runtime<DType = DType>> {
    pub previous_volume: Tensor<R>,
    pub previous_momentum: Tensor<R>,
    pub previous_lateral: LateralState<R>,
    pub previous_ledger: ChannelWaterLedger,
    pub previous_time: f64,
    pub previous_steps: usize,
    pub forcing: BoundChannelForcing<R>,
    pub lateral_receipt: LateralReceipt<R>,
    pub lateral_momentum_bound: Tensor<R>,
    pub requested_end: f64,
    pub scalar: ChannelStepDiagnostics,
    pub riemann: Option<RiemannDiagnosticState<R>>,
    pub boundary_traces: Option<std::sync::Arc<super::boundary::types::NativeBoundaryTraces<R>>>,
    pub face_volume: Tensor<R>,
    pub face_momentum: Tensor<R>,
    pub left_pressure: Tensor<R>,
    pub right_pressure: Tensor<R>,
    pub friction: Tensor<R>,
    pub momentum_underflow_bound: Tensor<R>,
    pub transfer_underflow_bound: Tensor<R>,
}

impl<R: Runtime<DType = DType>> LastInterval<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub fn project(
        &self,
        client: &R::Client,
        lateral: &LateralContext<R>,
    ) -> Result<ChannelStepDiagnostics, ChannelError> {
        let mut result = self.scalar.clone();
        result.riemann = self
            .riemann
            .as_ref()
            .map(|state| state.project(client))
            .transpose()?;
        result.boundary_traces = match &self.boundary_traces {
            Some(traces) => traces.project()?,
            None => [None, None],
        };
        result.face_volume_m3 = self.face_volume.try_to_vec::<f64>()?;
        result.face_momentum_m4_per_s = self.face_momentum.try_to_vec::<f64>()?;
        result.left_pressure_impulse_m4_per_s = self.left_pressure.try_to_vec::<f64>()?;
        result.right_pressure_impulse_m4_per_s = self.right_pressure.try_to_vec::<f64>()?;
        result.friction_impulse_m4_per_s = self.friction.try_to_vec::<f64>()?;
        result.momentum_underflow_bound_m4_per_s =
            self.momentum_underflow_bound.try_to_vec::<f64>()?;
        result.face_transfer_underflow_bound_m3 =
            self.transfer_underflow_bound.try_to_vec::<f64>()?;
        result.lateral_momentum_abs_bound_m4_per_s =
            self.lateral_momentum_bound.try_to_vec::<f64>()?;
        result.lateral_receipts = lateral.project_receipts(
            client,
            &self.forcing.lateral,
            &self.previous_lateral,
            &self.lateral_receipt,
        )?;
        Ok(result)
    }
}
