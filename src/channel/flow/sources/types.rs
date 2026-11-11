use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct LateralState<R: Runtime<DType = DType>> {
    pub cumulative_volume: Tensor<R>,
    pub counter_roundoff_bound: Tensor<R>,
    pub cumulative_assembly_bound_m3: f64,
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct LateralReceipt<R: Runtime<DType = DType>> {
    pub volume: Tensor<R>,
    pub momentum: Tensor<R>,
    pub applied_velocity: Tensor<R>,
    pub velocity_defined: Tensor<R>,
    pub counter_addition_low: Tensor<R>,
    pub next_counter: Tensor<R>,
    pub next_counter_bound: Tensor<R>,
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct LateralSummary {
    pub incoming_m3: f64,
    pub withdrawal_m3: f64,
    pub physical_total_roundoff_bound_m3: f64,
    pub counter_addition_bound_m3: f64,
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct LateralTotals {
    pub incoming_m3: f64,
    pub withdrawal_m3: f64,
    pub counter_abs_bound_m3: f64,
    pub assembly_abs_bound_m3: f64,
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct PreparedLateral<R: Runtime<DType = DType>> {
    pub incoming_volume: Tensor<R>,
    pub withdrawal_volume: Tensor<R>,
    pub incoming_momentum: Tensor<R>,
    pub withdrawal_momentum: Tensor<R>,
    pub incoming_volume_bound: Tensor<R>,
    pub withdrawal_volume_bound: Tensor<R>,
    pub incoming_momentum_bound: Tensor<R>,
    pub withdrawal_momentum_bound: Tensor<R>,
    pub next_state: LateralState<R>,
    pub summary: LateralSummary,
    pub receipt: LateralReceipt<R>,
    pub has_nonzero_values: bool,
}

pub(in crate::channel::flow) struct SourceApplication<R: Runtime<DType = DType>> {
    pub volume: Tensor<R>,
    pub momentum: Tensor<R>,
    pub water_assembly_bound: Tensor<R>,
    pub momentum_assembly_bound: Tensor<R>,
    pub water_assembly_charge_m3: f64,
}
