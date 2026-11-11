use crate::channel::{
    ChannelBoundary, ChannelLateralInput, ChannelTensorBoundary, ChannelTensorLateralInput,
};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Complete external forcing for one named reach, independent of record order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolReachForcing {
    /// Exact setup reach identity, present once per interval.
    pub reach_id: String,
    /// External left policy. Connected endpoints require `None`, and unconnected endpoints require `Some`.
    pub left: Option<ChannelBoundary>,
    /// External right policy. Connected endpoints require `None`, and unconnected endpoints require `Some`.
    pub right: Option<ChannelBoundary>,
    /// One explicit input per declared lateral binding, including zeros and original rate/volume variants.
    pub lateral_inputs: Vec<ChannelLateralInput>,
}

/// Complete held external policies and original lateral inputs over one common interval.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolForcingInterval {
    /// Exact common accepted start time, in seconds.
    pub start_s: f64,
    /// End of forcing support, in seconds. Integrated volumes belong to this exact interval.
    pub end_s: f64,
    /// Exactly one record per setup reach, in arbitrary record order.
    pub reaches: Vec<ChannelPoolReachForcing>,
}

/// Complete native external operands for one reach on the caller-selected runtime and device.
#[derive(Clone, Debug)]
pub struct ChannelPoolTensorReachForcing<R: Runtime<DType = DType>> {
    /// Exact setup reach identity, present once per interval.
    pub reach_id: String,
    /// External left policy. Connected endpoints require `None`, and unconnected endpoints require `Some`.
    pub left: Option<ChannelTensorBoundary<R>>,
    /// External right policy. Connected endpoints require `None`, and unconnected endpoints require `Some`.
    pub right: Option<ChannelTensorBoundary<R>>,
    /// Complete native lateral inputs, preserving original rate/volume variants and SI units.
    pub lateral_inputs: Vec<ChannelTensorLateralInput<R>>,
}

/// Native forcing copied once during binding before immutable common-interval trials.
#[derive(Clone, Debug)]
pub struct ChannelPoolTensorForcingInterval<R: Runtime<DType = DType>> {
    /// Exact common accepted start time, in seconds.
    pub start_s: f64,
    /// End of forcing support, in seconds. Integrated volumes require complete recomputation after shortening.
    pub end_s: f64,
    /// Exactly one record per setup reach, in arbitrary record order.
    pub reaches: Vec<ChannelPoolTensorReachForcing<R>>,
}
