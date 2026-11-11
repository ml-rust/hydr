use super::super::{ChannelBoundary, ChannelLateralDirection, ChannelReachDefinition};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(in crate::channel::flow) enum NativeWaterAmount<R: Runtime<DType = DType>> {
    Rate { rate_m3_per_s: Tensor<R> },
    IntervalVolume { volume_m3: Tensor<R> },
}

impl<R: Runtime<DType = DType>> NativeWaterAmount<R> {
    pub fn tensor(&self) -> &Tensor<R> {
        match self {
            Self::Rate { rate_m3_per_s } => rate_m3_per_s,
            Self::IntervalVolume { volume_m3 } => volume_m3,
        }
    }
    pub fn is_rate(&self) -> bool {
        matches!(self, Self::Rate { .. })
    }
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) enum NativeLateralInput<R: Runtime<DType = DType>> {
    Inflow {
        boundary_id: String,
        amount: NativeWaterAmount<R>,
        longitudinal_velocity_m_per_s: Tensor<R>,
    },
    Withdrawal {
        boundary_id: String,
        amount: NativeWaterAmount<R>,
    },
}

impl<R: Runtime<DType = DType>> NativeLateralInput<R> {
    pub fn boundary_id(&self) -> &str {
        match self {
            Self::Inflow { boundary_id, .. } | Self::Withdrawal { boundary_id, .. } => boundary_id,
        }
    }
    pub fn amount(&self) -> &NativeWaterAmount<R> {
        match self {
            Self::Inflow { amount, .. } | Self::Withdrawal { amount, .. } => amount,
        }
    }
    pub fn direction(&self) -> ChannelLateralDirection {
        match self {
            Self::Inflow { .. } => ChannelLateralDirection::Inflow,
            Self::Withdrawal { .. } => ChannelLateralDirection::Withdrawal,
        }
    }
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct BoundLateralInputs<R: Runtime<DType = DType>> {
    pub definition: Arc<ChannelReachDefinition>,
    pub original: Vec<NativeLateralInput<R>>,
    pub amounts: Tensor<R>,
    pub rate_mask: Tensor<R>,
    pub inflow_velocities: Tensor<R>,
    pub has_interval_volumes: bool,
    pub has_nonzero_values: bool,
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct BoundChannelForcing<R: Runtime<DType = DType>> {
    pub definition: Arc<ChannelReachDefinition>,
    pub start_s: f64,
    pub end_s: f64,
    pub left: ChannelBoundary,
    pub right: ChannelBoundary,
    pub lateral: BoundLateralInputs<R>,
}
