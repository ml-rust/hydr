use super::super::{ChannelLateralDirection, ChannelReachDefinition};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

/// Original nonnegative lateral rate or interval volume, using an F64 tensor of shape `[1]`.
#[derive(Clone, Debug)]
pub enum ChannelTensorWaterAmount<R: Runtime<DType = DType>> {
    /// Held water rate in cubic metres per second.
    Rate {
        /// Nonnegative rate in cubic metres per second.
        rate_m3_per_s: Tensor<R>,
    },
    /// Original water volume in cubic metres over the exact forcing interval.
    IntervalVolume {
        /// Nonnegative integrated volume in cubic metres.
        volume_m3: Tensor<R>,
    },
}

impl<R: Runtime<DType = DType>> ChannelTensorWaterAmount<R> {
    pub(in crate::channel::flow) fn tensor(&self) -> &Tensor<R> {
        match self {
            Self::Rate { rate_m3_per_s } => rate_m3_per_s,
            Self::IntervalVolume { volume_m3 } => volume_m3,
        }
    }
    pub(in crate::channel::flow) fn is_rate(&self) -> bool {
        matches!(self, Self::Rate { .. })
    }
}

/// Original lateral input tensors on the configured reach device.
#[derive(Clone, Debug)]
pub enum ChannelTensorLateralInput<R: Runtime<DType = DType>> {
    /// Water enters with the supplied longitudinal velocity in metres per second.
    Inflow {
        /// Exact declared lateral boundary identity.
        boundary_id: String,
        /// Original nonnegative rate or integrated volume.
        amount: ChannelTensorWaterAmount<R>,
        /// Incoming velocity along the increasing reach coordinate, in metres per second.
        longitudinal_velocity_m_per_s: Tensor<R>,
    },
    /// Water leaves at the accepted-start donor velocity.
    Withdrawal {
        /// Exact declared lateral boundary identity.
        boundary_id: String,
        /// Original nonnegative rate or integrated volume.
        amount: ChannelTensorWaterAmount<R>,
    },
}

impl<R: Runtime<DType = DType>> ChannelTensorLateralInput<R> {
    pub(in crate::channel::flow) fn boundary_id(&self) -> &str {
        match self {
            Self::Inflow { boundary_id, .. } | Self::Withdrawal { boundary_id, .. } => boundary_id,
        }
    }
    pub(in crate::channel::flow) fn amount(&self) -> &ChannelTensorWaterAmount<R> {
        match self {
            Self::Inflow { amount, .. } | Self::Withdrawal { amount, .. } => amount,
        }
    }
    pub(in crate::channel::flow) fn direction(&self) -> ChannelLateralDirection {
        match self {
            Self::Inflow { .. } => ChannelLateralDirection::Inflow,
            Self::Withdrawal { .. } => ChannelLateralDirection::Withdrawal,
        }
    }
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct BoundLateralInputs<R: Runtime<DType = DType>> {
    pub definition: Arc<ChannelReachDefinition>,
    pub original: Vec<ChannelTensorLateralInput<R>>,
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
    pub left: ChannelTensorBoundary<R>,
    pub right: ChannelTensorBoundary<R>,
    pub lateral: BoundLateralInputs<R>,
}

/// Held endpoint policy with original F64 tensors of shape `[1]` on the reach device.
#[derive(Clone, Debug)]
pub enum ChannelTensorBoundary<R: Runtime<DType = DType>> {
    /// Impermeable endpoint reflecting longitudinal velocity.
    Wall,
    /// Exterior depth and longitudinal velocity, in metres and metres per second.
    Exterior {
        /// Exterior water depth above the endpoint cell bed, in metres.
        depth_m: Tensor<R>,
        /// Exterior velocity along the increasing reach coordinate, in metres per second.
        velocity_m_per_s: Tensor<R>,
    },
    /// Water elevation in metres in the reach datum.
    Stage {
        /// Water elevation in metres in the reach datum.
        elevation_m: Tensor<R>,
    },
    /// Quiescent reservoir with complete exit mixing and explicit entrance loss, held over the interval.
    ///
    /// The interior and delivered trace remain strictly wet and subcritical.
    /// Zero entrance loss selects an ideal entrance. This policy supplies no finite reservoir storage.
    QuiescentReservoir {
        /// Finite reservoir water elevation in metres in the reach datum.
        elevation_m: Tensor<R>,
        /// Finite nonnegative dimensionless entrance loss referenced to incoming channel velocity head.
        entrance_loss_coefficient: Tensor<R>,
    },
    /// Original longitudinal full-section discharge in cubic metres per second.
    Discharge {
        /// Original full-section discharge along the increasing reach coordinate, in cubic metres per second.
        discharge_m3_per_s: Tensor<R>,
    },
    /// Direct water and momentum rates from a flux provider, inserted at the end face.
    FluxPort(Box<super::ChannelTensorFluxPort<R>>),
}

impl<R: Runtime<DType = DType>> ChannelTensorBoundary<R> {
    /// True for a scalar trace policy, whose face state comes from a solved trace.
    pub(in crate::channel::flow) fn is_scalar(&self) -> bool {
        matches!(
            self,
            Self::Stage { .. } | Self::Discharge { .. } | Self::QuiescentReservoir { .. }
        )
    }

    /// True when the end face takes its rates from the policy, not from a face Riemann problem.
    pub(in crate::channel::flow) fn is_direct(&self) -> bool {
        self.is_scalar() || matches!(self, Self::FluxPort(_))
    }
}

/// Held tensor endpoint and lateral operands over one bounded interval.
#[derive(Clone, Debug)]
pub struct ChannelTensorForcingInterval<R: Runtime<DType = DType>> {
    /// Exact accepted start time in seconds.
    pub start_s: f64,
    /// End of forcing support in seconds.
    pub end_s: f64,
    /// Upstream endpoint policy.
    pub left: ChannelTensorBoundary<R>,
    /// Downstream endpoint policy.
    pub right: ChannelTensorBoundary<R>,
    /// One explicit input per declared lateral binding, including zeros.
    pub lateral_inputs: Vec<ChannelTensorLateralInput<R>>,
}
