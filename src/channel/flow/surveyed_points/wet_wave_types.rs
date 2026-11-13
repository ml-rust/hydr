use super::{wave_curve_types::CurveValue, wet_star_types::WetStarDomain};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum WetWaveStatus {
    Certified = 0,
    Unavailable = 1,
    Unresolved = 2,
}

/// Branch masks mean the complete root domain cannot exclude that branch.
pub(super) struct WetWaveSide<R: Runtime<DType = DType>> {
    pub(super) shock_possible: Tensor<R>,
    pub(super) rarefaction_possible: Tensor<R>,
    pub(super) equality_possible: Tensor<R>,
    pub(super) shock_actual: Tensor<R>,
    pub(super) shock_conditional: Tensor<R>,
    pub(super) source_sound: CurveValue<R>,
    pub(super) shock_offset: CurveValue<R>,
    pub(super) shock_speed: CurveValue<R>,
    pub(super) head: CurveValue<R>,
    pub(super) tail: CurveValue<R>,
    pub(super) shock_stationary: Tensor<R>,
    pub(super) equality_stationary: Tensor<R>,
    pub(super) valid: Tensor<R>,
}

/// Zero-containing speeds preserve every adjacent candidate. No mask selects a trace or flux.
pub(super) struct WetWaveRegions<R: Runtime<DType = DType>> {
    pub(super) left_exterior: Tensor<R>,
    pub(super) left_fan: Tensor<R>,
    pub(super) star: Tensor<R>,
    pub(super) right_fan: Tensor<R>,
    pub(super) right_exterior: Tensor<R>,
}

pub(super) struct WetWaveDomain<R: Runtime<DType = DType>> {
    pub(super) star: WetStarDomain<R>,
    pub(super) endpoint_sound: [CurveValue<R>; 2],
    pub(super) sides: [WetWaveSide<R>; 2],
    pub(super) regions: WetWaveRegions<R>,
    pub(super) status: Tensor<R>,
}
