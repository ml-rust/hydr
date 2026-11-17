use super::config::PrismaticRiemannConfig;
use super::work::WorkLedger;
use crate::channel::section::{Constitutive, GatheredBands, HydraulicGeometry};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::Interval;

pub(crate) struct CapturedInput<R: Runtime<DType = DType>> {
    pub(crate) owners: Tensor<R>,
    pub(crate) left_area: Tensor<R>,
    pub(crate) left_discharge: Tensor<R>,
    pub(crate) right_area: Tensor<R>,
    pub(crate) right_discharge: Tensor<R>,
}

pub(crate) struct SideState<R: Runtime<DType = DType>> {
    pub(crate) original_area: Tensor<R>,
    pub(crate) original_discharge: Tensor<R>,
    pub(crate) geometry: Constitutive<R>,
    pub(crate) velocity: Interval<R>,
    pub(crate) sound: Interval<R>,
    pub(crate) pressure: Interval<R>,
    pub(crate) potential: Interval<R>,
    pub(crate) wet: Tensor<R>,
    pub(crate) dry: Tensor<R>,
}

#[derive(Clone, Copy)]
pub(crate) enum WaveSide {
    Left,
    Right,
}

pub(crate) struct Oracle<'a, R: Runtime<DType = DType>> {
    pub(crate) geometry: &'a HydraulicGeometry<R>,
    pub(crate) bands: &'a GatheredBands<R>,
    pub(crate) left: &'a SideState<R>,
    pub(crate) right: &'a SideState<R>,
    pub(crate) config: &'a PrismaticRiemannConfig,
    pub(crate) panels: usize,
    pub(crate) work: &'a WorkLedger,
}
