use crate::channel::flow::surveyed::work::PrismaticRiemannWork;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Interface trace that supplies a face's selected point fluxes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub enum PrismaticPointTraceKind {
    /// The unchanged left state.
    LeftState = 0,
    /// A point inside the left rarefaction fan.
    LeftFan = 1,
    /// The star state between the two waves.
    Star = 2,
    /// A point inside the right rarefaction fan.
    RightFan = 3,
    /// The unchanged right state.
    RightState = 4,
    /// A dry trace with zero area and discharge.
    Dry = 5,
}

impl PrismaticPointTraceKind {
    /// Returns the kind for an I64 `trace_kind` code, or `None` for an unknown code.
    pub fn from_code(code: i64) -> Option<Self> {
        [
            Self::LeftState,
            Self::LeftFan,
            Self::Star,
            Self::RightFan,
            Self::RightState,
            Self::Dry,
        ]
        .into_iter()
        .find(|kind| *kind as i64 == code)
    }
}

/// Certified point fluxes and their selected traces in original query order.
///
/// Each face takes its selected fluxes and trace from one candidate.
/// Keep tensor storage and component aliases immutable, including aliases returned by tensor cloning.
#[derive(Debug)]
pub struct PrismaticPointRiemannResult<R: Runtime<DType = DType>> {
    pub(super) owners: Tensor<R>,
    pub(super) left_area: CertifiedExpansion<R>,
    pub(super) left_discharge: CertifiedExpansion<R>,
    pub(super) right_area: CertifiedExpansion<R>,
    pub(super) right_discharge: CertifiedExpansion<R>,
    pub(super) water_flux: CertifiedExpansion<R>,
    pub(super) momentum_flux: CertifiedExpansion<R>,
    pub(super) trace_area: CertifiedExpansion<R>,
    pub(super) trace_discharge: CertifiedExpansion<R>,
    pub(super) trace_kind: Tensor<R>,
    pub(super) physical_water_flux: Interval<R>,
    pub(super) physical_momentum_flux: Interval<R>,
    pub(super) water_flux_bound: Interval<R>,
    pub(super) momentum_flux_bound: Interval<R>,
    pub(super) wave_speed: Interval<R>,
    pub(super) work: PrismaticRiemannWork,
}

impl<R: Runtime<DType = DType>> PrismaticPointRiemannResult<R> {
    /// Immutable I64 owners in original query order.
    pub fn owners(&self) -> &Tensor<R> {
        &self.owners
    }
    /// Immutable captured left area point in original query order.
    pub fn left_area_m2(&self) -> &CertifiedExpansion<R> {
        &self.left_area
    }
    /// Immutable captured left discharge point in original query order.
    pub fn left_discharge_m3_per_s(&self) -> &CertifiedExpansion<R> {
        &self.left_discharge
    }
    /// Immutable captured right area point in original query order.
    pub fn right_area_m2(&self) -> &CertifiedExpansion<R> {
        &self.right_area
    }
    /// Immutable captured right discharge point in original query order.
    pub fn right_discharge_m3_per_s(&self) -> &CertifiedExpansion<R> {
        &self.right_discharge
    }
    /// Immutable selected water flux point, the selected trace discharge, in original query order.
    /// Its error scales with the face jump. `water_flux_bound_m3_per_s` holds it.
    pub fn water_flux_m3_per_s(&self) -> &CertifiedExpansion<R> {
        &self.water_flux
    }
    /// Immutable selected momentum flux point in original query order.
    /// Its error scales with the face jump. `momentum_flux_bound_m4_per_s2` holds it.
    pub fn momentum_flux_m4_per_s2(&self) -> &CertifiedExpansion<R> {
        &self.momentum_flux
    }
    /// Immutable selected trace area point in original query order.
    pub fn trace_area_m2(&self) -> &CertifiedExpansion<R> {
        &self.trace_area
    }
    /// Immutable selected trace discharge point in original query order.
    pub fn trace_discharge_m3_per_s(&self) -> &CertifiedExpansion<R> {
        &self.trace_discharge
    }
    /// Immutable I64 `PrismaticPointTraceKind` codes in original query order.
    pub fn trace_kind(&self) -> &Tensor<R> {
        &self.trace_kind
    }
    /// Immutable water flux hull over every possible trace in original query order.
    pub fn physical_water_flux_m3_per_s(&self) -> &Interval<R> {
        &self.physical_water_flux
    }
    /// Immutable momentum flux hull over every possible trace in original query order.
    pub fn physical_momentum_flux_m4_per_s2(&self) -> &Interval<R> {
        &self.physical_momentum_flux
    }
    /// Immutable accepted water flux bound in original query order.
    /// It holds the physical hull and the selected point with its arithmetic bounds.
    pub fn water_flux_bound_m3_per_s(&self) -> &Interval<R> {
        &self.water_flux_bound
    }
    /// Immutable accepted momentum flux bound in original query order.
    /// It holds the physical hull and the selected point with its arithmetic bounds.
    pub fn momentum_flux_bound_m4_per_s2(&self) -> &Interval<R> {
        &self.momentum_flux_bound
    }
    /// Immutable signal speeds in original query order: lower is left, upper is right.
    pub fn wave_speed_m_per_s(&self) -> &Interval<R> {
        &self.wave_speed
    }
    /// Executed work across every refinement round.
    pub fn work(&self) -> PrismaticRiemannWork {
        self.work
    }
}
