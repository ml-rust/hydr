use super::sampling::SampledSolution;
use super::types::CapturedInput;
use super::work::PrismaticRiemannWork;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::Interval;

/// Certified fluxes and admissible traces in original query order.
///
/// Keep tensor storage and endpoint aliases immutable, including aliases returned by tensor cloning.
#[derive(Debug)]
pub struct PrismaticRiemannResult<R: Runtime<DType = DType>> {
    owners: Tensor<R>,
    left_area: Tensor<R>,
    left_discharge: Tensor<R>,
    right_area: Tensor<R>,
    right_discharge: Tensor<R>,
    trace_dry_possible: Tensor<R>,
    trace_wet_possible: Tensor<R>,
    water_flux: Interval<R>,
    momentum_flux: Interval<R>,
    trace_area: Interval<R>,
    trace_discharge: Interval<R>,
    trace_depth: Interval<R>,
    wave_speed: Interval<R>,
    work: PrismaticRiemannWork,
}

impl<R: Runtime<DType = DType>> PrismaticRiemannResult<R> {
    pub(crate) fn from_solution(
        input: CapturedInput<R>,
        solution: SampledSolution<R>,
        work: PrismaticRiemannWork,
    ) -> Self {
        Self {
            owners: input.owners,
            left_area: input.left_area,
            left_discharge: input.left_discharge,
            right_area: input.right_area,
            right_discharge: input.right_discharge,
            water_flux: solution.water_flux,
            momentum_flux: solution.momentum_flux,
            trace_area: solution.trace_area,
            trace_discharge: solution.trace_discharge,
            trace_depth: solution.trace_depth,
            wave_speed: solution.wave_speed,
            trace_dry_possible: solution.trace_dry_possible,
            trace_wet_possible: solution.trace_wet_possible,
            work,
        }
    }
    /// Immutable owners in original query order.
    pub fn owners(&self) -> &Tensor<R> {
        &self.owners
    }
    /// Immutable left area in original query order.
    pub fn left_area_m2(&self) -> &Tensor<R> {
        &self.left_area
    }
    /// Immutable left discharge in original query order.
    pub fn left_discharge_m3_per_s(&self) -> &Tensor<R> {
        &self.left_discharge
    }
    /// Immutable right area in original query order.
    pub fn right_area_m2(&self) -> &Tensor<R> {
        &self.right_area
    }
    /// Immutable right discharge in original query order.
    pub fn right_discharge_m3_per_s(&self) -> &Tensor<R> {
        &self.right_discharge
    }
    /// Immutable water flux in original query order.
    pub fn water_flux_m3_per_s(&self) -> &Interval<R> {
        &self.water_flux
    }
    /// Immutable momentum flux in original query order.
    pub fn momentum_flux_m4_per_s2(&self) -> &Interval<R> {
        &self.momentum_flux
    }
    /// Immutable trace area in original query order.
    pub fn trace_area_m2(&self) -> &Interval<R> {
        &self.trace_area
    }
    /// Immutable trace discharge in original query order.
    pub fn trace_discharge_m3_per_s(&self) -> &Interval<R> {
        &self.trace_discharge
    }
    /// Immutable trace depth in original query order.
    pub fn trace_depth_m(&self) -> &Interval<R> {
        &self.trace_depth
    }
    /// Immutable wave speed in original query order.
    pub fn wave_speed_m_per_s(&self) -> &Interval<R> {
        &self.wave_speed
    }
    /// Immutable trace dry possible in original query order.
    pub fn trace_dry_possible(&self) -> &Tensor<R> {
        &self.trace_dry_possible
    }
    /// Immutable trace wet possible in original query order.
    pub fn trace_wet_possible(&self) -> &Tensor<R> {
        &self.trace_wet_possible
    }
    /// Executed work across every refinement level.
    pub fn work(&self) -> PrismaticRiemannWork {
        self.work
    }
}
