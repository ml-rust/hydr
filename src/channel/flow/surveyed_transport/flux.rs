use super::error::riemann_error;
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    conserved_update::{Parts, parts},
    context::Context,
    flux::{Flux, SurveyedRates},
    geometry::SurveyedGeometry,
    riemann_diagnostics::RiemannDiagnosticState,
    surveyed_points::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, ShapeOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
    },
};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    /// Solve every face from the two-part cell points.
    pub(super) fn surveyed_point_riemann(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<PrismaticPointRiemannResult<R>, ChannelError> {
        let faces = self.surveyed_faces(geometry, fields, left, right)?;
        geometry
            .section
            .riemann_flux_points_tensor(
                &geometry.face_owners,
                &faces.left_area,
                &faces.left_discharge,
                &faces.right_area,
                &faces.right_discharge,
                &geometry.riemann,
            )
            .map_err(riemann_error)
    }

    pub(in crate::channel::flow) fn surveyed_flux(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Flux<R>, ChannelError> {
        let result = self.surveyed_point_riemann(geometry, fields, left, right)?;
        let work = result.work();
        let water_flux_width = self.surveyed_width(result.water_flux_bound_m3_per_s())?;
        let momentum_flux_width = self.surveyed_width(result.momentum_flux_bound_m4_per_s2())?;
        let water = self.surveyed_wall_water(&result, left, right)?;
        let momentum = parts(result.momentum_flux_m4_per_s2());
        let speed = self.surveyed_wave_envelope(result.wave_speed_m_per_s())?;
        let zero = self.constant(&water.high, 0.0)?;
        Ok(Flux {
            boundary_traces: None,
            prescribed_water: None,
            water: water.high.clone(),
            momentum: momentum.high.clone(),
            surveyed: Some(SurveyedRates { water, momentum }),
            speed,
            internal_speed: None,
            water_bound: zero.clone(),
            momentum_bound: zero.clone(),
            riemann: Some(RiemannDiagnosticState {
                water_flux_width,
                momentum_flux_width,
                root_iterations: work.maximum_root_iterations,
                surveyed_work: Some(work),
            }),
            left_pressure: zero.clone(),
            right_pressure: zero.clone(),
            left_reconstructed_pressure: zero.clone(),
            right_reconstructed_pressure: zero.clone(),
            left_pressure_bound: zero.clone(),
            right_pressure_bound: zero.clone(),
            left_reconstructed_pressure_bound: zero.clone(),
            right_reconstructed_pressure_bound: zero,
        })
    }

    /// Selected water point with every wall component set to `+0.0`.
    ///
    /// A wall face must enclose zero water flux and select a star or dry trace.
    fn surveyed_wall_water(
        &self,
        result: &PrismaticPointRiemannResult<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Parts<R>, ChannelError> {
        let c = &self.client;
        let bounds = result.water_flux_bound_m3_per_s();
        let mut water = parts(result.water_flux_m3_per_s());
        for (boundary, face) in [(left, 0), (right, self.count())] {
            if !matches!(boundary, ChannelTensorBoundary::Wall) {
                continue;
            }
            let zero = self.constant(&water.high.narrow(0, face, 1)?, 0.0)?;
            let contains_zero = c.mul(
                &c.le(&bounds.lower().narrow(0, face, 1)?, &zero)?,
                &c.ge(&bounds.upper().narrow(0, face, 1)?, &zero)?,
            )?;
            if !self.all(&contains_zero)? {
                return Err(ChannelError::numerical(format!(
                    "surveyed wall face={face} excludes zero water flux: inspect reflected Riemann arithmetic"
                )));
            }
            let code = result.trace_kind().narrow(0, face, 1)?.item::<i64>()?;
            let kind = PrismaticPointTraceKind::from_code(code);
            if !matches!(
                kind,
                Some(PrismaticPointTraceKind::Star | PrismaticPointTraceKind::Dry)
            ) {
                return Err(ChannelError::numerical(format!(
                    "surveyed wall face={face} selects trace {kind:?} (code {code}), not a star or dry trace: inspect the reflected wall state"
                )));
            }
            water = Parts {
                high: self.replace_face(&water.high, face, &zero)?,
                low: self.replace_face(&water.low, face, &zero)?,
                bound: self.replace_face(&water.bound, face, &zero)?,
            };
        }
        Ok(water)
    }

    /// Copy of `value` with `face` set to the one-element `replacement`.
    fn replace_face(
        &self,
        value: &Tensor<R>,
        face: usize,
        replacement: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let faces = self.count() + 1;
        let mut pieces = Vec::with_capacity(3);
        if face > 0 {
            pieces.push(value.narrow(0, 0, face)?);
        }
        pieces.push(replacement.clone());
        if face + 1 < faces {
            pieces.push(value.narrow(0, face + 1, faces - face - 1)?);
        }
        let pieces: Vec<&Tensor<R>> = pieces.iter().collect();
        Ok(self.client.cat(&pieces, 0)?)
    }
}

#[cfg(test)]
#[path = "flux_tests.rs"]
mod tests;
