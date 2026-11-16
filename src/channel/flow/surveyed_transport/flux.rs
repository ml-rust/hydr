use super::error::riemann_error;
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary, admission::Fields, context::Context, flux::Flux,
    geometry::SurveyedGeometry, riemann_diagnostics::RiemannDiagnosticState,
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
    common::arithmetic::Interval,
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
    pub(in crate::channel::flow) fn surveyed_flux(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Flux<R>, ChannelError> {
        let faces = self.surveyed_faces(geometry, fields, left, right)?;
        let result = geometry
            .section
            .riemann_flux_tensor(
                &geometry.face_owners,
                &faces.left_area,
                &faces.left_discharge,
                &faces.right_area,
                &faces.right_discharge,
                &geometry.riemann,
            )
            .map_err(riemann_error)?;
        let work = result.work();
        let (water, water_flux_width) =
            self.surveyed_nominal(result.water_flux_m3_per_s(), work)?;
        let water = self.surveyed_wall_water(water, result.water_flux_m3_per_s(), left, right)?;
        let (momentum, momentum_flux_width) =
            self.surveyed_nominal(result.momentum_flux_m4_per_s2(), work)?;
        let speed = self.surveyed_wave_envelope(result.wave_speed_m_per_s())?;
        let zero = self.constant(&water, 0.0)?;
        Ok(Flux {
            boundary_traces: None,
            prescribed_water: None,
            water,
            momentum,
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

    fn surveyed_wall_water(
        &self,
        mut water: Tensor<R>,
        bounds: &Interval<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        for (boundary, face) in [(left, 0), (right, self.count())] {
            if !matches!(boundary, ChannelTensorBoundary::Wall) {
                continue;
            }
            let zero = self.constant(&water.narrow(0, face, 1)?, 0.0)?;
            let contains_zero = c.mul(
                &c.le(&bounds.lower().narrow(0, face, 1)?, &zero)?,
                &c.ge(&bounds.upper().narrow(0, face, 1)?, &zero)?,
            )?;
            if !self.all(&contains_zero)? {
                return Err(ChannelError::numerical(format!(
                    "surveyed wall face={face} excludes zero water flux: inspect reflected Riemann arithmetic"
                )));
            }
            water = if face == 0 {
                c.cat(&[&zero, &water.narrow(0, 1, self.count())?], 0)?
            } else {
                c.cat(&[&water.narrow(0, 0, self.count())?, &zero], 0)?
            };
        }
        Ok(water)
    }
}
