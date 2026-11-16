use super::error::riemann_error;
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary, admission::Fields, context::Context,
    geometry::SurveyedGeometry, reconstruction::CflSpeeds,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
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
        + CertifiedUpperBracketAlgorithms<R>,
{
    pub(in crate::channel::flow) fn surveyed_speed(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<CflSpeeds<R>, ChannelError> {
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
        Ok(CflSpeeds {
            outer: self.surveyed_wave_envelope(result.wave_speed_m_per_s())?,
            internal: None,
        })
    }
}
