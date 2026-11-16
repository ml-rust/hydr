use super::error::boundary_error;
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary, admission::Fields, context::Context,
    geometry::SurveyedGeometry,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, ShapeOps, TensorOps,
        TwoSumOps, UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{integrate::CertifiedSimpsonAlgorithms, optimize::roots::CertifiedBisectionAlgorithms};

pub(super) struct Faces<R: Runtime<DType = DType>> {
    pub left_area: Tensor<R>,
    pub left_discharge: Tensor<R>,
    pub right_area: Tensor<R>,
    pub right_discharge: Tensor<R>,
}

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
        + CertifiedBisectionAlgorithms<R>,
{
    pub(super) fn surveyed_faces(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Faces<R>, ChannelError> {
        let (la, lq) = self.surveyed_exterior(geometry, fields, left, 0, 0)?;
        let (ra, rq) =
            self.surveyed_exterior(geometry, fields, right, self.count() - 1, self.count())?;
        let c = &self.client;
        Ok(Faces {
            left_area: c.cat(&[&la, &fields.area], 0)?,
            left_discharge: c.cat(&[&lq, &fields.discharge], 0)?,
            right_area: c.cat(&[&fields.area, &ra], 0)?,
            right_discharge: c.cat(&[&fields.discharge, &rq], 0)?,
        })
    }

    fn surveyed_exterior(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        boundary: &ChannelTensorBoundary<R>,
        cell: usize,
        face: usize,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.client;
        match boundary {
            ChannelTensorBoundary::Wall => Ok((
                fields.area.narrow(0, cell, 1)?,
                c.neg(&fields.discharge.narrow(0, cell, 1)?)?,
            )),
            ChannelTensorBoundary::Exterior {
                depth_m,
                velocity_m_per_s,
            } => {
                let area = geometry
                    .exterior_area(depth_m)
                    .map_err(|e| boundary_error(e, face))?;
                let discharge = c.mul(&area, velocity_m_per_s)?;
                let zero = self.constant(&area, 0.0)?;
                let required = c.mul(
                    &c.gt(&area, &zero)?,
                    &c.gt(&c.abs(velocity_m_per_s)?, &zero)?,
                )?;
                self.represented(&required, &discharge, "surveyed exterior discharge")
                    .map_err(|e| boundary_error(e, face))?;
                Ok((area, discharge))
            }
            _ => Err(ChannelError::input(format!(
                "surveyed boundary face={face} requires Wall or complete Exterior depth and velocity: replace scalar characteristic controls"
            ))),
        }
    }
}
