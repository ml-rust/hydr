use super::error::boundary_error;
use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    conserved_point::ConservedPoint,
    conserved_update::{exact_over, expanded},
    context::Context,
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
use solvr::{
    common::arithmetic::CertifiedExpansion, integrate::CertifiedSimpsonAlgorithms,
    optimize::roots::CertifiedBisectionAlgorithms,
};

/// Zero-radius two-part face states in face order.
pub(super) struct Faces<R: Runtime<DType = DType>> {
    pub left_area: CertifiedExpansion<R>,
    pub left_discharge: CertifiedExpansion<R>,
    pub right_area: CertifiedExpansion<R>,
    pub right_discharge: CertifiedExpansion<R>,
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
    /// Face states from the cell points. Exterior states are F64 with a `+0.0` low part.
    pub(super) fn surveyed_faces(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Faces<R>, ChannelError> {
        let (area, discharge) = self.surveyed_states(geometry, fields, left, right)?;
        let faces = self.count() + 1;
        let side = |point: &ConservedPoint<R>, start: usize| {
            Ok::<_, ChannelError>(ConservedPoint {
                high: point.high.narrow(0, start, faces)?,
                low: point.low.narrow(0, start, faces)?,
            })
        };
        self.surveyed_face_points(
            &side(&area, 0)?,
            &side(&discharge, 0)?,
            &side(&area, 1)?,
            &side(&discharge, 1)?,
        )
    }

    /// Area and discharge of the left exterior, every cell, then the right exterior.
    pub(super) fn surveyed_states(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<(ConservedPoint<R>, ConservedPoint<R>), ChannelError> {
        let (la, lq) = self.surveyed_exterior(geometry, fields, left, 0, 0)?;
        let (ra, rq) =
            self.surveyed_exterior(geometry, fields, right, self.count() - 1, self.count())?;
        let c = &self.client;
        let join =
            |first: &ConservedPoint<R>, cells: &ConservedPoint<R>, last: &ConservedPoint<R>| {
                Ok::<_, ChannelError>(ConservedPoint {
                    high: c.cat(&[&first.high, &cells.high, &last.high], 0)?,
                    low: c.cat(&[&first.low, &cells.low, &last.low], 0)?,
                })
            };
        Ok((
            join(&la, &fields.area_point, &ra)?,
            join(&lq, &fields.discharge_point, &rq)?,
        ))
    }

    /// Zero-radius certified face points from two-part face states.
    pub(super) fn surveyed_face_points(
        &self,
        left_area: &ConservedPoint<R>,
        left_discharge: &ConservedPoint<R>,
        right_area: &ConservedPoint<R>,
        right_discharge: &ConservedPoint<R>,
    ) -> Result<Faces<R>, ChannelError> {
        let c = &self.client;
        expanded(c, "surveyed face points", |a| {
            let zero = Tensor::zeros(left_area.high.shape(), DType::F64, c.device())?;
            Ok(Faces {
                left_area: exact_over(a, &left_area.high, &left_area.low, &zero)?,
                left_discharge: exact_over(a, &left_discharge.high, &left_discharge.low, &zero)?,
                right_area: exact_over(a, &right_area.high, &right_area.low, &zero)?,
                right_discharge: exact_over(a, &right_discharge.high, &right_discharge.low, &zero)?,
            })
        })
    }

    fn surveyed_exterior(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        boundary: &ChannelTensorBoundary<R>,
        cell: usize,
        face: usize,
    ) -> Result<(ConservedPoint<R>, ConservedPoint<R>), ChannelError> {
        let c = &self.client;
        let cell_part = |value: &Tensor<R>| value.narrow(0, cell, 1);
        match boundary {
            ChannelTensorBoundary::Wall => Ok((
                ConservedPoint {
                    high: cell_part(&fields.area_point.high)?,
                    low: cell_part(&fields.area_point.low)?,
                },
                // Negation is exact on both parts.
                ConservedPoint {
                    high: c.neg(&cell_part(&fields.discharge_point.high)?)?,
                    low: c.neg(&cell_part(&fields.discharge_point.low)?)?,
                },
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
                Ok((
                    ConservedPoint::from_high(&area)?,
                    ConservedPoint::from_high(&discharge)?,
                ))
            }
            _ => Err(ChannelError::input(format!(
                "surveyed boundary face={face} requires Wall or complete Exterior depth and velocity: replace scalar characteristic controls"
            ))),
        }
    }
}

#[cfg(test)]
#[path = "faces_tests.rs"]
mod tests;
