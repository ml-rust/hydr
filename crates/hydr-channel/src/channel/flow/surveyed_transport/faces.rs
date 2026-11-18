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
mod tests {
    use crate::channel::flow::{
        ChannelReach, ChannelTensorBoundary,
        admission::Fields,
        conserved_point::ConservedPoint,
        geometry::TensorGeometry,
        surveyed_update::tests::{Client, TestResult, cpu, moving_reach},
    };
    use numr::dtype::DType;
    use numr::ops::ScalarOps;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;
    use solvr::common::arithmetic::CertifiedExpansion;

    fn bits<R: Runtime<DType = DType>>(value: &Tensor<R>) -> TestResult<Vec<u64>> {
        Ok(value
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect())
    }

    fn reach<R>(client: R::Client) -> TestResult<ChannelReach<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        moving_reach::<R>(client, &[20.0, 10.0, 15.0], &[4.0, -2.0, 3.0], &[0.0; 3])
    }

    /// Fields with nonzero normalized low parts on both points.
    fn two_part_fields<R>(reach: &ChannelReach<R>) -> TestResult<Fields<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let context = &reach.context;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        // A high part scaled by 2^-60 is exact and far below half an ulp.
        let low = |point: &ConservedPoint<R>| -> TestResult<ConservedPoint<R>> {
            Ok(ConservedPoint {
                high: point.high.clone(),
                low: context.client.mul_scalar(&point.high, 2f64.powi(-60))?,
            })
        };
        Ok(Fields {
            area_point: low(&fields.area_point)?,
            discharge_point: low(&fields.discharge_point)?,
            ..fields
        })
    }

    fn zero_radius<R: Runtime<DType = DType>>(value: &CertifiedExpansion<R>) -> TestResult {
        assert!(
            value
                .abs_error()
                .try_to_vec::<f64>()?
                .iter()
                .all(|r| r.to_bits() == 0)
        );
        Ok(())
    }

    fn interior_points_match_fields<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client)?;
        let context = &reach.context;
        let TensorGeometry::Surveyed(geometry) = &context.geometry else {
            return Err("surveyed geometry required".into());
        };
        let fields = two_part_fields(&reach)?;
        assert!(
            fields
                .area_point
                .low
                .try_to_vec::<f64>()?
                .iter()
                .all(|v| *v != 0.0)
        );
        let wall = ChannelTensorBoundary::Wall;
        let faces = context.surveyed_faces(geometry, &fields, &wall, &wall)?;
        let n = context.count();
        for (face, offset, point) in [
            (&faces.left_area, 1, &fields.area_point),
            (&faces.left_discharge, 1, &fields.discharge_point),
            (&faces.right_area, 0, &fields.area_point),
            (&faces.right_discharge, 0, &fields.discharge_point),
        ] {
            assert_eq!(
                bits(&face.high().narrow(0, offset, n)?)?,
                bits(&point.high)?
            );
            assert_eq!(bits(&face.low().narrow(0, offset, n)?)?, bits(&point.low)?);
            zero_radius(face)?;
        }
        Ok(())
    }

    fn wall_mirror_negates_both_parts<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client)?;
        let context = &reach.context;
        let TensorGeometry::Surveyed(geometry) = &context.geometry else {
            return Err("surveyed geometry required".into());
        };
        let fields = two_part_fields(&reach)?;
        let wall = ChannelTensorBoundary::Wall;
        let faces = context.surveyed_faces(geometry, &fields, &wall, &wall)?;
        let n = context.count();
        let one = |value: &Tensor<R>, index: usize| -> TestResult<u64> {
            Ok(value.narrow(0, index, 1)?.item::<f64>()?.to_bits())
        };
        let negated = |value: &Tensor<R>, index: usize| -> TestResult<u64> {
            Ok((-value.narrow(0, index, 1)?.item::<f64>()?).to_bits())
        };
        let (area, discharge) = (&fields.area_point, &fields.discharge_point);
        for (mirror_area, mirror_discharge, face, cell) in [
            (&faces.left_area, &faces.left_discharge, 0, 0),
            (&faces.right_area, &faces.right_discharge, n, n - 1),
        ] {
            assert_eq!(one(mirror_area.high(), face)?, one(&area.high, cell)?);
            assert_eq!(one(mirror_area.low(), face)?, one(&area.low, cell)?);
            assert_eq!(
                one(mirror_discharge.high(), face)?,
                negated(&discharge.high, cell)?
            );
            assert_eq!(
                one(mirror_discharge.low(), face)?,
                negated(&discharge.low, cell)?
            );
        }
        Ok(())
    }

    fn exterior_faces_have_zero_low_and_radius<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client)?;
        let context = &reach.context;
        let TensorGeometry::Surveyed(geometry) = &context.geometry else {
            return Err("surveyed geometry required".into());
        };
        let fields = two_part_fields(&reach)?;
        let exterior = |depth: f64, velocity: f64| -> TestResult<ChannelTensorBoundary<R>> {
            Ok(ChannelTensorBoundary::Exterior {
                depth_m: context.tensor(&[depth])?,
                velocity_m_per_s: context.tensor(&[velocity])?,
            })
        };
        let (left, right) = (exterior(1.5, 0.3)?, exterior(0.5, -0.7)?);
        let faces = context.surveyed_faces(geometry, &fields, &left, &right)?;
        let n = context.count();
        for (area, discharge, face, depth, velocity) in [
            (&faces.left_area, &faces.left_discharge, 0, 1.5_f64, 0.3_f64),
            (&faces.right_area, &faces.right_discharge, n, 0.5, -0.7),
        ] {
            // The width-2 rectangular survey gives A = 2 h exactly.
            let a = area.high().narrow(0, face, 1)?.item::<f64>()?;
            assert_eq!(a.to_bits(), (2.0 * depth).to_bits());
            let q = discharge.high().narrow(0, face, 1)?.item::<f64>()?;
            assert_eq!(q.to_bits(), (a * velocity).to_bits());
            for value in [area.low(), discharge.low()] {
                assert_eq!(value.narrow(0, face, 1)?.item::<f64>()?.to_bits(), 0);
            }
            zero_radius(area)?;
            zero_radius(discharge)?;
        }
        Ok(())
    }

    #[test]
    fn cpu_interior_face_points_equal_the_field_points() -> TestResult {
        interior_points_match_fields::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_wall_mirror_negates_both_discharge_parts() -> TestResult {
        wall_mirror_negates_both_parts::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_exterior_faces_have_zero_low_parts_and_radius() -> TestResult {
        exterior_faces_have_zero_low_and_radius::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_interior_face_points_equal_the_field_points() -> TestResult {
            interior_points_match_fields::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_wall_mirror_negates_both_discharge_parts() -> TestResult {
            wall_mirror_negates_both_parts::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_exterior_faces_have_zero_low_parts_and_radius() -> TestResult {
            exterior_faces_have_zero_low_and_radius::<CudaRuntime>(CudaClient::new(
                CudaDevice::new(0),
            )?)
        }
    }
}
