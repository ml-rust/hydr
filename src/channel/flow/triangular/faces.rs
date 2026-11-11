use super::super::{
    ChannelError, ChannelTensorBoundary, ChannelTriangularReconstruction, admission::Fields,
    boundary::EndpointInterior, context::Context, geometry::TriangleGeometry,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
    },
    runtime::Runtime,
    tensor::Tensor,
};

pub(super) struct Face<R: Runtime<DType = DType>> {
    pub area: Tensor<R>,
    pub discharge: Tensor<R>,
    pub depth: Tensor<R>,
    pub velocity: Tensor<R>,
}

pub(super) struct TriangularFaces<R: Runtime<DType = DType>> {
    pub left: Face<R>,
    pub right: Face<R>,
    pub internal_speed: Option<Tensor<R>>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub(super) fn triangular_faces(
        &self,
        geometry: &TriangleGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<TriangularFaces<R>, ChannelError> {
        let endpoints = self.triangular_endpoints(fields)?;
        let (la, lq) = self.triangular_exterior(
            geometry,
            &endpoints.left_area,
            &endpoints.left_discharge,
            left,
            0,
        )?;
        let (ra, rq) = self.triangular_exterior(
            geometry,
            &endpoints.right_area,
            &endpoints.right_discharge,
            right,
            self.count() - 1,
        )?;
        let c = &self.client;
        let make = |area: Tensor<R>, discharge: Tensor<R>| -> Result<Face<R>, ChannelError> {
            let zero = self.constant(&area, 0.0)?;
            let wet = c.gt(&area, &zero)?;
            let safe = c.where_cond(&wet, &area, &self.constant(&area, 1.0)?)?;
            let velocity = c.div(&discharge, &safe)?;
            let depth = c.div(&c.sqrt(&area)?, &geometry.root_coefficient)?;
            self.finite(&velocity, "triangular conserved face velocity")?;
            self.finite(&depth, "triangular face depth")?;
            self.represented(
                &c.gt(&c.abs(&discharge)?, &zero)?,
                &velocity,
                "triangular conserved face velocity",
            )?;
            Ok(Face {
                area,
                discharge,
                depth,
                velocity,
            })
        };
        let internal_speed = if self.definition.config.triangular_reconstruction
            == ChannelTriangularReconstruction::LimitedLinear
        {
            let cell_left = make(
                endpoints.left_area.clone(),
                endpoints.left_discharge.clone(),
            )?;
            let cell_right = make(
                endpoints.right_area.clone(),
                endpoints.right_discharge.clone(),
            )?;
            let (sl, sr) = self.triangular_bounds(geometry, &cell_left, &cell_right)?;
            Some(c.maximum(&c.abs(&sl)?, &c.abs(&sr)?)?)
        } else {
            None
        };
        Ok(TriangularFaces {
            left: make(
                c.cat(&[&la, &endpoints.right_area], 0)?,
                c.cat(&[&lq, &endpoints.right_discharge], 0)?,
            )?,
            right: make(
                c.cat(&[&endpoints.left_area, &ra], 0)?,
                c.cat(&[&endpoints.left_discharge, &rq], 0)?,
            )?,
            internal_speed,
        })
    }

    pub(super) fn triangular_endpoint_interior(
        &self,
        faces: &TriangularFaces<R>,
    ) -> Result<EndpointInterior<R>, ChannelError> {
        Ok(EndpointInterior {
            area: self.endpoint_pair(&faces.right.area, &faces.left.area)?,
            discharge: self.endpoint_pair(&faces.right.discharge, &faces.left.discharge)?,
            depth: self.endpoint_pair(&faces.right.depth, &faces.left.depth)?,
            velocity: self.endpoint_pair(&faces.right.velocity, &faces.left.velocity)?,
        })
    }

    fn triangular_exterior(
        &self,
        geometry: &TriangleGeometry<R>,
        area_field: &Tensor<R>,
        discharge_field: &Tensor<R>,
        boundary: &ChannelTensorBoundary<R>,
        index: usize,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let area = area_field.narrow(0, index, 1)?;
        let discharge = discharge_field.narrow(0, index, 1)?;
        match boundary {
            ChannelTensorBoundary::Stage { .. } | ChannelTensorBoundary::Discharge { .. } => {
                Ok((area, discharge))
            }
            ChannelTensorBoundary::Wall => Ok((area, self.client.neg(&discharge)?)),
            ChannelTensorBoundary::Exterior {
                depth_m,
                velocity_m_per_s,
            } => {
                let depth = depth_m;
                let kh = self.client.mul(&geometry.coefficient, depth)?;
                let area = self.client.mul(&kh, depth)?;
                let discharge = self.client.mul(&area, velocity_m_per_s)?;
                let zero = self.constant(&area, 0.0)?;
                self.represented(
                    &self.client.gt(depth, &zero)?,
                    &area,
                    "triangular exterior area",
                )?;
                let required = self.client.mul(
                    &self.client.gt(depth, &zero)?,
                    &self.client.gt(&self.client.abs(velocity_m_per_s)?, &zero)?,
                )?;
                self.represented(&required, &discharge, "triangular exterior discharge")?;
                Ok((area, discharge))
            }
        }
    }
}

pub(super) fn narrow_face<R: Runtime<DType = DType>>(
    face: &Face<R>,
    start: usize,
    len: usize,
) -> Result<Face<R>, ChannelError> {
    Ok(Face {
        area: face.area.narrow(0, start, len)?,
        discharge: face.discharge.narrow(0, start, len)?,
        depth: face.depth.narrow(0, start, len)?,
        velocity: face.velocity.narrow(0, start, len)?,
    })
}
