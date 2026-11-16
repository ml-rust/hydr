use super::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    boundary::{
        EndpointInterior,
        assembly::{is_scalar, numerical_range},
    },
    context::Context,
    geometry::TensorGeometry,
};
use crate::numerics::{TensorFaceState, homogeneous_wave_speed_tensor};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

pub(super) struct CflSpeeds<R: Runtime<DType = DType>> {
    pub outer: Tensor<R>,
    pub internal: Option<Tensor<R>>,
}

pub(super) struct ReconstructedFaces<R: Runtime<DType = DType>> {
    pub left: TensorFaceState<R>,
    pub right: TensorFaceState<R>,
    pub left_depth: Tensor<R>,
    pub right_depth: Tensor<R>,
    pub endpoints: EndpointInterior<R>,
    pub left_width: Tensor<R>,
    pub right_width: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    pub fn reconstruct(
        &self,
        fields: &Fields<R>,
        left_boundary: &ChannelTensorBoundary<R>,
        right_boundary: &ChannelTensorBoundary<R>,
    ) -> Result<ReconstructedFaces<R>, ChannelError> {
        let TensorGeometry::Rectangular(section) = &self.geometry else {
            return Err(ChannelError::numerical(
                "rectangular reconstruction requires rectangular geometry: select the triangular face method",
            ));
        };
        let width = &section.width;
        let c = &self.client;
        let count = self.count();
        let left_h = self.exterior(&fields.depth, left_boundary, 0, false)?;
        let right_h = self.exterior(&fields.depth, right_boundary, count - 1, false)?;
        let left_u = self.exterior(&fields.velocity, left_boundary, 0, true)?;
        let right_u = self.exterior(&fields.velocity, right_boundary, count - 1, true)?;
        let left_depth = c.cat(&[&left_h, &fields.depth], 0)?;
        let right_depth = c.cat(&[&fields.depth, &right_h], 0)?;
        let left_velocity = c.cat(&[&left_u, &fields.velocity], 0)?;
        let right_velocity = c.cat(&[&fields.velocity, &right_u], 0)?;
        let left_bed = c.cat(&[&self.bed.narrow(0, 0, 1)?, &self.bed], 0)?;
        let right_bed = c.cat(&[&self.bed, &self.bed.narrow(0, count - 1, 1)?], 0)?;
        let left_width = c.cat(&[&width.narrow(0, 0, 1)?, width], 0)?;
        let right_width = c.cat(&[width, &width.narrow(0, count - 1, 1)?], 0)?;
        let common_bed = c.maximum(&left_bed, &right_bed)?;
        let zero = self.constant(&left_depth, 0.0)?;
        let left_step = c.sub(&common_bed, &left_bed)?;
        let right_step = c.sub(&common_bed, &right_bed)?;
        self.finite(&left_step, "left hydrostatic bed step")?;
        self.finite(&right_step, "right hydrostatic bed step")?;
        let left_h = c.maximum(&c.sub(&left_depth, &left_step)?, &zero)?;
        let right_h = c.maximum(&c.sub(&right_depth, &right_step)?, &zero)?;
        self.finite(&left_h, "left reconstructed depth")?;
        self.finite(&right_h, "right reconstructed depth")?;
        let left_discharge = self.reconstructed_unit_discharge(
            &left_h,
            &left_velocity,
            left_boundary,
            right_boundary,
            "left reconstructed unit discharge",
        )?;
        let right_discharge = self.reconstructed_unit_discharge(
            &right_h,
            &right_velocity,
            left_boundary,
            right_boundary,
            "right reconstructed unit discharge",
        )?;
        let left = TensorFaceState {
            q_normal: left_discharge,
            q_tangent: zero.clone(),
            h: left_h.clone(),
            bed: common_bed.clone(),
        };
        let right = TensorFaceState {
            q_normal: right_discharge,
            q_tangent: zero,
            h: right_h.clone(),
            bed: common_bed,
        };
        let endpoints = EndpointInterior {
            area: self.endpoint_pair(&fields.area, &fields.area)?,
            discharge: self.endpoint_pair(&fields.discharge, &fields.discharge)?,
            depth: self.endpoint_pair(&right.h, &left.h)?,
            velocity: self.endpoint_pair(&fields.velocity, &fields.velocity)?,
        };
        Ok(ReconstructedFaces {
            endpoints,
            left,
            right,
            left_depth,
            right_depth,
            left_width,
            right_width,
        })
    }

    fn reconstructed_unit_discharge(
        &self,
        depth: &Tensor<R>,
        velocity: &Tensor<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
        name: &str,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let (start, len) = numerical_range(self.count(), left, right);
        let mut pieces = Vec::with_capacity(3);
        if is_scalar(left) {
            pieces.push(self.constant(&depth.narrow(0, 0, 1)?, 0.0)?);
        }
        if len > 0 {
            let depth = depth.narrow(0, start, len)?;
            let velocity = velocity.narrow(0, start, len)?;
            let discharge = c.mul(&depth, &velocity)?;
            let zero = self.constant(&depth, 0.0)?;
            let required = c.mul(&c.gt(&depth, &zero)?, &c.gt(&c.abs(&velocity)?, &zero)?)?;
            self.represented(&required, &discharge, name)?;
            pieces.push(discharge);
        }
        if is_scalar(right) {
            pieces.push(self.constant(&depth.narrow(0, self.count(), 1)?, 0.0)?);
        }
        if pieces.len() == 1 {
            return pieces.pop().ok_or_else(|| {
                ChannelError::numerical(
                    "reconstructed discharge array absent: inspect numerical face assembly",
                )
            });
        }
        let refs: Vec<_> = pieces.iter().collect();
        Ok(c.cat(&refs, 0)?)
    }

    fn exterior(
        &self,
        field: &Tensor<R>,
        boundary: &ChannelTensorBoundary<R>,
        index: usize,
        velocity: bool,
    ) -> Result<Tensor<R>, ChannelError> {
        let endpoint = field.narrow(0, index, 1)?;
        match boundary {
            ChannelTensorBoundary::Stage { .. }
            | ChannelTensorBoundary::Discharge { .. }
            | ChannelTensorBoundary::QuiescentReservoir { .. } => Ok(endpoint),
            ChannelTensorBoundary::Wall => {
                if velocity {
                    Ok(self.client.neg(&endpoint)?)
                } else {
                    Ok(endpoint)
                }
            }
            ChannelTensorBoundary::Exterior {
                depth_m,
                velocity_m_per_s,
            } => Ok(if velocity {
                velocity_m_per_s.clone()
            } else {
                depth_m.clone()
            }),
        }
    }
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
        + UnaryOps<R>
        + TensorOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub fn candidate_speed(
        &self,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<CflSpeeds<R>, ChannelError> {
        if let TensorGeometry::Triangular(geometry) = &self.geometry {
            return self.triangular_speed(geometry, fields, left, right);
        }
        if let TensorGeometry::Surveyed(geometry) = &self.geometry {
            return self.surveyed_speed(geometry, fields, left, right);
        }
        let reconstructed = self.reconstruct(fields, left, right)?;
        if !is_scalar(left) && !is_scalar(right) {
            return Ok(CflSpeeds {
                outer: homogeneous_wave_speed_tensor(
                    &self.client,
                    &reconstructed.left,
                    &reconstructed.right,
                    self.definition.config.gravity,
                    &self.definition.config.wave_config(),
                )?,
                internal: None,
            });
        }
        let frame = self.scalar_boundary_frame(&reconstructed.endpoints, left, right)?;
        let (start, len) = numerical_range(self.count(), left, right);
        let numerical = if len > 0 {
            Some(homogeneous_wave_speed_tensor(
                &self.client,
                &narrow_face(&reconstructed.left, start, len)?,
                &narrow_face(&reconstructed.right, start, len)?,
                self.definition.config.gravity,
                &self.definition.config.wave_config(),
            )?)
        } else {
            None
        };
        Ok(CflSpeeds {
            outer: self.assemble_boundary_faces(
                frame.as_ref(),
                numerical.as_ref(),
                frame
                    .as_ref()
                    .map(|frame| &frame.traces.signal_speed_bound_m_per_s),
            )?,
            internal: None,
        })
    }
}

pub(super) fn narrow_face<R: Runtime<DType = DType>>(
    face: &TensorFaceState<R>,
    start: usize,
    len: usize,
) -> Result<TensorFaceState<R>, ChannelError> {
    Ok(TensorFaceState {
        h: face.h.narrow(0, start, len)?,
        q_normal: face.q_normal.narrow(0, start, len)?,
        q_tangent: face.q_tangent.narrow(0, start, len)?,
        bed: face.bed.narrow(0, start, len)?,
    })
}
