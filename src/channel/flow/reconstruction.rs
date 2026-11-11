use super::{
    ChannelBoundary, ChannelError, admission::Fields, context::Context, geometry::TensorGeometry,
};
use crate::numerics::{TensorFaceState, homogeneous_wave_speed_tensor};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

pub(super) struct CflSpeeds<R: Runtime<DType = DType>> {
    pub outer: Tensor<R>,
    pub internal: Option<Tensor<R>>,
}

pub(super) struct ReconstructedFaces<R: Runtime<DType = DType>> {
    pub left: TensorFaceState<R>,
    pub right: TensorFaceState<R>,
    pub left_depth: Tensor<R>,
    pub right_depth: Tensor<R>,
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
        left_boundary: &ChannelBoundary,
        right_boundary: &ChannelBoundary,
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
        let left_discharge = c.mul(&left_h, &left_velocity)?;
        let right_discharge = c.mul(&right_h, &right_velocity)?;
        for (depth, velocity, discharge, name) in [
            (
                &left_h,
                &left_velocity,
                &left_discharge,
                "left reconstructed unit discharge",
            ),
            (
                &right_h,
                &right_velocity,
                &right_discharge,
                "right reconstructed unit discharge",
            ),
        ] {
            let required = c.mul(&c.gt(depth, &zero)?, &c.gt(&c.abs(velocity)?, &zero)?)?;
            self.represented(&required, discharge, name)?;
        }
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
        Ok(ReconstructedFaces {
            left,
            right,
            left_depth,
            right_depth,
            left_width,
            right_width,
        })
    }

    fn exterior(
        &self,
        field: &Tensor<R>,
        boundary: &ChannelBoundary,
        index: usize,
        velocity: bool,
    ) -> Result<Tensor<R>, ChannelError> {
        let endpoint = field.narrow(0, index, 1)?;
        match boundary {
            ChannelBoundary::Wall => {
                if velocity {
                    Ok(self.client.neg(&endpoint)?)
                } else {
                    Ok(endpoint)
                }
            }
            ChannelBoundary::Exterior {
                depth_m,
                velocity_m_per_s,
            } => self.constant(
                &endpoint,
                if velocity {
                    *velocity_m_per_s
                } else {
                    *depth_m
                },
            ),
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
        + BatchedBisectionAlgorithms<R>,
{
    pub fn candidate_speed(
        &self,
        fields: &Fields<R>,
        left: &ChannelBoundary,
        right: &ChannelBoundary,
    ) -> Result<CflSpeeds<R>, ChannelError> {
        if let TensorGeometry::Triangular(geometry) = &self.geometry {
            return self.triangular_speed(geometry, fields, left, right);
        }
        let reconstructed = self.reconstruct(fields, left, right)?;
        Ok(CflSpeeds {
            outer: homogeneous_wave_speed_tensor(
                &self.client,
                &reconstructed.left,
                &reconstructed.right,
                self.definition.config.gravity,
                &self.definition.config.wave_config(),
            )?,
            internal: None,
        })
    }
}
