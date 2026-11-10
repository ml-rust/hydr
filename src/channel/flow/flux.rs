use super::{
    ChannelBoundary, ChannelBoundaryInterval, ChannelError, admission::Fields, context::Context,
};
use crate::numerics::{TensorFaceState, homogeneous_hll_tensor};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

pub(super) struct Flux<R: Runtime<DType = DType>> {
    pub water: Tensor<R>,
    pub momentum: Tensor<R>,
    pub left_pressure: Tensor<R>,
    pub right_pressure: Tensor<R>,
    pub left_reconstructed_pressure: Tensor<R>,
    pub right_reconstructed_pressure: Tensor<R>,
    pub momentum_bound: Tensor<R>,
    pub left_pressure_bound: Tensor<R>,
    pub right_pressure_bound: Tensor<R>,
    pub left_reconstructed_pressure_bound: Tensor<R>,
    pub right_reconstructed_pressure_bound: Tensor<R>,
    pub speed: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub fn flux(
        &self,
        fields: &Fields<R>,
        forcing: &ChannelBoundaryInterval,
    ) -> Result<Flux<R>, ChannelError> {
        let c = &self.client;
        let count = self.count();
        let left_h = self.exterior(&fields.depth, &forcing.left, 0, false)?;
        let right_h = self.exterior(&fields.depth, &forcing.right, count - 1, false)?;
        let left_u = self.exterior(&fields.velocity, &forcing.left, 0, true)?;
        let right_u = self.exterior(&fields.velocity, &forcing.right, count - 1, true)?;
        let left_depth = c.cat(&[&left_h, &fields.depth], 0)?;
        let right_depth = c.cat(&[&fields.depth, &right_h], 0)?;
        let left_velocity = c.cat(&[&left_u, &fields.velocity], 0)?;
        let right_velocity = c.cat(&[&fields.velocity, &right_u], 0)?;
        let left_bed = c.cat(&[&self.bed.narrow(0, 0, 1)?, &self.bed], 0)?;
        let right_bed = c.cat(&[&self.bed, &self.bed.narrow(0, count - 1, 1)?], 0)?;
        let left_width = c.cat(&[&self.width.narrow(0, 0, 1)?, &self.width], 0)?;
        let right_width = c.cat(&[&self.width, &self.width.narrow(0, count - 1, 1)?], 0)?;
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
        let homogeneous = homogeneous_hll_tensor(
            c,
            &left,
            &right,
            self.definition.config.gravity,
            &self.definition.config.wave_config(),
        )?;
        let water = c.mul(&homogeneous.shared[0], &self.face_width)?;
        let momentum = c.mul(&homogeneous.shared[1], &self.face_width)?;
        let required = c.gt(
            &c.abs(&homogeneous.shared[0])?,
            &self.constant(&water, 0.0)?,
        )?;
        self.represented(&required, &water, "full-width shared water rate")?;
        let left_reconstructed = self.pressure(&left_h, &self.face_width)?;
        let right_reconstructed = self.pressure(&right_h, &self.face_width)?;
        let pressure_bound = c.maximum(
            &left_reconstructed.unit_bound,
            &right_reconstructed.unit_bound,
        )?;
        let momentum_bound = self.product_bound(
            &homogeneous.shared[1],
            &pressure_bound,
            &self.face_width,
            &momentum,
            "full-width shared momentum rate",
        )?;
        let left_original = self.pressure(&left_depth, &left_width)?;
        let right_original = self.pressure(&right_depth, &right_width)?;
        let left_pressure_bound =
            self.bound_add(&left_original.bound, &left_reconstructed.bound)?;
        let right_pressure_bound =
            self.bound_add(&right_original.bound, &right_reconstructed.bound)?;
        let left_reconstructed_pressure = left_reconstructed.value;
        let right_reconstructed_pressure = right_reconstructed.value;
        let left_pressure = c.sub(&left_original.value, &left_reconstructed_pressure)?;
        let right_pressure = c.sub(&right_original.value, &right_reconstructed_pressure)?;
        for (value, name) in [
            (&water, "shared water flux"),
            (&momentum, "shared momentum flux"),
            (&left_pressure, "left pressure correction"),
            (&right_pressure, "right pressure correction"),
        ] {
            self.finite(value, name)?;
        }
        for (boundary, index) in [(&forcing.left, 0), (&forcing.right, count)] {
            if matches!(boundary, ChannelBoundary::Wall)
                && water.narrow(0, index, 1)?.item::<f64>()? != 0.0
            {
                return Err(ChannelError::numerical(format!(
                    "wall face={index} has nonzero water flux: inspect reflected HLL arithmetic"
                )));
            }
        }
        Ok(Flux {
            water,
            momentum,
            left_pressure,
            right_pressure,
            left_reconstructed_pressure,
            right_reconstructed_pressure,
            momentum_bound,
            left_pressure_bound,
            right_pressure_bound,
            left_reconstructed_pressure_bound: left_reconstructed.bound,
            right_reconstructed_pressure_bound: right_reconstructed.bound,
            speed: homogeneous.speed,
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
