use super::{ChannelBoundary, ChannelError, admission::Fields, context::Context};
use crate::numerics::homogeneous_hll_tensor;
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
        left_boundary: &ChannelBoundary,
        right_boundary: &ChannelBoundary,
    ) -> Result<Flux<R>, ChannelError> {
        let c = &self.client;
        let count = self.count();
        let reconstructed = self.reconstruct(fields, left_boundary, right_boundary)?;
        let left = reconstructed.left;
        let right = reconstructed.right;
        let left_h = &left.h;
        let right_h = &right.h;
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
        let left_reconstructed = self.pressure(left_h, &self.face_width)?;
        let right_reconstructed = self.pressure(right_h, &self.face_width)?;
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
        let left_original = self.pressure(&reconstructed.left_depth, &reconstructed.left_width)?;
        let right_original =
            self.pressure(&reconstructed.right_depth, &reconstructed.right_width)?;
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
        for (boundary, index) in [(left_boundary, 0), (right_boundary, count)] {
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
}
