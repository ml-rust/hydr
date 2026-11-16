use super::{ChannelError, context::Context, flux::Flux, geometry::TensorGeometry};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) struct Transport<R: Runtime<DType = DType>> {
    pub face_volume: Tensor<R>,
    pub face_momentum: Tensor<R>,
    pub left_pressure: Tensor<R>,
    pub right_pressure: Tensor<R>,
    pub water_delta: Tensor<R>,
    pub momentum_delta: Tensor<R>,
    pub momentum_underflow_bound: Tensor<R>,
    pub transfer_underflow_bound: Tensor<R>,
    pub max_momentum_underflow_bound: f64,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    pub fn integrate_transport(
        &self,
        flux: &Flux<R>,
        dt: f64,
    ) -> Result<Transport<R>, ChannelError> {
        let c = &self.client;
        let count = self.count();
        let face_volume = c.mul_scalar(&flux.water, dt)?;
        if let Some(prescribed) = &flux.prescribed_water {
            let required = c.mul(
                prescribed,
                &c.gt(&c.abs(&flux.water)?, &self.constant(&flux.water, 0.0)?)?,
            )?;
            self.represented(&required, &face_volume, "prescribed discharge face volume")?;
        }
        let transfer_underflow_bound = self.product_bound(
            &flux.water,
            &flux.water_bound,
            &self.constant(&flux.water, dt)?,
            &face_volume,
            "shared numerical face volume",
        )?;
        let (face_momentum, face_momentum_bound) = self.integrate_momentum(
            &flux.momentum,
            &flux.momentum_bound,
            dt,
            "shared face momentum",
        )?;
        if matches!(
            &self.geometry,
            TensorGeometry::Triangular(_) | TensorGeometry::Surveyed(_)
        ) {
            let left_pressure = self.constant(&face_momentum, 0.0)?;
            let right_pressure = left_pressure.clone();
            let water_delta = c.sub(
                &face_volume.narrow(0, 0, count)?,
                &face_volume.narrow(0, 1, count)?,
            )?;
            let momentum_delta = c.sub(
                &face_momentum.narrow(0, 0, count)?,
                &face_momentum.narrow(0, 1, count)?,
            )?;
            let momentum_underflow_bound = self.bound_add(
                &face_momentum_bound.narrow(0, 0, count)?,
                &face_momentum_bound.narrow(0, 1, count)?,
            )?;
            let max_momentum_underflow_bound = c
                .max(&momentum_underflow_bound, &[0], false)?
                .item::<f64>()?;
            return Ok(Transport {
                face_volume,
                face_momentum,
                left_pressure,
                right_pressure,
                water_delta,
                momentum_delta,
                momentum_underflow_bound,
                transfer_underflow_bound,
                max_momentum_underflow_bound,
            });
        }
        let (left_pressure, left_pressure_bound) = self.integrate_momentum(
            &flux.left_pressure,
            &flux.left_pressure_bound,
            dt,
            "left pressure impulse",
        )?;
        let (right_pressure, right_pressure_bound) = self.integrate_momentum(
            &flux.right_pressure,
            &flux.right_pressure_bound,
            dt,
            "right pressure impulse",
        )?;
        let (left_reconstructed_pressure, left_reconstructed_bound) = self.integrate_momentum(
            &flux.left_reconstructed_pressure,
            &flux.left_reconstructed_pressure_bound,
            dt,
            "left reconstructed pressure impulse",
        )?;
        let (right_reconstructed_pressure, right_reconstructed_bound) = self.integrate_momentum(
            &flux.right_reconstructed_pressure,
            &flux.right_reconstructed_pressure_bound,
            dt,
            "right reconstructed pressure impulse",
        )?;
        let left_bound = self.bound_add(
            &face_momentum_bound.narrow(0, 0, count)?,
            &right_pressure_bound.narrow(0, 0, count)?,
        )?;
        let left_bound =
            self.bound_add(&left_bound, &right_reconstructed_bound.narrow(0, 0, count)?)?;
        let right_bound = self.bound_add(
            &face_momentum_bound.narrow(0, 1, count)?,
            &left_pressure_bound.narrow(0, 1, count)?,
        )?;
        let right_bound =
            self.bound_add(&right_bound, &left_reconstructed_bound.narrow(0, 1, count)?)?;
        let momentum_underflow_bound = self.bound_add(&left_bound, &right_bound)?;
        let max_momentum_underflow_bound = c
            .max(&momentum_underflow_bound, &[0], false)?
            .item::<f64>()?;
        let water_delta = c.sub(
            &face_volume.narrow(0, 0, count)?,
            &face_volume.narrow(0, 1, count)?,
        )?;
        // Original cell pressure cancels between its two faces before rounded subtraction.
        let left_momentum = c.sub(
            &face_momentum.narrow(0, 0, count)?,
            &right_reconstructed_pressure.narrow(0, 0, count)?,
        )?;
        let right_momentum = c.sub(
            &face_momentum.narrow(0, 1, count)?,
            &left_reconstructed_pressure.narrow(0, 1, count)?,
        )?;
        let momentum_delta = c.sub(&left_momentum, &right_momentum)?;
        Ok(Transport {
            face_volume,
            face_momentum,
            left_pressure,
            right_pressure,
            water_delta,
            momentum_delta,
            momentum_underflow_bound,
            transfer_underflow_bound,
            max_momentum_underflow_bound,
        })
    }
}
