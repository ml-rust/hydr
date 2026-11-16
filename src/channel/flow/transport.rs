use super::{
    ChannelError,
    conserved_update::{ExpansionClient, Parts, surveyed_faces},
    context::Context,
    flux::Flux,
    geometry::TensorGeometry,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Shared face impulses and the cell transfers formed from them.
///
/// Surveyed faces hold two-part points. Other geometry keeps zero low parts and zero transfer bounds.
pub(super) struct Transport<R: Runtime<DType = DType>> {
    pub face_volume: Tensor<R>,
    pub face_volume_low: Tensor<R>,
    pub face_momentum: Tensor<R>,
    pub face_momentum_low: Tensor<R>,
    pub left_pressure: Tensor<R>,
    pub right_pressure: Tensor<R>,
    pub water_delta: Parts<R>,
    pub momentum_delta: Parts<R>,
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
        + UnaryOps<R>
        + ExpansionClient<R>,
{
    pub fn integrate_transport(
        &self,
        flux: &Flux<R>,
        dt: f64,
    ) -> Result<Transport<R>, ChannelError> {
        if matches!(&self.geometry, TensorGeometry::Surveyed(_)) {
            return self.surveyed_transport(flux, dt);
        }
        let c = &self.client;
        let count = self.count();
        let face_volume = c.mul_scalar(&flux.water, dt)?;
        self.check_prescribed(flux, &face_volume)?;
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
        let face_volume_low = self.constant(&face_volume, 0.0)?;
        let face_momentum_low = face_volume_low.clone();
        let cell_zero = self.constant(&self.length, 0.0)?;
        let zero_delta = |high: Tensor<R>| Parts {
            high,
            low: cell_zero.clone(),
            bound: cell_zero.clone(),
        };
        if matches!(&self.geometry, TensorGeometry::Triangular(_)) {
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
                face_volume_low,
                face_momentum,
                face_momentum_low,
                left_pressure,
                right_pressure,
                water_delta: zero_delta(water_delta),
                momentum_delta: zero_delta(momentum_delta),
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
            face_volume_low,
            face_momentum,
            face_momentum_low,
            left_pressure,
            right_pressure,
            water_delta: zero_delta(water_delta),
            momentum_delta: zero_delta(momentum_delta),
            momentum_underflow_bound,
            transfer_underflow_bound,
            max_momentum_underflow_bound,
        })
    }

    fn check_prescribed(
        &self,
        flux: &Flux<R>,
        face_volume: &Tensor<R>,
    ) -> Result<(), ChannelError> {
        if let Some(prescribed) = &flux.prescribed_water {
            let c = &self.client;
            let required = c.mul(
                prescribed,
                &c.gt(&c.abs(&flux.water)?, &self.constant(&flux.water, 0.0)?)?,
            )?;
            self.represented(&required, face_volume, "prescribed discharge face volume")?;
        }
        Ok(())
    }

    /// Integrate surveyed face rates as exact products and keep both parts of each face point.
    ///
    /// The product radius joins the detected underflow bound of its face.
    fn surveyed_transport(&self, flux: &Flux<R>, dt: f64) -> Result<Transport<R>, ChannelError> {
        let c = &self.client;
        let count = self.count();
        let faces = surveyed_faces(c, &flux.water, &flux.momentum, dt, count)?;
        let face_volume = faces.volume.high;
        self.check_prescribed(flux, &face_volume)?;
        let duration = self.constant(&flux.water, dt)?;
        let transfer_underflow_bound = self.bound_add(
            &self.product_bound(
                &flux.water,
                &flux.water_bound,
                &duration,
                &face_volume,
                "shared numerical face volume",
            )?,
            &faces.volume.bound,
        )?;
        let face_momentum_bound = self.bound_add(
            &self.product_bound(
                &flux.momentum,
                &flux.momentum_bound,
                &duration,
                &faces.momentum.high,
                "shared face momentum",
            )?,
            &faces.momentum.bound,
        )?;
        let momentum_underflow_bound = self.bound_add(
            &face_momentum_bound.narrow(0, 0, count)?,
            &face_momentum_bound.narrow(0, 1, count)?,
        )?;
        let max_momentum_underflow_bound = c
            .max(&momentum_underflow_bound, &[0], false)?
            .item::<f64>()?;
        // Surveyed faces carry no separate pressure corrections.
        let left_pressure = self.constant(&face_volume, 0.0)?;
        let right_pressure = left_pressure.clone();
        Ok(Transport {
            face_volume,
            face_volume_low: faces.volume.low,
            face_momentum: faces.momentum.high,
            face_momentum_low: faces.momentum.low,
            left_pressure,
            right_pressure,
            water_delta: faces.water_delta,
            momentum_delta: faces.momentum_delta,
            momentum_underflow_bound,
            transfer_underflow_bound,
            max_momentum_underflow_bound,
        })
    }
}
