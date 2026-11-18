use super::{
    super::{ChannelError, context::Context},
    faces::Face,
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

pub(super) struct Physical<R: Runtime<DType = DType>> {
    pub momentum: Tensor<R>,
    pub momentum_bound: Tensor<R>,
    #[cfg(test)]
    pub pressure: Tensor<R>,
    #[cfg(test)]
    pub pressure_bound: Tensor<R>,
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
    #[cfg(test)]
    pub(super) fn triangular_physical(
        &self,
        face: &Face<R>,
        shared_active: &Tensor<R>,
    ) -> Result<Physical<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(&face.area, 0.0)?;
        let shared_face = Face {
            area: c.where_cond(shared_active, &face.area, &zero)?,
            discharge: c.where_cond(shared_active, &face.discharge, &zero)?,
            depth: c.where_cond(shared_active, &face.depth, &zero)?,
            velocity: c.where_cond(shared_active, &face.velocity, &zero)?,
        };
        let mut physical = self.triangular_sample_physical(&shared_face, &zero, &zero)?;
        let (pressure, pressure_bound) = self.triangular_pressure_value(face, &zero)?;
        physical.pressure = pressure;
        physical.pressure_bound = pressure_bound;
        Ok(physical)
    }

    pub(in crate::channel::flow::triangular) fn triangular_sample_physical(
        &self,
        face: &Face<R>,
        area_bound: &Tensor<R>,
        discharge_bound: &Tensor<R>,
    ) -> Result<Physical<R>, ChannelError> {
        let c = &self.client;
        let (pressure, pressure_bound) = self.triangular_pressure_value(face, area_bound)?;
        let advective = c.mul(&face.discharge, &face.velocity)?;
        let advective_bound = self.product_bound(
            &face.discharge,
            discharge_bound,
            &face.velocity,
            &advective,
            "Godunov sampled advective momentum",
        )?;
        let momentum = c.add(&advective, &pressure)?;
        self.finite(&momentum, "Godunov sampled momentum flux")?;
        let momentum_bound = self.bound_add(&advective_bound, &pressure_bound)?;
        Ok(Physical {
            momentum,
            momentum_bound,
            #[cfg(test)]
            pressure,
            #[cfg(test)]
            pressure_bound,
        })
    }
    pub(in crate::channel::flow::triangular) fn triangular_pressure_value(
        &self,
        face: &Face<R>,
        area_bound: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.client;
        let zero = self.constant(&face.area, 0.0)?;
        let gravity = self.constant(&face.area, self.definition.config.gravity)?;
        let coefficient_tensor = c.div(&gravity, &self.constant(&face.area, 3.0)?)?;
        let coefficient_bound = c.where_cond(
            &c.eq(&coefficient_tensor, &zero)?,
            &self.constant(&face.area, f64::from_bits(1))?,
            &zero,
        )?;
        let ga = c.mul(&coefficient_tensor, &face.area)?;
        let ga_product_bound = self.product_bound(
            &coefficient_tensor,
            &coefficient_bound,
            &face.area,
            &ga,
            "triangular pressure coefficient-area product",
        )?;
        let area_coefficient_bound = self.bound_add(
            &self.bound_scale(area_bound, &coefficient_tensor)?,
            &self.bound_scale(area_bound, &coefficient_bound)?,
        )?;
        let ga_bound = self.bound_add(&ga_product_bound, &area_coefficient_bound)?;
        let pressure = c.mul(&ga, &face.depth)?;
        let pressure_bound = self.product_bound(
            &ga,
            &ga_bound,
            &face.depth,
            &pressure,
            "triangular hydrostatic pressure",
        )?;
        Ok((pressure, pressure_bound))
    }
}

#[cfg(test)]
mod tests {
    use super::super::faces::Face;
    use super::super::fixture::{TestResult, context};
    use crate::channel::flow::context::Context;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn pressure_charge_retains_joint_erased_area_and_coefficient_uncertainty() -> TestResult {
        let original = context()?;
        let mut definition = original.definition.as_ref().clone();
        definition.config.gravity = f64::from_bits(1);
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let ctx =
            Context::<CpuRuntime>::new(client, definition, original.original_initial.clone())?;
        // For k=1 and h=1e-200, exact positive area rounds to zero within one subnormal.
        let zero = ctx.tensor(&[0.0])?;
        let sample = Face {
            area: zero.clone(),
            discharge: zero.clone(),
            depth: ctx.tensor(&[1e-200])?,
            velocity: zero,
        };
        let area_bound = ctx.tensor(&[f64::from_bits(1)])?;
        let (pressure, bound) = ctx.triangular_pressure_value(&sample, &area_bound)?;
        assert_eq!(pressure.item::<f64>()?, 0.0);
        assert!(bound.item::<f64>()? > 0.0);
        Ok(())
    }
}
