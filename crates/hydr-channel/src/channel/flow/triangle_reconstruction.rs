use super::{ChannelError, ChannelTriangularReconstruction, admission::Fields, context::Context};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps, UnaryOps,
    },
    runtime::Runtime,
    tensor::Tensor,
};

pub(super) struct TriangularEndpoints<R: Runtime<DType = DType>> {
    pub left_area: Tensor<R>,
    pub right_area: Tensor<R>,
    pub left_discharge: Tensor<R>,
    pub right_discharge: Tensor<R>,
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
    pub(super) fn triangular_endpoints(
        &self,
        fields: &Fields<R>,
    ) -> Result<TriangularEndpoints<R>, ChannelError> {
        let flat = || TriangularEndpoints {
            left_area: fields.area.clone(),
            right_area: fields.area.clone(),
            left_discharge: fields.discharge.clone(),
            right_discharge: fields.discharge.clone(),
        };
        if self.definition.config.triangular_reconstruction
            == ChannelTriangularReconstruction::FirstOrderReference
            || self.count() < 3
        {
            return Ok(flat());
        }
        let c = &self.client;
        let zero = self.constant(&fields.area, 0.0)?;
        let one = self.constant(&fields.area, 1.0)?;
        let (da, area_resolved) = self.triangular_limited_increment(&fields.area)?;
        let (dq, discharge_resolved) = self.triangular_limited_increment(&fields.discharge)?;
        let wet = c.gt(&fields.area, &zero)?;
        let denominator = c.maximum(&fields.area, &c.abs(&da)?)?;
        let safe = c.where_cond(&wet, &denominator, &one)?;
        let theta = c.minimum(&c.div(&fields.area, &safe)?, &one)?;
        let scaled_a = c.mul(&theta, &da)?;
        let scaled_q = c.mul(&theta, &dq)?;
        let mut admitted = c.mul(&area_resolved, &discharge_resolved)?;
        for (increment, scaled) in [(&da, &scaled_a), (&dq, &scaled_q)] {
            let required = c.mul(&c.gt(&c.abs(increment)?, &zero)?, &c.gt(&theta, &zero)?)?;
            admitted = c.mul(
                &admitted,
                &c.maximum(&c.eq(&required, &zero)?, &c.gt(&c.abs(scaled)?, &zero)?)?,
            )?;
        }
        let left_area = c.sub(&fields.area, &scaled_a)?;
        let right_area = c.add(&fields.area, &scaled_a)?;
        let left_discharge = c.sub(&fields.discharge, &scaled_q)?;
        let right_discharge = c.add(&fields.discharge, &scaled_q)?;
        for (area, discharge) in [
            (&left_area, &left_discharge),
            (&right_area, &right_discharge),
        ] {
            admitted = c.mul(
                &admitted,
                &self.triangular_endpoint_admitted(area, discharge, &wet)?,
            )?;
        }
        admitted = c.mul(
            &admitted,
            &self.triangular_endpoint_mean(&fields.area, &left_area, &right_area)?,
        )?;
        admitted = c.mul(
            &admitted,
            &self.triangular_endpoint_mean(&fields.discharge, &left_discharge, &right_discharge)?,
        )?;
        // Both conserved slopes reset together. Stored cell means remain authoritative.
        Ok(TriangularEndpoints {
            left_area: c.where_cond(&admitted, &left_area, &fields.area)?,
            right_area: c.where_cond(&admitted, &right_area, &fields.area)?,
            left_discharge: c.where_cond(&admitted, &left_discharge, &fields.discharge)?,
            right_discharge: c.where_cond(&admitted, &right_discharge, &fields.discharge)?,
        })
    }

    fn triangular_limited_increment(
        &self,
        value: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.client;
        let n = self.count() - 2;
        let center = value.narrow(0, 1, n)?;
        let zero = self.constant(&center, 0.0)?;
        let left_distance = c.add(
            &c.mul_scalar(&self.length.narrow(0, 0, n)?, 0.5)?,
            &c.mul_scalar(&self.length.narrow(0, 1, n)?, 0.5)?,
        )?;
        let right_distance = c.add(
            &c.mul_scalar(&self.length.narrow(0, 1, n)?, 0.5)?,
            &c.mul_scalar(&self.length.narrow(0, 2, n)?, 0.5)?,
        )?;
        let left_difference = c.sub(&center, &value.narrow(0, 0, n)?)?;
        let right_difference = c.sub(&value.narrow(0, 2, n)?, &center)?;
        let secant = |difference: &Tensor<R>,
                      distance: &Tensor<R>|
         -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
            let positive = c.gt(distance, &zero)?;
            let safe = c.where_cond(&positive, distance, &self.constant(&center, 1.0)?)?;
            let slope = c.div(difference, &safe)?;
            let mut admitted = positive;
            for operand in [difference, distance, &slope] {
                admitted = c.mul(&admitted, &self.triangular_finite_mask(operand)?)?;
            }
            admitted = c.mul(
                &admitted,
                &c.maximum(&c.eq(difference, &zero)?, &c.gt(&c.abs(&slope)?, &zero)?)?,
            )?;
            Ok((slope, admitted))
        };
        let (left, left_admitted) = secant(&left_difference, &left_distance)?;
        let (right, right_admitted) = secant(&right_difference, &right_distance)?;
        let mut admitted = c.mul(&left_admitted, &right_admitted)?;
        let same_sign = c.maximum(
            &c.mul(&c.gt(&left, &zero)?, &c.gt(&right, &zero)?)?,
            &c.mul(&c.lt(&left, &zero)?, &c.lt(&right, &zero)?)?,
        )?;
        let magnitude = c.minimum(&c.abs(&left)?, &c.abs(&right)?)?;
        let signed = c.where_cond(&c.gt(&left, &zero)?, &magnitude, &c.neg(&magnitude)?)?;
        let slope = c.where_cond(&same_sign, &signed, &zero)?;
        let half_length = c.mul_scalar(&self.length.narrow(0, 1, n)?, 0.5)?;
        let increment = c.mul(&slope, &half_length)?;
        admitted = c.mul(&admitted, &self.triangular_finite_mask(&increment)?)?;
        admitted = c.mul(
            &admitted,
            &c.maximum(&c.eq(&slope, &zero)?, &c.gt(&c.abs(&increment)?, &zero)?)?,
        )?;
        let endpoint_zero = self.constant(&value.narrow(0, 0, 1)?, 0.0)?;
        let endpoint_one = self.constant(&endpoint_zero, 1.0)?;
        Ok((
            c.cat(&[&endpoint_zero, &increment, &endpoint_zero], 0)?,
            c.cat(&[&endpoint_one, &admitted, &endpoint_one], 0)?,
        ))
    }

    fn triangular_finite_mask(&self, value: &Tensor<R>) -> Result<Tensor<R>, ChannelError> {
        Ok(self
            .client
            .le(&self.client.abs(value)?, &self.constant(value, f64::MAX)?)?)
    }

    fn triangular_endpoint_admitted(
        &self,
        area: &Tensor<R>,
        discharge: &Tensor<R>,
        wet_cell: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(area, 0.0)?;
        let wet = c.gt(area, &zero)?;
        let safe_area = c.where_cond(&wet, area, &self.constant(area, 1.0)?)?;
        let velocity = c.div(discharge, &safe_area)?;
        let super::geometry::TensorGeometry::Triangular(geometry) = &self.geometry else {
            return Err(ChannelError::input(
                "triangular endpoint admission requires triangular geometry: select the matching face method",
            ));
        };
        let depth = c.div(
            &c.sqrt(&c.where_cond(&wet, area, &zero)?)?,
            &geometry.root_coefficient,
        )?;
        let mut admitted = c.maximum(
            &c.mul(wet_cell, &wet)?,
            &c.mul(
                &c.eq(wet_cell, &zero)?,
                &c.mul(&c.eq(area, &zero)?, &c.eq(discharge, &zero)?)?,
            )?,
        )?;
        for value in [area, discharge, &velocity, &depth] {
            admitted = c.mul(&admitted, &self.triangular_finite_mask(value)?)?;
        }
        admitted = c.mul(
            &admitted,
            &c.maximum(&c.eq(wet_cell, &zero)?, &c.gt(&depth, &zero)?)?,
        )?;
        admitted = c.mul(&admitted, &c.le(&depth, &self.cap)?)?;
        admitted = c.mul(
            &admitted,
            &c.maximum(&c.eq(discharge, &zero)?, &c.gt(&c.abs(&velocity)?, &zero)?)?,
        )?;
        Ok(admitted)
    }

    fn triangular_endpoint_mean(
        &self,
        mean: &Tensor<R>,
        left: &Tensor<R>,
        right: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let negative_mean = c.neg(mean)?;
        let (left_high, left_low) = c.two_sum(left, &negative_mean)?;
        let (right_high, right_low) = c.two_sum(right, &negative_mean)?;
        Ok(c.mul(
            &c.eq(&left_high, &c.neg(&right_high)?)?,
            &c.eq(&left_low, &c.neg(&right_low)?)?,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::{
        ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
        TriangularChannelCell,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn fixture(
        lengths: &[f64],
        areas: &[f64],
        discharges: &[f64],
    ) -> Result<(Context<CpuRuntime>, Fields<CpuRuntime>), Box<dyn std::error::Error>> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let volume: Vec<_> = areas.iter().zip(lengths).map(|(a, dx)| a * dx).collect();
        let momentum: Vec<_> = discharges
            .iter()
            .zip(lengths)
            .map(|(q, dx)| q * dx)
            .collect();
        let context = Context::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "test datum".into(),
                provenance: "triangular conserved reconstruction fixture".into(),
                geometry: ChannelGeometry::PrismaticTriangular {
                    left_bank_slope_h_per_v: 1.0,
                    right_bank_slope_h_per_v: 1.0,
                    bed_elevation_m: 0.0,
                    cells: lengths
                        .iter()
                        .enumerate()
                        .map(|(index, length)| TriangularChannelCell {
                            id: format!("cell-{index}"),
                            length_m: *length,
                            maximum_depth_m: 10.0,
                            manning_n: 0.0,
                        })
                        .collect(),
                },
                lateral_bindings: Vec::new(),
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: volume.clone(),
                longitudinal_momentum_m4_per_s: momentum.clone(),
            },
        )?;
        let fields = context.fields_high(&context.tensor(&volume)?, &context.tensor(&momentum)?)?;
        Ok((context, fields))
    }

    #[test]
    fn nonuniform_center_secants_preserve_nonzero_conservative_endpoints() -> TestResult {
        let (context, fields) = fixture(&[2.0, 4.0, 6.0], &[1.0, 4.0, 9.0], &[2.0, 8.0, 18.0])?;
        let endpoints = context.triangular_endpoints(&fields)?;
        assert_eq!(endpoints.left_area.try_to_vec::<f64>()?, [1.0, 2.0, 9.0]);
        assert_eq!(endpoints.right_area.try_to_vec::<f64>()?, [1.0, 6.0, 9.0]);
        assert_eq!(
            endpoints.left_discharge.try_to_vec::<f64>()?,
            [2.0, 4.0, 18.0]
        );
        assert_eq!(
            endpoints.right_discharge.try_to_vec::<f64>()?,
            [2.0, 12.0, 18.0]
        );
        for (mean, left, right) in [
            (&fields.area, &endpoints.left_area, &endpoints.right_area),
            (
                &fields.discharge,
                &endpoints.left_discharge,
                &endpoints.right_discharge,
            ),
        ] {
            assert!(context.all(&context.triangular_endpoint_mean(mean, left, right)?)?);
        }
        assert_eq!(fields.area.try_to_vec::<f64>()?, [1.0, 4.0, 9.0]);
        assert_eq!(fields.discharge.try_to_vec::<f64>()?, [2.0, 8.0, 18.0]);
        Ok(())
    }

    #[test]
    fn dry_cell_retains_zero_area_and_discharge_endpoints() -> TestResult {
        let (context, fields) = fixture(&[2.0; 3], &[1.0, 0.0, 1.0], &[1.0, 0.0, -1.0])?;
        let endpoints = context.triangular_endpoints(&fields)?;
        for value in [
            endpoints.left_area,
            endpoints.right_area,
            endpoints.left_discharge,
            endpoints.right_discharge,
        ] {
            assert_eq!(value.narrow(0, 1, 1)?.item::<f64>()?, 0.0);
        }
        Ok(())
    }

    #[test]
    fn binade_mean_discrepancy_resets_both_conserved_slopes() -> TestResult {
        let epsilon = f64::EPSILON;
        let (context, fields) = fixture(
            &[2.0; 3],
            &[1.0 - epsilon, 1.0, 1.0 + epsilon],
            &[0.0, 1.0, 2.0],
        )?;
        let mean = context.tensor(&[1.0])?;
        let left = context.tensor(&[1.0 - epsilon / 2.0])?;
        let right = context.tensor(&[1.0])?;
        assert!(!context.all(&context.triangular_endpoint_mean(&mean, &left, &right)?)?);
        let endpoints = context.triangular_endpoints(&fields)?;
        for value in [
            endpoints.left_area,
            endpoints.right_area,
            endpoints.left_discharge,
            endpoints.right_discharge,
        ] {
            assert_eq!(value.narrow(0, 1, 1)?.item::<f64>()?, 1.0);
        }
        Ok(())
    }

    #[test]
    fn reference_selection_retains_cell_mean_faces() -> TestResult {
        let (mut context, fields) = fixture(&[2.0; 3], &[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0])?;
        let mut definition = context.definition.as_ref().clone();
        definition.config.triangular_reconstruction =
            ChannelTriangularReconstruction::FirstOrderReference;
        context.definition = std::sync::Arc::new(definition);
        let endpoints = context.triangular_endpoints(&fields)?;
        assert_eq!(
            endpoints.left_area.try_to_vec::<f64>()?,
            fields.area.try_to_vec::<f64>()?
        );
        assert_eq!(
            endpoints.right_area.try_to_vec::<f64>()?,
            fields.area.try_to_vec::<f64>()?
        );
        assert_eq!(
            endpoints.left_discharge.try_to_vec::<f64>()?,
            fields.discharge.try_to_vec::<f64>()?
        );
        assert_eq!(
            endpoints.right_discharge.try_to_vec::<f64>()?,
            fields.discharge.try_to_vec::<f64>()?
        );
        Ok(())
    }
}
