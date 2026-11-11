use super::super::{ChannelBoundaryTrace, ChannelError, ChannelReservoirTrace};
use super::types::NativeBoundaryTraces;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> NativeBoundaryTraces<R> {
    pub fn project(&self) -> Result<[Option<ChannelBoundaryTrace>; 2], ChannelError> {
        let count = self.endpoints.len();
        if !(1..=2).contains(&count) || self.root_iterations.len() != count {
            return Err(ChannelError::numerical(format!(
                "scalar trace endpoint count={count}, root counts={}: retain one receipt per scalar endpoint",
                self.root_iterations.len()
            )));
        }
        let read = |tensor: &numr::tensor::Tensor<R>| -> Result<Vec<f64>, ChannelError> {
            if tensor.shape() != [count] || tensor.dtype() != DType::F64 {
                return Err(ChannelError::numerical(format!(
                    "scalar trace shape={:?}, dtype={:?}: retain packed F64 endpoint receipts of length {count}",
                    tensor.shape(),
                    tensor.dtype()
                )));
            }
            Ok(tensor.try_to_vec::<f64>()?)
        };
        let depth_m = read(&self.depth_m)?;
        let area_m2 = read(&self.area_m2)?;
        let outward_velocity_m_per_s = read(&self.outward_velocity_m_per_s)?;
        let outward_discharge_m3_per_s = read(&self.outward_discharge_m3_per_s)?;
        let momentum_flux_m4_per_s2 = read(&self.momentum_flux_m4_per_s2)?;
        let signal_speed_bound_m_per_s = read(&self.signal_speed_bound_m_per_s)?;
        let velocity_compatibility_abs_bound_m_per_s =
            read(&self.velocity_compatibility_abs_bound_m_per_s)?;
        let water_compatibility_abs_bound_m3_per_s =
            read(&self.water_compatibility_abs_bound_m3_per_s)?;
        let momentum_compatibility_abs_bound_m4_per_s2 =
            read(&self.momentum_compatibility_abs_bound_m4_per_s2)?;
        let water_flux_width_m3_per_s = read(&self.water_flux_width_m3_per_s)?;
        let momentum_flux_width_m4_per_s2 = read(&self.momentum_flux_width_m4_per_s2)?;
        let cap = read(&self.cap_compatibility_approximation)?;
        let mut reservoir_receipts = [None, None];
        if let Some(reservoir) = &self.reservoir {
            let count = reservoir.endpoints.len();
            if !(1..=2).contains(&count) {
                return Err(ChannelError::numerical(
                    "reservoir endpoint count is invalid: retain actual reservoir endpoints",
                ));
            }
            let read = |tensor: &numr::tensor::Tensor<R>| -> Result<Vec<f64>, ChannelError> {
                if tensor.shape() != [count] || tensor.dtype() != DType::F64 {
                    return Err(ChannelError::numerical(
                        "reservoir receipt metadata is invalid: retain packed F64 actual endpoints",
                    ));
                }
                Ok(tensor.try_to_vec::<f64>()?)
            };
            let lower = read(&reservoir.depth_lower_m)?;
            let upper = read(&reservoir.depth_upper_m)?;
            let residual = read(&reservoir.head_compatibility_abs_bound_m)?;
            let dissipation_lower = read(&reservoir.dissipation_head_lower_m)?;
            let dissipation_upper = read(&reservoir.dissipation_head_upper_m)?;
            for (index, &endpoint) in reservoir.endpoints.iter().enumerate() {
                let slot = reservoir_receipts.get_mut(endpoint).ok_or_else(|| {
                    ChannelError::numerical(
                        "reservoir endpoint is invalid: retain left/right indices zero or one",
                    )
                })?;
                if slot.is_some() || !self.endpoints.contains(&endpoint) {
                    return Err(ChannelError::numerical(
                        "reservoir endpoint mapping is invalid: retain unique actual scalar endpoints",
                    ));
                }
                *slot = Some(ChannelReservoirTrace {
                    depth_lower_m: lower[index],
                    depth_upper_m: upper[index],
                    head_compatibility_abs_bound_m: residual[index],
                    dissipation_head_lower_m: dissipation_lower[index],
                    dissipation_head_upper_m: dissipation_upper[index],
                });
            }
        }
        let mut result = [None, None];
        for (index, &endpoint) in self.endpoints.iter().enumerate() {
            let slot = result.get_mut(endpoint).ok_or_else(|| {
                ChannelError::numerical(format!(
                    "scalar trace endpoint={endpoint}: retain left/right indices zero or one"
                ))
            })?;
            if slot.is_some() || (cap[index] != 0.0 && cap[index] != 1.0) {
                return Err(ChannelError::numerical(format!(
                    "scalar trace endpoint={endpoint}, cap mask={}: retain unique endpoints and binary masks",
                    cap[index]
                )));
            }
            *slot = Some(ChannelBoundaryTrace {
                depth_m: depth_m[index],
                area_m2: area_m2[index],
                outward_velocity_m_per_s: outward_velocity_m_per_s[index],
                outward_discharge_m3_per_s: outward_discharge_m3_per_s[index],
                momentum_flux_m4_per_s2: momentum_flux_m4_per_s2[index],
                signal_speed_bound_m_per_s: signal_speed_bound_m_per_s[index],
                velocity_compatibility_abs_bound_m_per_s: velocity_compatibility_abs_bound_m_per_s
                    [index],
                water_compatibility_abs_bound_m3_per_s: water_compatibility_abs_bound_m3_per_s
                    [index],
                momentum_compatibility_abs_bound_m4_per_s2:
                    momentum_compatibility_abs_bound_m4_per_s2[index],
                water_flux_width_m3_per_s: water_flux_width_m3_per_s[index],
                momentum_flux_width_m4_per_s2: momentum_flux_width_m4_per_s2[index],
                cap_compatibility_approximation: cap[index] == 1.0,
                root_iterations: self.root_iterations[index],
                reservoir: reservoir_receipts[endpoint].take(),
            });
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::{
        runtime::{
            RuntimeClient,
            cpu::{CpuClient, CpuDevice, CpuRuntime},
        },
        tensor::Tensor,
    };

    fn traces() -> Result<NativeBoundaryTraces<CpuRuntime>, ChannelError> {
        let client = CpuClient::new(CpuDevice::new());
        let values = Tensor::from_slice(&[-0.0, 2.0], &[2], client.device())?;
        Ok(NativeBoundaryTraces {
            depth_m: values.clone(),
            area_m2: values.clone(),
            outward_velocity_m_per_s: values.clone(),
            outward_discharge_m3_per_s: values.clone(),
            momentum_flux_m4_per_s2: values.clone(),
            signal_speed_bound_m_per_s: values.clone(),
            velocity_compatibility_abs_bound_m_per_s: values.clone(),
            water_compatibility_abs_bound_m3_per_s: values.clone(),
            momentum_compatibility_abs_bound_m4_per_s2: values.clone(),
            water_flux_width_m3_per_s: values.clone(),
            momentum_flux_width_m4_per_s2: values.clone(),
            cap_compatibility_approximation: Tensor::from_slice(
                &[0.0, 1.0],
                &[2],
                client.device(),
            )?,
            reservoir: None,
            endpoints: vec![1, 0],
            root_iterations: vec![3, 7],
        })
    }

    #[test]
    fn projection_preserves_endpoint_mapping_signed_zero_and_root_counts()
    -> Result<(), ChannelError> {
        let projected = traces()?.project()?;
        let left = projected[0]
            .as_ref()
            .ok_or_else(|| ChannelError::numerical("left trace absent"))?;
        let right = projected[1]
            .as_ref()
            .ok_or_else(|| ChannelError::numerical("right trace absent"))?;
        assert_eq!(left.depth_m.to_bits(), 2.0f64.to_bits());
        assert_eq!(
            right.outward_discharge_m3_per_s.to_bits(),
            (-0.0f64).to_bits()
        );
        assert!(left.cap_compatibility_approximation);
        assert!(!right.cap_compatibility_approximation);
        assert_eq!(left.root_iterations, 7);
        assert_eq!(right.root_iterations, 3);
        Ok(())
    }

    #[test]
    fn reservoir_projection_retains_only_actual_endpoint_bounds() -> Result<(), ChannelError> {
        let mut native = traces()?;
        let value = native.depth_m.narrow(0, 1, 1)?;
        native.reservoir = Some(super::super::types::NativeReservoirTraces {
            depth_lower_m: value.clone(),
            depth_upper_m: value.clone(),
            head_compatibility_abs_bound_m: value.clone(),
            dissipation_head_lower_m: value.clone(),
            dissipation_head_upper_m: value,
            endpoints: vec![0],
        });
        let projected = native.project()?;
        let left = projected[0]
            .as_ref()
            .and_then(|trace| trace.reservoir.as_ref())
            .ok_or_else(|| ChannelError::numerical("left reservoir trace absent"))?;
        assert_eq!(left.depth_lower_m.to_bits(), 2.0f64.to_bits());
        assert_eq!(left.dissipation_head_upper_m.to_bits(), 2.0f64.to_bits());
        assert!(
            projected[1]
                .as_ref()
                .is_some_and(|trace| trace.reservoir.is_none())
        );
        Ok(())
    }

    #[test]
    fn projection_rejects_inconsistent_native_receipts() -> Result<(), ChannelError> {
        for index in 0..4 {
            let mut receipt = traces()?;
            match index {
                0 => receipt.endpoints = vec![0, 0],
                1 => receipt.endpoints = vec![0, 2],
                2 => receipt.root_iterations.clear(),
                _ => receipt.endpoints.clear(),
            }
            assert!(receipt.project().is_err());
        }
        Ok(())
    }
}
