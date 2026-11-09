//! Complete flattened support and configured component reductions.

use super::group::{CpuSubareaGroup, index_tensor};
use crate::cpu::CpuTensor;
use crate::runoff::{Subarea, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentError};
use numr::ops::{BinaryOps, CompareOps, ReduceOps, TypeConversionOps, UnaryOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};
use std::collections::BTreeSet;

#[derive(Debug)]
pub(crate) struct CpuSubcatchmentContext {
    pub definitions: Vec<SubcatchmentDefinition>,
    pub config: SubcatchmentConfig,
    pub client: CpuClient,
    pub device: CpuDevice,
    pub area: CpuTensor,
    pub zero: CpuTensor,
    pub total_area: CpuTensor,
    pub catchment_zero: CpuTensor,
    pub catchment_indices: CpuTensor,
    pub catchment_slots: CpuTensor,
    pub phase1: Option<CpuSubareaGroup>,
    pub phase2: Option<CpuSubareaGroup>,
}

impl CpuSubcatchmentContext {
    pub(crate) fn new(
        definitions: Vec<SubcatchmentDefinition>,
        config: SubcatchmentConfig,
    ) -> Result<Self, SubcatchmentError> {
        config.check()?;
        if definitions.is_empty() {
            return Err(SubcatchmentError::input(
                "component batch requires at least one named catchment",
            ));
        }
        let count = definitions.len().checked_mul(3).ok_or_else(|| {
            SubcatchmentError::input("flattened catchment support overflows usize")
        })?;
        let mut names = BTreeSet::new();
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut areas = Vec::with_capacity(count);
        for (i, definition) in definitions.iter().enumerate() {
            if definition.id.is_empty()
                || definition.id.chars().any(char::is_control)
                || !names.insert(&definition.id)
            {
                return Err(SubcatchmentError::input(
                    "IDs must be unique, nonempty, and free of control characters",
                )
                .at_catchment(i));
            }
            for area in Subarea::ALL {
                let p = &definition.parameters;
                areas.push(p.area_m2(area));
                if p.closure(area).is_some() {
                    if p.receiver() == Some(area) {
                        second.push(i * 3 + area.index());
                    } else {
                        first.push(i * 3 + area.index());
                    }
                }
            }
        }
        drop(names);
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        if let Some(error) = crate::cpu::arithmetic::probe_f64(&client, &device)? {
            return Err(SubcatchmentError::input(error.message("subcatchment")));
        }
        let tensor = |values: &[f64]| CpuTensor::from_slice(values, &[values.len()], &device);
        let phase1 = CpuSubareaGroup::new(first, &definitions, &config, &device)?;
        let phase2 = CpuSubareaGroup::new(second, &definitions, &config, &device)?;
        let catchments: Vec<usize> = (0..definitions.len()).flat_map(|i| [i; 3]).collect();
        let slots: Vec<usize> = (0..definitions.len()).map(|i| i * 3).collect();
        Ok(Self {
            area: tensor(&areas)?,
            zero: tensor(&vec![0.0; count])?,
            total_area: tensor(
                &definitions
                    .iter()
                    .map(|d| d.parameters.geometry().total_area_m2)
                    .collect::<Vec<_>>(),
            )?,
            catchment_zero: tensor(&vec![0.0; definitions.len()])?,
            catchment_indices: index_tensor(&device, &catchments)?,
            catchment_slots: index_tensor(&device, &slots)?,
            definitions,
            config,
            client,
            device,
            phase1,
            phase2,
        })
    }
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<CpuTensor, SubcatchmentError> {
        Ok(CpuTensor::from_slice(
            values,
            &[values.len()],
            &self.device,
        )?)
    }
    pub(crate) fn any(&self, mask: &CpuTensor) -> Result<bool, SubcatchmentError> {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }
    pub(crate) fn finite(
        &self,
        value: &CpuTensor,
        quantity: &str,
    ) -> Result<(), SubcatchmentError> {
        let invalid = self.client.maximum(
            &self
                .client
                .cast(&self.client.isnan(value)?, value.dtype())?,
            &self
                .client
                .cast(&self.client.isinf(value)?, value.dtype())?,
        )?;
        self.reject_mask(
            &invalid,
            quantity,
            &[("value", value)],
            &[],
            "Rescale component inputs and support",
        )
    }
    pub(crate) fn reject_mask(
        &self,
        mask: &CpuTensor,
        quantity: &str,
        fields: &[(&str, &CpuTensor)],
        scalars: &[(&str, f64)],
        action: &str,
    ) -> Result<(), SubcatchmentError> {
        if !self.any(mask)? {
            return Ok(());
        }
        let flags = mask.try_to_vec::<f64>()?;
        let lane = flags.iter().position(|v| *v != 0.0).ok_or_else(|| {
            SubcatchmentError::numerical("rejection mask contradicts its reduction")
        })?;
        let mut message = format!("quantity={quantity}");
        for (name, tensor) in fields {
            let values = tensor.try_to_vec::<f64>()?;
            let value = values.get(lane).ok_or_else(|| {
                SubcatchmentError::numerical("diagnostic support differs from rejected mask")
            })?;
            message.push_str(&format!(", {name}={value:.17e}"));
        }
        for (name, value) in scalars {
            message.push_str(&format!(", {name}={value:.17e}"));
        }
        let error = SubcatchmentError::numerical(format!("{message}: {action}"));
        if mask.shape() == [self.definitions.len()] {
            Err(error.at_catchment(lane))
        } else {
            Err(error.at(lane / 3, Subarea::from_index(lane % 3)))
        }
    }
    pub(crate) fn total(
        &self,
        value: &CpuTensor,
        quantity: &str,
    ) -> Result<f64, SubcatchmentError> {
        self.finite(value, quantity)?;
        let sum = self.client.sum(value, &[], false)?.item::<f64>()?;
        if !sum.is_finite() {
            return Err(SubcatchmentError::numerical(format!(
                "{quantity} reduction overflows. Rescale support"
            )));
        }
        Ok(sum)
    }
    pub(crate) fn storage_volume(&self, depth: &CpuTensor) -> Result<f64, SubcatchmentError> {
        self.finite(depth, "complete component depth")?;
        let volume = self.client.mul(depth, &self.area)?;
        self.finite(&volume, "component storage")?;
        self.reject_mask(
            &self.client.mul(
                &self.client.gt(depth, &self.zero)?,
                &self.client.eq(&volume, &self.zero)?,
            )?,
            "positive storage underflow",
            &[("depth", depth), ("volume", &volume)],
            &[],
            "Rescale physical area or depth",
        )?;
        self.total(&volume, "component storage")
    }
}
