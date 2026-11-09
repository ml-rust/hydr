//! Configured persistent pool curves, identity groups, and reduction support.

use crate::cpu::CpuTensor;
use crate::storage::{LevelPoolConfig, LevelPoolElement, LevelPoolError};
use numr::ops::{BinaryOps, CompareOps, ReduceOps, TypeConversionOps, UnaryOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub(crate) struct CpuPoolGroup {
    pub indices: CpuTensor,
    pub volumes: CpuTensor,
    pub stages: CpuTensor,
}

#[derive(Debug)]
pub(crate) struct CpuPoolContext {
    pub elements: Vec<LevelPoolElement>,
    pub config: LevelPoolConfig,
    pub client: CpuClient,
    pub device: CpuDevice,
    pub zero: CpuTensor,
    pub maximum: CpuTensor,
    pub origin: CpuTensor,
    pub groups: Vec<CpuPoolGroup>,
    pub inverse_permutation: CpuTensor,
}
impl CpuPoolContext {
    pub(crate) fn new(
        elements: Vec<LevelPoolElement>,
        config: LevelPoolConfig,
    ) -> Result<Self, LevelPoolError> {
        config.check()?;
        if elements.is_empty() {
            return Err(LevelPoolError::input(
                "pool batch requires at least one explicit element",
            ));
        }
        let mut ids = BTreeSet::new();
        let mut supports = BTreeSet::new();
        let mut groups = BTreeMap::<_, Vec<i64>>::new();
        for (index, element) in elements.iter().enumerate() {
            for (value, field) in [
                (&element.id, "element.id"),
                (
                    &element.connected_pool_assertion,
                    "connected_pool_assertion",
                ),
                (
                    &element.storage_support_identity,
                    "storage_support_identity",
                ),
            ] {
                crate::storage::definition::text(value, field).map_err(|e| e.at(index))?;
            }
            if !ids.insert(&element.id) || !supports.insert(&element.storage_support_identity) {
                return Err(LevelPoolError::input(
                    "element and storage-support identities must each be unique",
                )
                .at(index));
            }
            let id = i64::try_from(index)
                .map_err(|_| LevelPoolError::input("pool element count exceeds I64 indexing"))?;
            groups.entry(element.curve.key()).or_default().push(id);
        }
        drop(ids);
        drop(supports);
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        if let Some(error) = crate::cpu::arithmetic::probe_f64(&client, &device)? {
            return Err(LevelPoolError::input(error.message("level-pool")));
        }
        let tensor = |v: &[f64]| CpuTensor::from_slice(v, &[v.len()], &device);
        let mut cached = Vec::with_capacity(groups.len());
        let mut inverse = vec![0i64; elements.len()];
        let mut position = 0i64;
        for ((_, _, knots), indices) in groups {
            for index in &indices {
                inverse[*index as usize] = position;
                position += 1;
            }
            let volumes: Vec<_> = knots.iter().map(|(v, _)| f64::from_bits(*v)).collect();
            let stages: Vec<_> = knots.iter().map(|(_, h)| f64::from_bits(*h)).collect();
            cached.push(CpuPoolGroup {
                indices: CpuTensor::from_slice(&indices, &[indices.len()], &device)?,
                volumes: tensor(&volumes)?,
                stages: tensor(&stages)?,
            });
        }
        Ok(Self {
            maximum: tensor(
                &elements
                    .iter()
                    .map(|e| e.curve.maximum_volume_m3())
                    .collect::<Vec<_>>(),
            )?,
            origin: tensor(
                &elements
                    .iter()
                    .map(|e| e.curve.origin())
                    .collect::<Vec<_>>(),
            )?,
            zero: tensor(&vec![0.0; elements.len()])?,
            inverse_permutation: CpuTensor::from_slice(&inverse, &[inverse.len()], &device)?,
            elements,
            config,
            client,
            device,
            groups: cached,
        })
    }
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<CpuTensor, LevelPoolError> {
        Ok(CpuTensor::from_slice(
            values,
            &[values.len()],
            &self.device,
        )?)
    }
    pub(crate) fn any(&self, mask: &CpuTensor) -> Result<bool, LevelPoolError> {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }
    pub(crate) fn reject(&self, mask: &CpuTensor, quantity: &str) -> Result<(), LevelPoolError> {
        if !self.any(mask)? {
            return Ok(());
        }
        let values = mask.try_to_vec::<f64>()?;
        let index = values.iter().position(|v| *v != 0.0).ok_or_else(|| {
            LevelPoolError::numerical("element rejection mask contradicts its reduction")
        })?;
        Err(LevelPoolError::numerical(format!("{quantity}: rescale storage/transfer quantities or supply an admitted curve and interval")).at(index))
    }
    pub(crate) fn finite(&self, value: &CpuTensor, quantity: &str) -> Result<(), LevelPoolError> {
        self.reject(
            &self.client.maximum(
                &self
                    .client
                    .cast(&self.client.isnan(value)?, value.dtype())?,
                &self
                    .client
                    .cast(&self.client.isinf(value)?, value.dtype())?,
            )?,
            quantity,
        )
    }
    pub(crate) fn nonnegative(
        &self,
        value: &CpuTensor,
        quantity: &str,
    ) -> Result<(), LevelPoolError> {
        self.finite(value, quantity)?;
        self.reject(&self.client.lt(value, &self.zero)?, quantity)
    }
    pub(crate) fn range(&self, value: &CpuTensor, quantity: &str) -> Result<(), LevelPoolError> {
        self.nonnegative(value, quantity)?;
        self.reject(&self.client.gt(value, &self.maximum)?, quantity)
    }
    pub(crate) fn sum(&self, value: &CpuTensor, quantity: &str) -> Result<f64, LevelPoolError> {
        self.finite(value, quantity)?;
        let result = self.client.sum(value, &[], false)?.item::<f64>()?;
        if !result.is_finite() {
            return Err(LevelPoolError::numerical(format!(
                "{quantity}: aggregate reduction overflows. Rescale the batch"
            )));
        }
        Ok(result)
    }
}
