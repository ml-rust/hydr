//! Configured persistent pool curves, identity groups, and reduction support.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::storage::LevelPoolSourcePolicy;
use crate::storage::{LevelPoolConfig, LevelPoolElement, LevelPoolError};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub(crate) struct TensorPoolGroup<R: Runtime<DType = DType>> {
    pub indices: Tensor<R>,
    pub volumes: Tensor<R>,
    pub stages: Tensor<R>,
    pub areas: Tensor<R>,
    pub index_one: Tensor<R>,
    pub index_last: Tensor<R>,
}

pub(crate) struct TensorPoolContext<R: Runtime<DType = DType>> {
    pub elements: Vec<LevelPoolElement>,
    pub config: LevelPoolConfig,
    pub client: R::Client,
    pub device: R::Device,
    pub zero: Tensor<R>,
    pub maximum: Tensor<R>,
    pub origin: Tensor<R>,
    pub groups: Vec<TensorPoolGroup<R>>,
    pub inverse_permutation: Tensor<R>,
    pub source_policies: Option<Vec<LevelPoolSourcePolicy>>,
}
impl<R> TensorPoolContext<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(crate) fn new(
        client: R::Client,
        elements: Vec<LevelPoolElement>,
        config: LevelPoolConfig,
    ) -> Result<Self, LevelPoolError> {
        Self::new_with_sources(client, elements, config, None)
    }
    pub(crate) fn new_with_sources(
        client: R::Client,
        elements: Vec<LevelPoolElement>,
        config: LevelPoolConfig,
        source_policies: Option<Vec<LevelPoolSourcePolicy>>,
    ) -> Result<Self, LevelPoolError> {
        config.check()?;
        if elements.is_empty() {
            return Err(LevelPoolError::input(
                "pool batch requires at least one explicit element",
            ));
        }
        if let Some(policies) = &source_policies {
            if policies.len() != elements.len() {
                return Err(LevelPoolError::input(
                    "source policy count must match complete ordered pool support",
                ));
            }
            for (index, policy) in policies.iter().enumerate() {
                policy.check().map_err(|e| e.at(index))?;
            }
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
        let device = client.device().clone();
        if let Some(error) = crate::tensor::arithmetic::probe_f64::<R, _>(&client)? {
            return Err(LevelPoolError::input(error.message("level-pool")));
        }
        let tensor = |v: &[f64]| Tensor::<R>::from_slice(v, &[v.len()], &device);
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
            cached.push(TensorPoolGroup {
                indices: Tensor::<R>::from_slice(&indices, &[indices.len()], &device)?,
                volumes: tensor(&volumes)?,
                stages: tensor(&stages)?,
                areas: tensor(
                    &volumes
                        .windows(2)
                        .zip(stages.windows(2))
                        .map(|(v, h)| (v[1] - v[0]) / (h[1] - h[0]))
                        .collect::<Vec<_>>(),
                )?,
                index_one: Tensor::<R>::from_slice(
                    &vec![1i64; indices.len()],
                    &[indices.len()],
                    &device,
                )?,
                index_last: Tensor::<R>::from_slice(
                    &vec![
                        i64::try_from(volumes.len() - 2).map_err(|_| LevelPoolError::input(
                            "curve segment count exceeds I64 indexing"
                        ))?;
                        indices.len()
                    ],
                    &[indices.len()],
                    &device,
                )?,
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
            inverse_permutation: Tensor::<R>::from_slice(&inverse, &[inverse.len()], &device)?,
            elements,
            config,
            client,
            device,
            groups: cached,
            source_policies,
        })
    }
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<Tensor<R>, LevelPoolError> {
        Ok(Tensor::<R>::from_slice(
            values,
            &[values.len()],
            &self.device,
        )?)
    }
    pub(crate) fn any(&self, mask: &Tensor<R>) -> Result<bool, LevelPoolError> {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }
    pub(crate) fn reject(&self, mask: &Tensor<R>, quantity: &str) -> Result<(), LevelPoolError> {
        if !self.any(mask)? {
            return Ok(());
        }
        let values = mask.try_to_vec::<f64>()?;
        let index = values.iter().position(|v| *v != 0.0).ok_or_else(|| {
            LevelPoolError::numerical("element rejection mask contradicts its reduction")
        })?;
        Err(LevelPoolError::numerical(format!("{quantity}: rescale storage/transfer quantities or supply an admitted curve and interval")).at(index))
    }
    pub(crate) fn finite(&self, value: &Tensor<R>, quantity: &str) -> Result<(), LevelPoolError> {
        if value.dtype() != DType::F64 || !value.device().is_same(self.client.device()) {
            return Err(LevelPoolError::input(format!(
                "{quantity}: require F64 tensors on the configured runtime client device"
            )));
        }
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
        value: &Tensor<R>,
        quantity: &str,
    ) -> Result<(), LevelPoolError> {
        self.finite(value, quantity)?;
        self.reject(&self.client.lt(value, &self.zero)?, quantity)
    }
    pub(crate) fn range(&self, value: &Tensor<R>, quantity: &str) -> Result<(), LevelPoolError> {
        self.nonnegative(value, quantity)?;
        self.reject(&self.client.gt(value, &self.maximum)?, quantity)
    }
    pub(crate) fn sum(&self, value: &Tensor<R>, quantity: &str) -> Result<f64, LevelPoolError> {
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

impl<R: Runtime<DType = DType>> std::fmt::Debug for TensorPoolContext<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorPoolContext").finish_non_exhaustive()
    }
}
