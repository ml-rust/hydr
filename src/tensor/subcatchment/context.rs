//! Complete flattened support and configured component reductions.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::group::{TensorSubareaGroup, index_tensor};
use crate::runoff::{Subarea, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentError};
use std::collections::BTreeSet;

pub(crate) struct TensorSubcatchmentContext<R: Runtime<DType = DType>> {
    pub definitions: Vec<SubcatchmentDefinition>,
    pub config: SubcatchmentConfig,
    pub client: R::Client,
    pub device: R::Device,
    pub area: Tensor<R>,
    pub zero: Tensor<R>,
    pub total_area: Tensor<R>,
    pub catchment_zero: Tensor<R>,
    pub catchment_indices: Tensor<R>,
    pub catchment_slots: Tensor<R>,
    pub phase1: Option<TensorSubareaGroup<R>>,
    pub phase2: Option<TensorSubareaGroup<R>>,
}

impl<R> TensorSubcatchmentContext<R>
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
        let device = client.device().clone();
        if let Some(error) = crate::tensor::arithmetic::probe_f64::<R, _>(&client)? {
            return Err(SubcatchmentError::input(error.message("subcatchment")));
        }
        let tensor = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], &device);
        let phase1 = TensorSubareaGroup::<R>::new(client.clone(), first, &definitions, &config)?;
        let phase2 = TensorSubareaGroup::<R>::new(client.clone(), second, &definitions, &config)?;
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
            catchment_indices: index_tensor::<R>(&device, &catchments)?,
            catchment_slots: index_tensor::<R>(&device, &slots)?,
            definitions,
            config,
            client,
            device,
            phase1,
            phase2,
        })
    }
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<Tensor<R>, SubcatchmentError> {
        Ok(Tensor::<R>::from_slice(
            values,
            &[values.len()],
            &self.device,
        )?)
    }
    pub(crate) fn any(&self, mask: &Tensor<R>) -> Result<bool, SubcatchmentError> {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }
    pub(crate) fn finite(
        &self,
        value: &Tensor<R>,
        quantity: &str,
    ) -> Result<(), SubcatchmentError> {
        if value.dtype() != DType::F64 || !value.device().is_same(self.client.device()) {
            return Err(SubcatchmentError::input(format!(
                "{quantity}: require F64 tensors on the configured runtime client device"
            )));
        }
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
        mask: &Tensor<R>,
        quantity: &str,
        fields: &[(&str, &Tensor<R>)],
        scalars: &[(&str, f64)],
        action: &str,
    ) -> Result<(), SubcatchmentError> {
        if !self.any(mask)? {
            return Ok(());
        }
        let selected = self.client.argmax(mask, 0, false)?.item::<i64>()?;
        let lane = usize::try_from(selected)
            .ok()
            .filter(|lane| *lane < mask.numel())
            .ok_or_else(|| {
                SubcatchmentError::numerical("rejection mask contradicts its reduction")
            })?;
        if mask.narrow(0, lane, 1)?.item::<f64>()? == 0.0 {
            return Err(SubcatchmentError::numerical(
                "rejection mask contradicts its reduction",
            ));
        }
        let mut message = format!("quantity={quantity}");
        for (name, tensor) in fields {
            if lane >= tensor.numel() {
                return Err(SubcatchmentError::numerical(
                    "diagnostic support differs from rejected mask",
                ));
            }
            let value = tensor.narrow(0, lane, 1)?.item::<f64>()?;
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
        value: &Tensor<R>,
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
    pub(crate) fn storage_volume(&self, depth: &Tensor<R>) -> Result<f64, SubcatchmentError> {
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

impl<R: Runtime<DType = DType>> std::fmt::Debug for TensorSubcatchmentContext<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorSubcatchmentContext")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "rejection_tests.rs"]
mod rejection_tests;
