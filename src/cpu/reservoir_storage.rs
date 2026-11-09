//! Persistent configured CPU reservoir geometry, branch masks, and antecedent state.

use super::CpuTensor;
use crate::runoff::{ReservoirConfig, ReservoirElement, ReservoirError, ReservoirState};
use numr::ops::{BinaryOps, CompareOps, ReduceOps, TypeConversionOps, UnaryOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};
use std::collections::BTreeSet;

#[derive(Debug)]
pub(crate) struct CpuReservoirContext {
    pub elements: Vec<ReservoirElement>,
    pub config: ReservoirConfig,
    pub client: CpuClient,
    pub device: CpuDevice,
    pub area: CpuTensor,
    pub depression: CpuTensor,
    pub alpha: CpuTensor,
    pub instantaneous: CpuTensor,
    pub closed: CpuTensor,
    pub zero: CpuTensor,
    pub one: CpuTensor,
}

#[derive(Clone, Debug)]
pub(crate) struct CpuReservoirState {
    pub depth: CpuTensor,
}

impl CpuReservoirContext {
    pub(crate) fn new(
        elements: Vec<ReservoirElement>,
        config: ReservoirConfig,
    ) -> Result<Self, ReservoirError> {
        config.check()?;
        if elements.is_empty() {
            return Err(ReservoirError::input(
                "reservoir batch requires at least one explicit element",
            ));
        }
        let mut ids = BTreeSet::new();
        for (index, element) in elements.iter().enumerate() {
            if element.id.is_empty()
                || element.id.chars().any(char::is_control)
                || !ids.insert(&element.id)
            {
                return Err(ReservoirError::input(format!(
                    "id={:?}: require unique nonempty identities without control characters",
                    element.id
                ))
                .at_element(index));
            }
        }
        drop(ids);
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        if let Some(error) = super::arithmetic::probe_f64(&client, &device)? {
            return Err(ReservoirError::input(error.message("reservoir")));
        }
        let tensor = |values: Vec<f64>| CpuTensor::from_slice(&values, &[values.len()], &device);
        Ok(Self {
            area: tensor(elements.iter().map(|e| e.parameters.area_m2()).collect())?,
            depression: tensor(
                elements
                    .iter()
                    .map(|e| e.parameters.depression_depth_m())
                    .collect(),
            )?,
            alpha: tensor(
                elements
                    .iter()
                    .map(|e| e.parameters.coefficient_si().unwrap_or(0.0))
                    .collect(),
            )?,
            instantaneous: tensor(
                elements
                    .iter()
                    .map(|e| {
                        if e.parameters.is_instantaneous() {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            )?,
            closed: tensor(
                elements
                    .iter()
                    .map(|e| if e.parameters.is_closed() { 1.0 } else { 0.0 })
                    .collect(),
            )?,
            zero: tensor(vec![0.0; elements.len()])?,
            one: tensor(vec![1.0; elements.len()])?,
            elements,
            config,
            client,
            device,
        })
    }
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<CpuTensor, ReservoirError> {
        Ok(CpuTensor::from_slice(
            values,
            &[values.len()],
            &self.device,
        )?)
    }
    pub(crate) fn any(&self, mask: &CpuTensor) -> Result<bool, ReservoirError> {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }
    pub(crate) fn finite(&self, value: &CpuTensor, quantity: &str) -> Result<(), ReservoirError> {
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
            "Rescale reservoir inputs and geometry",
        )
    }
    pub(crate) fn reject_mask(
        &self,
        mask: &CpuTensor,
        quantity: &str,
        fields: &[(&str, &CpuTensor)],
        scalars: &[(&str, f64)],
        action: &str,
    ) -> Result<(), ReservoirError> {
        if !self.any(mask)? {
            return Ok(());
        }
        let flags = mask.try_to_vec::<f64>()?;
        let index = flags
            .iter()
            .position(|x| *x != 0.0)
            .ok_or_else(|| ReservoirError::numerical("rejection mask contradicts its reduction"))?;
        let element = self.elements.get(index).ok_or_else(|| {
            ReservoirError::numerical("rejection support differs from ordered elements")
        })?;
        let mut context = format!("quantity={quantity}, id={:?}", element.id);
        for (name, tensor) in fields {
            let values = tensor.try_to_vec::<f64>()?;
            let value = values.get(index).ok_or_else(|| {
                ReservoirError::numerical("diagnostic tensor support differs from ordered elements")
            })?;
            context.push_str(&format!(", {name}={value:.17e}"));
        }
        for (name, value) in scalars {
            context.push_str(&format!(", {name}={value:.17e}"));
        }
        Err(ReservoirError::numerical(format!(
            "{context}, parameters={:?}: {action}",
            element.parameters
        ))
        .at_element(index))
    }
    pub(crate) fn total(&self, value: &CpuTensor, quantity: &str) -> Result<f64, ReservoirError> {
        self.finite(value, quantity)?;
        let total = self.client.sum(value, &[], false)?.item::<f64>()?;
        if !total.is_finite() {
            return Err(ReservoirError::numerical(format!(
                "{quantity} aggregate={total}: rescale reservoir support"
            )));
        }
        Ok(total)
    }
    pub(crate) fn storage_volume(&self, depth: &CpuTensor) -> Result<f64, ReservoirError> {
        self.finite(depth, "reservoir depth")?;
        self.reject_mask(
            &self.client.lt(depth, &self.zero)?,
            "negative reservoir depth",
            &[("depth_m", depth)],
            &[],
            "Supply nonnegative antecedent depths",
        )?;
        let volume = self.client.mul(depth, &self.area)?;
        self.finite(&volume, "reservoir storage volume")?;
        self.reject_mask(
            &self.client.mul(
                &self.client.gt(depth, &self.zero)?,
                &self.client.eq(&volume, &self.zero)?,
            )?,
            "positive storage underflow",
            &[("depth_m", depth), ("storage_m3", &volume)],
            &[],
            "Rescale reservoir area or depth",
        )?;
        self.total(&volume, "reservoir storage volume")
    }
}

impl CpuReservoirState {
    pub(crate) fn new(
        context: &CpuReservoirContext,
        states: &[ReservoirState],
    ) -> Result<Self, ReservoirError> {
        if states.len() != context.elements.len() {
            return Err(ReservoirError::input(format!(
                "states={}: require {}",
                states.len(),
                context.elements.len()
            )));
        }
        for (index, state) in states.iter().enumerate() {
            if !state.depth_m.is_finite() || state.depth_m < 0.0 {
                return Err(ReservoirError::input(format!(
                    "depth_m={}: require finite nonnegative antecedent storage",
                    state.depth_m
                ))
                .at_element(index));
            }
        }
        Ok(Self {
            depth: context.tensor(&states.iter().map(|s| s.depth_m).collect::<Vec<_>>())?,
        })
    }
    pub(crate) fn project(
        &self,
        context: &CpuReservoirContext,
    ) -> Result<Vec<ReservoirState>, ReservoirError> {
        let depths = self.depth.try_to_vec::<f64>()?;
        if depths.len() != context.elements.len() {
            return Err(ReservoirError::numerical(
                "depth output support differs from ordered elements",
            ));
        }
        Ok(depths
            .into_iter()
            .map(|depth_m| ReservoirState { depth_m })
            .collect())
    }
}
