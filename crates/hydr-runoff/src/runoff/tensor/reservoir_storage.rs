//! Persistent configured runtime reservoir geometry, branch masks, and antecedent state.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::runoff::{ReservoirConfig, ReservoirElement, ReservoirError, ReservoirState};
use std::collections::BTreeSet;

pub(crate) struct TensorReservoirContext<R: Runtime<DType = DType>> {
    pub elements: Vec<ReservoirElement>,
    pub config: ReservoirConfig,
    pub client: R::Client,
    pub device: R::Device,
    pub area: Tensor<R>,
    pub depression: Tensor<R>,
    pub alpha: Tensor<R>,
    pub instantaneous: Tensor<R>,
    pub closed: Tensor<R>,
    pub zero: Tensor<R>,
    pub one: Tensor<R>,
}

#[derive(Clone, Debug)]
pub(crate) struct TensorReservoirState<R: Runtime<DType = DType>> {
    pub depth: Tensor<R>,
}

impl<R> TensorReservoirContext<R>
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
        let device = client.device().clone();
        if let Some(error) = crate::tensor::probe_f64::<R, _>(&client)? {
            return Err(ReservoirError::input(error.message("reservoir")));
        }
        let tensor = |values: Vec<f64>| Tensor::<R>::from_slice(&values, &[values.len()], &device);
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
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<Tensor<R>, ReservoirError> {
        Ok(Tensor::<R>::from_slice(
            values,
            &[values.len()],
            &self.device,
        )?)
    }
    pub(crate) fn any(&self, mask: &Tensor<R>) -> Result<bool, ReservoirError> {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }
    pub(crate) fn finite(&self, value: &Tensor<R>, quantity: &str) -> Result<(), ReservoirError> {
        if value.dtype() != DType::F64 || !value.device().is_same(self.client.device()) {
            return Err(ReservoirError::input(format!(
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
            "Rescale reservoir inputs and geometry",
        )
    }
    pub(crate) fn reject_mask(
        &self,
        mask: &Tensor<R>,
        quantity: &str,
        fields: &[(&str, &Tensor<R>)],
        scalars: &[(&str, f64)],
        action: &str,
    ) -> Result<(), ReservoirError> {
        if !self.any(mask)? {
            return Ok(());
        }
        let selected = self.client.argmax(mask, 0, false)?.item::<i64>()?;
        let index = usize::try_from(selected)
            .ok()
            .filter(|index| *index < mask.numel())
            .ok_or_else(|| ReservoirError::numerical("rejection mask contradicts its reduction"))?;
        if mask.narrow(0, index, 1)?.item::<f64>()? == 0.0 {
            return Err(ReservoirError::numerical(
                "rejection mask contradicts its reduction",
            ));
        }
        let element = self.elements.get(index).ok_or_else(|| {
            ReservoirError::numerical("rejection support differs from ordered elements")
        })?;
        let mut context = format!("quantity={quantity}, id={:?}", element.id);
        for (name, tensor) in fields {
            if index >= tensor.numel() {
                return Err(ReservoirError::numerical(
                    "diagnostic tensor support differs from ordered elements",
                ));
            }
            let value = tensor.narrow(0, index, 1)?.item::<f64>()?;
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
    pub(crate) fn total(&self, value: &Tensor<R>, quantity: &str) -> Result<f64, ReservoirError> {
        self.finite(value, quantity)?;
        let total = self.client.sum(value, &[], false)?.item::<f64>()?;
        if !total.is_finite() {
            return Err(ReservoirError::numerical(format!(
                "{quantity} aggregate={total}: rescale reservoir support"
            )));
        }
        Ok(total)
    }
    pub(crate) fn storage_volume(&self, depth: &Tensor<R>) -> Result<f64, ReservoirError> {
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

impl<R> TensorReservoirState<R>
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
        context: &TensorReservoirContext<R>,
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
        context: &TensorReservoirContext<R>,
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

impl<R: Runtime<DType = DType>> std::fmt::Debug for TensorReservoirContext<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorReservoirContext")
            .finish_non_exhaustive()
    }
}
