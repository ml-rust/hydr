//! Checked physical bindings and cached boundary-to-pool assembly.

use super::super::{CoupledLevelPoolError, topology::TransferTree};
use super::{CoupledPoolInflow, CoupledPoolInflowBinding};
use crate::numerics::outward::AdjustmentReduction;
use crate::storage::LevelPoolElement;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct InflowContext<R: Runtime<DType = DType>> {
    pub(super) bindings: Arc<Vec<CoupledPoolInflowBinding>>,
    pub(super) tree: TransferTree<R>,
    pub(super) capacity: Tensor<R>,
    pub(super) boundary_reduction: AdjustmentReduction,
    pub(super) pool_reduction: AdjustmentReduction,
    pub(super) pool_count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct BoundInflows<R: Runtime<DType = DType>> {
    pub(super) bindings: Arc<Vec<CoupledPoolInflowBinding>>,
    pub(super) values: Tensor<R>,
    pub(super) rate_mask: Tensor<R>,
    pub(super) input_is_rate: Arc<Vec<bool>>,
    pub(super) projection_order: Arc<Vec<usize>>,
    has_interval_volumes: bool,
}

impl<R: Runtime<DType = DType>> InflowContext<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(crate) fn new(
        client: &R::Client,
        pools: &[LevelPoolElement],
        bindings: &[CoupledPoolInflowBinding],
    ) -> Result<Self, CoupledLevelPoolError> {
        let pools_by_id: HashMap<_, _> = pools
            .iter()
            .enumerate()
            .map(|(i, p)| (p.id.as_str(), i))
            .collect();
        let mut boundaries = HashSet::new();
        let mut inputs = HashSet::new();
        let mut targets = Vec::with_capacity(bindings.len());
        let mut capacities = Vec::with_capacity(bindings.len());
        for (index, binding) in bindings.iter().enumerate() {
            for (field, value) in [
                ("boundary_id", &binding.boundary_id),
                ("pool_id", &binding.pool_id),
                ("input_id", &binding.input_id),
                ("physical_support_id", &binding.physical_support_id),
                ("boundary_assertion", &binding.boundary_assertion),
                ("ownership_assertion", &binding.ownership_assertion),
            ] {
                if value.trim().is_empty() {
                    return Err(CoupledLevelPoolError::input(format!(
                        "external inflow {index} has empty {field}. Supply explicit identity and ownership evidence"
                    )));
                }
            }
            if !boundaries.insert(binding.boundary_id.as_str())
                || !inputs.insert(binding.input_id.as_str())
            {
                return Err(CoupledLevelPoolError::input(
                    "external boundary and supplied input identities must each be unique",
                ));
            }
            let pool = *pools_by_id.get(binding.pool_id.as_str()).ok_or_else(|| {
                CoupledLevelPoolError::input(format!(
                    "external boundary {} references unknown pool {}",
                    binding.boundary_id, binding.pool_id
                ))
            })?;
            targets.push((pool, index, binding.boundary_id.as_str()));
            let element = pools.get(pool).ok_or_else(|| {
                CoupledLevelPoolError::input("resolved external pool index exceeds pool count")
            })?;
            capacities.push(element.curve.maximum_volume_m3());
        }
        let tree = TransferTree::new_grouped(client, pools.len(), bindings.len(), &targets)?;
        Ok(Self {
            bindings: Arc::new(bindings.to_vec()),
            tree,
            capacity: Tensor::from_slice(&capacities, &[capacities.len()], client.device())?,
            boundary_reduction: AdjustmentReduction::new(bindings.len()).map_err(map_error)?,
            pool_reduction: AdjustmentReduction::new(pools.len()).map_err(map_error)?,
            pool_count: pools.len(),
        })
    }

    pub(crate) fn bind_inputs(
        &self,
        client: &R::Client,
        inputs: &[CoupledPoolInflow],
    ) -> Result<BoundInflows<R>, CoupledLevelPoolError> {
        if inputs.len() != self.bindings.len() {
            return Err(CoupledLevelPoolError::input(
                "supply one explicit external input per declared boundary, including zeros",
            ));
        }
        let mut supplied = HashMap::with_capacity(inputs.len());
        for input in inputs {
            let scalar = input.value.scalar();
            if !scalar.is_finite() || scalar < 0.0 {
                return Err(CoupledLevelPoolError::input(format!(
                    "external input {}={:?} must be finite and nonnegative",
                    input.boundary_id, input.value,
                )));
            }
            if supplied.insert(input.boundary_id.as_str(), input).is_some() {
                return Err(CoupledLevelPoolError::input(format!(
                    "external input {} is duplicated",
                    input.boundary_id,
                )));
            }
        }
        let mut ordered = Vec::with_capacity(inputs.len());
        let mut mask = Vec::with_capacity(inputs.len());
        let mut kinds = Vec::with_capacity(inputs.len());
        for binding in self.bindings.iter() {
            let input = supplied
                .remove(binding.boundary_id.as_str())
                .ok_or_else(|| {
                    CoupledLevelPoolError::input(format!(
                        "external boundary {} has no supplied input",
                        binding.boundary_id
                    ))
                })?;
            if input.input_id != binding.input_id {
                return Err(CoupledLevelPoolError::input(format!(
                    "external boundary {} expects input {}, received {}",
                    binding.boundary_id, binding.input_id, input.input_id,
                )));
            }
            ordered.push(input.value.scalar());
            kinds.push(input.value.is_rate());
            mask.push(if input.value.is_rate() { 1.0 } else { 0.0 });
        }
        let indices: HashMap<_, _> = self
            .bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| (binding.boundary_id.as_str(), index))
            .collect();
        let projection_order = inputs
            .iter()
            .map(|input| {
                indices
                    .get(input.boundary_id.as_str())
                    .copied()
                    .ok_or_else(|| {
                        CoupledLevelPoolError::input(
                            "original input identity disappeared during binding",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BoundInflows {
            bindings: Arc::clone(&self.bindings),
            projection_order: Arc::new(projection_order),
            values: Tensor::from_slice(&ordered, &[ordered.len()], client.device())?,
            rate_mask: Tensor::from_slice(&mask, &[mask.len()], client.device())?,
            has_interval_volumes: kinds.iter().any(|is_rate| !is_rate),
            input_is_rate: Arc::new(kinds),
        })
    }

    pub(crate) fn bind_interval_volumes(
        &self,
        client: &R::Client,
        values: &Tensor<R>,
    ) -> Result<BoundInflows<R>, CoupledLevelPoolError> {
        if self.bindings.is_empty() {
            return Err(CoupledLevelPoolError::input(
                "native interval volumes require declared nonempty receiving boundaries",
            ));
        }
        self.check_tensor(client, values)?;
        Ok(BoundInflows {
            bindings: Arc::clone(&self.bindings),
            values: values.clone(),
            rate_mask: Tensor::zeros(&[self.bindings.len()], DType::F64, client.device())?,
            input_is_rate: Arc::new(vec![false; self.bindings.len()]),
            projection_order: Arc::new((0..self.bindings.len()).collect()),
            has_interval_volumes: !self.bindings.is_empty(),
        })
    }

    pub(crate) fn restore_inputs(
        &self,
        client: &R::Client,
        inputs: &BoundInflows<R>,
    ) -> Result<BoundInflows<R>, CoupledLevelPoolError> {
        if self.bindings.as_ref() != inputs.bindings.as_ref()
            || inputs.input_is_rate.len() != self.bindings.len()
            || inputs.projection_order.len() != self.bindings.len()
        {
            return Err(CoupledLevelPoolError::input(
                "trusted inflow input identities differ from the destination",
            ));
        }
        self.check_tensor(client, &inputs.values)?;
        self.check_tensor(client, &inputs.rate_mask)?;
        Ok(BoundInflows {
            bindings: Arc::clone(&self.bindings),
            values: inputs.values.clone(),
            rate_mask: inputs.rate_mask.clone(),
            input_is_rate: Arc::clone(&inputs.input_is_rate),
            projection_order: Arc::clone(&inputs.projection_order),
            has_interval_volumes: inputs.has_interval_volumes,
        })
    }
}

pub(super) fn map_error(error: crate::numerics::outward::OutwardError) -> CoupledLevelPoolError {
    CoupledLevelPoolError::numerical(error.to_string())
}

impl<R: Runtime<DType = DType>> BoundInflows<R> {
    pub(crate) fn has_interval_volumes(&self) -> bool {
        self.has_interval_volumes
    }
}
