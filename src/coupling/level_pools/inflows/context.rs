//! Checked physical bindings and cached boundary-to-pool assembly.

use super::super::{CoupledLevelPoolError, topology::TransferTree};
use super::{CoupledPoolInflowBinding, CoupledPoolInflowRate};
use crate::soil::admission::AdjustmentReduction;
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

pub(crate) struct BoundInflows<R: Runtime<DType = DType>> {
    pub(super) bindings: Arc<Vec<CoupledPoolInflowBinding>>,
    pub(super) rates: Tensor<R>,
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

    pub(crate) fn bind_rates(
        &self,
        client: &R::Client,
        rates: &[CoupledPoolInflowRate],
    ) -> Result<BoundInflows<R>, CoupledLevelPoolError> {
        if rates.len() != self.bindings.len() {
            return Err(CoupledLevelPoolError::input(
                "supply one explicit external rate per declared boundary, including zeros",
            ));
        }
        let mut supplied = HashMap::with_capacity(rates.len());
        for rate in rates {
            if !rate.rate_m3_per_s.is_finite() || rate.rate_m3_per_s < 0.0 {
                return Err(CoupledLevelPoolError::input(format!(
                    "external rate {}={} must be finite and nonnegative",
                    rate.boundary_id, rate.rate_m3_per_s
                )));
            }
            if supplied.insert(rate.boundary_id.as_str(), rate).is_some() {
                return Err(CoupledLevelPoolError::input(format!(
                    "external rate {} is duplicated",
                    rate.boundary_id
                )));
            }
        }
        let mut ordered = Vec::with_capacity(rates.len());
        for binding in self.bindings.iter() {
            let rate = supplied
                .remove(binding.boundary_id.as_str())
                .ok_or_else(|| {
                    CoupledLevelPoolError::input(format!(
                        "external boundary {} has no supplied rate",
                        binding.boundary_id
                    ))
                })?;
            if rate.input_id != binding.input_id {
                return Err(CoupledLevelPoolError::input(format!(
                    "external boundary {} expects input {}, received {}",
                    binding.boundary_id, binding.input_id, rate.input_id
                )));
            }
            ordered.push(rate.rate_m3_per_s);
        }
        Ok(BoundInflows {
            bindings: Arc::clone(&self.bindings),
            rates: Tensor::from_slice(&ordered, &[ordered.len()], client.device())?,
        })
    }
}

pub(super) fn map_error(error: crate::soil::SoilError) -> CoupledLevelPoolError {
    CoupledLevelPoolError::numerical(error.to_string())
}
