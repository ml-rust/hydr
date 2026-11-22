//! Checked physical bindings and cached boundary-to-pool assembly.
//!
//! An owner can append native bindings after the caller bindings. Callers supply inputs for
//! their own bindings only. The owner supplies the native interval volumes once per attempt.

use super::super::{CoupledLevelPoolError, topology::TransferTree};
use super::{CoupledPoolInflow, CoupledPoolInflowBinding};
use crate::numerics::outward::AdjustmentReduction;
use crate::storage::LevelPoolElement;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
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
    /// Count of trailing native bindings, after the caller bindings.
    pub(super) native: usize,
    pub(super) native_reduction: AdjustmentReduction,
}

#[derive(Clone, Debug)]
pub(crate) struct BoundInflows<R: Runtime<DType = DType>> {
    pub(super) bindings: Arc<Vec<CoupledPoolInflowBinding>>,
    pub(super) values: Tensor<R>,
    pub(super) rate_mask: Tensor<R>,
    pub(super) input_is_rate: Arc<Vec<bool>>,
    pub(super) projection_order: Arc<Vec<usize>>,
    /// A caller or native-only binding holds an interval volume.
    has_interval_volumes: bool,
    /// Some value, native ones included, holds an interval volume.
    pub(super) masked: bool,
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
        Self::with_native(client, pools, bindings, &[])
    }

    /// Caller `bindings`, then `native` bindings whose interval volumes the owner supplies.
    pub(crate) fn with_native(
        client: &R::Client,
        pools: &[LevelPoolElement],
        caller: &[CoupledPoolInflowBinding],
        native: &[CoupledPoolInflowBinding],
    ) -> Result<Self, CoupledLevelPoolError> {
        let bindings: Vec<CoupledPoolInflowBinding> =
            caller.iter().chain(native).cloned().collect();
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
            capacity: Tensor::from_slice(&capacities, &[capacities.len()], client.device())?,
            boundary_reduction: AdjustmentReduction::new(bindings.len()).map_err(map_error)?,
            bindings: Arc::new(bindings),
            tree,
            pool_reduction: AdjustmentReduction::new(pools.len()).map_err(map_error)?,
            pool_count: pools.len(),
            native: native.len(),
            native_reduction: AdjustmentReduction::new(native.len()).map_err(map_error)?,
        })
    }

    /// Count of caller bindings, before the native ones.
    pub(crate) fn caller_count(&self) -> usize {
        self.bindings.len() - self.native
    }

    /// Count of native bindings, after the caller ones.
    pub(crate) fn native_count(&self) -> usize {
        self.native
    }

    /// Binds one caller input per caller binding. Native bindings take zero placeholders until
    /// [`Self::with_native_volumes`] supplies them.
    pub(crate) fn bind_inputs(
        &self,
        client: &R::Client,
        inputs: &[CoupledPoolInflow],
    ) -> Result<BoundInflows<R>, CoupledLevelPoolError> {
        if inputs.len() != self.caller_count() {
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
        let mut ordered = Vec::with_capacity(self.bindings.len());
        let mut mask = Vec::with_capacity(self.bindings.len());
        let mut kinds = Vec::with_capacity(self.bindings.len());
        for binding in self.bindings.iter().take(self.caller_count()) {
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
        let caller_volumes = kinds.iter().any(|is_rate| !is_rate);
        for _ in 0..self.native {
            ordered.push(0.0);
            kinds.push(false);
            mask.push(0.0);
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
            .chain((self.caller_count()..self.bindings.len()).map(Ok))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BoundInflows {
            bindings: Arc::clone(&self.bindings),
            projection_order: Arc::new(projection_order),
            values: Tensor::from_slice(&ordered, &[ordered.len()], client.device())?,
            rate_mask: Tensor::from_slice(&mask, &[mask.len()], client.device())?,
            has_interval_volumes: caller_volumes,
            masked: kinds.iter().any(|is_rate| !is_rate),
            input_is_rate: Arc::new(kinds),
        })
    }

    /// The caller inputs of `bound` with `native` interval volumes, `[native count]`, in the
    /// native bindings.
    ///
    /// The native values enter unchanged. Only caller interval volumes fix the span.
    pub(crate) fn with_native_volumes(
        &self,
        client: &R::Client,
        bound: &BoundInflows<R>,
        native: &Tensor<R>,
    ) -> Result<BoundInflows<R>, CoupledLevelPoolError> {
        if !Arc::ptr_eq(&self.bindings, &bound.bindings) {
            return Err(CoupledLevelPoolError::input(
                "external inputs belong to another immutable boundary context",
            ));
        }
        if native.shape() != [self.native]
            || native.dtype() != DType::F64
            || !native.device().is_same(client.device())
        {
            return Err(CoupledLevelPoolError::input(
                "native interval volumes require one F64 value per native binding on the configured device",
            ));
        }
        let caller = self.caller_count();
        let values = if caller == 0 {
            native.contiguous()?
        } else {
            client.cat(&[&bound.values.narrow(0, 0, caller)?, native], 0)?
        };
        Ok(BoundInflows {
            bindings: Arc::clone(&self.bindings),
            values,
            rate_mask: bound.rate_mask.clone(),
            input_is_rate: Arc::clone(&bound.input_is_rate),
            projection_order: Arc::clone(&bound.projection_order),
            has_interval_volumes: bound.has_interval_volumes,
            masked: bound.masked,
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
            masked: !self.bindings.is_empty(),
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
            masked: inputs.masked,
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

#[cfg(test)]
mod tests {
    use super::super::CoupledPoolInflowValue;
    use super::super::fixture::{bindings, client, pools};
    use super::*;
    use numr::runtime::Device;
    use numr::runtime::cpu::CpuRuntime;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn native_exact_binding_preserves_authoritative_operands_and_matches_literal_preparation()
    -> TestResult {
        let client = client();
        let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &bindings())?;
        let original = context.zero_state(&client)?;
        let quantum = 3.0 * f64::from_bits(1);
        let values = Tensor::<CpuRuntime>::from_slice(&[quantum, 0.75], &[2], client.device())?;
        let native = context.bind_interval_volumes(&client, &values)?;
        let literal = context.bind_inputs(
            &client,
            &[
                CoupledPoolInflow {
                    boundary_id: "first".into(),
                    input_id: "input-first".into(),
                    value: CoupledPoolInflowValue::IntervalVolume { volume_m3: quantum },
                },
                CoupledPoolInflow {
                    boundary_id: "second".into(),
                    input_id: "input-second".into(),
                    value: CoupledPoolInflowValue::IntervalVolume { volume_m3: 0.75 },
                },
            ],
        )?;
        let prepared = context.prepare(&client, &native, &original, 2.0, 1)?;
        let expected = context.prepare(&client, &literal, &original, 2.0, 1)?;
        assert!(prepared.step.all_interval_volumes());
        assert_eq!(
            prepared
                .incoming
                .try_to_vec::<f64>()?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![quantum.to_bits(), 0.75f64.to_bits()]
        );
        assert!(prepared.incoming.device().is_same(client.device()));
        assert_eq!(
            context.project_step(&client, &prepared.step)?,
            context.project_step(&client, &expected.step)?
        );
        assert_eq!(
            context.project_state(&client, &prepared.next)?,
            context.project_state(&client, &expected.next)?
        );
        assert_eq!(
            context
                .project_state(&client, &original)?
                .iter()
                .map(|entry| entry.cumulative_volume_m3)
                .collect::<Vec<_>>(),
            vec![0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn native_binding_rejects_metadata_and_original_value_errors() -> TestResult {
        let client = client();
        let context = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &bindings())?;
        for shape in [vec![], vec![1], vec![2, 1], vec![3]] {
            let values = Tensor::<CpuRuntime>::zeros(&shape, DType::F64, client.device())?;
            assert!(context.bind_interval_volumes(&client, &values).is_err());
        }
        for dtype in [DType::F32, DType::I64] {
            let values = Tensor::<CpuRuntime>::zeros(&[2], dtype, client.device())?;
            assert!(context.bind_interval_volumes(&client, &values).is_err());
        }
        for value in [-1.0, f64::NAN, f64::INFINITY] {
            let values = Tensor::<CpuRuntime>::from_slice(&[0.0, value], &[2], client.device())?;
            assert!(context.bind_interval_volumes(&client, &values).is_err());
        }
        let empty = InflowContext::<CpuRuntime>::new(&client, &pools(4.0)?, &[])?;
        let values = Tensor::<CpuRuntime>::zeros(&[0], DType::F64, client.device())?;
        assert!(empty.bind_interval_volumes(&client, &values).is_err());
        assert!(empty.bind_inputs(&client, &[]).is_ok());
        Ok(())
    }
}
