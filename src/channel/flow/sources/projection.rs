use super::super::{
    ChannelError, ChannelLateralLedger, ChannelLateralReceipt, forcing::BoundLateralInputs,
};
use super::{LateralContext, LateralReceipt, LateralState};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use std::collections::BTreeMap;

impl<R: Runtime<DType = DType>> LateralContext<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub fn project_ledgers(
        &self,
        client: &R::Client,
        state: &LateralState<R>,
    ) -> Result<Vec<ChannelLateralLedger>, ChannelError> {
        let data = self.state_data(client, state)?;
        Ok(self
            .definition
            .lateral_bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| ChannelLateralLedger {
                boundary_id: binding.boundary_id.clone(),
                cumulative_volume_m3: data.cumulative_volume_m3[index],
                roundoff_bound_m3: data.counter_roundoff_bound_m3[index],
            })
            .collect())
    }

    pub fn project_receipts(
        &self,
        client: &R::Client,
        inputs: &BoundLateralInputs<R>,
        previous: &LateralState<R>,
        receipt: &LateralReceipt<R>,
    ) -> Result<Vec<ChannelLateralReceipt>, ChannelError> {
        let original = inputs.project(client)?;
        let mut lookup: BTreeMap<_, _> = original
            .into_iter()
            .map(|input| (input.boundary_id().to_owned(), input))
            .collect();
        let previous = self.state_data(client, previous)?;
        let volume = receipt.volume.try_to_vec::<f64>()?;
        let momentum = receipt.momentum.try_to_vec::<f64>()?;
        let velocity = receipt.applied_velocity.try_to_vec::<f64>()?;
        let defined = receipt.velocity_defined.try_to_vec::<f64>()?;
        let low = receipt.counter_addition_low.try_to_vec::<f64>()?;
        let next = receipt.next_counter.try_to_vec::<f64>()?;
        let next_bound = receipt.next_counter_bound.try_to_vec::<f64>()?;
        for (name, len) in [
            ("volume", volume.len()),
            ("momentum", momentum.len()),
            ("velocity", velocity.len()),
            ("velocity-defined mask", defined.len()),
            ("counter low", low.len()),
            ("next counter", next.len()),
            ("next bound", next_bound.len()),
        ] {
            if len != self.count() {
                return Err(ChannelError::input(format!(
                    "lateral receipt {name} length={len}: restore a complete receipt for every binding"
                )));
            }
        }
        self.definition
            .lateral_bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| {
                let input = lookup.remove(binding.boundary_id.as_str()).ok_or_else(|| {
                    ChannelError::input(format!(
                        "missing original boundary_id={:?}: restore the complete source input",
                        binding.boundary_id
                    ))
                })?;
                Ok(ChannelLateralReceipt {
                    input,
                    cell_id: binding.cell_id.clone(),
                    applied_longitudinal_velocity_m_per_s: if defined[index] == 0.0 {
                        None
                    } else {
                        Some(velocity[index])
                    },
                    volume_m3: volume[index],
                    momentum_m4_per_s: momentum[index],
                    cumulative_volume_before_m3: previous.cumulative_volume_m3[index],
                    cumulative_volume_after_m3: next[index],
                    counter_addition_low_m3: low[index],
                    counter_bound_before_m3: previous.counter_roundoff_bound_m3[index],
                    counter_bound_after_m3: next_bound[index],
                })
            })
            .collect()
    }
}
