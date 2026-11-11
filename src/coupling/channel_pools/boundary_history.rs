use super::ChannelPoolError;
use super::accounting_math::{add_bounds, nonnegative, zero};
use super::context_types::Context;
use super::state::ParentCounters;
use crate::channel::ChannelSnapshotData;
use crate::numerics::subcatchment::bounds::rounding_with_client;
use crate::numerics::water_transfers::{AggregationKind, TransferTree, finite, reject, upward_add};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Context<R>
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
    pub(super) fn check_boundary_history(
        &self,
        reaches: &[ChannelSnapshotData],
        counters: &ParentCounters<R>,
        steps: usize,
    ) -> Result<(), ChannelPoolError> {
        if reaches.len() != self.setup.reaches.len()
            || reaches.iter().any(|reach| reach.accepted_steps != steps)
        {
            return Err(ChannelPoolError::input(
                "checkpoint channel counts differ from common history: supply complete coordinated child packages",
            ));
        }
        let factor = child_history_factor(steps)?;
        let tree = self.boundary_history_tree()?;
        let c = &self.client;
        for (connection, external, connection_bound, external_bound, incoming) in [
            (
                &counters.connection_out_m3,
                &counters.external_in_m3,
                &counters.connection_out_bound_m3,
                &counters.external_in_bound_m3,
                true,
            ),
            (
                &counters.connection_in_m3,
                &counters.external_out_m3,
                &counters.connection_in_bound_m3,
                &counters.external_out_bound_m3,
                false,
            ),
        ] {
            let values = c.cat(&[connection, external], 0)?;
            nonnegative(c, &values, "checkpoint directional parent boundary history")?;
            let grouped = tree.aggregate(c, &values, AggregationKind::Momentum)?;
            let bounds = c.cat(&[connection_bound, external_bound], 0)?;
            nonnegative(c, &bounds, "checkpoint directional parent counter bounds")?;
            // Claimed historical bounds cannot supply arbitrary credit for invented gross traffic.
            let independent_bound = child_history_bound(c, &values, factor)?;
            let bounded = c.minimum(&bounds, &independent_bound)?;
            let grouped_bounds = tree.aggregate(c, &bounded, AggregationKind::Momentum)?;
            let child_values = reaches
                .iter()
                .map(|reach| {
                    if incoming {
                        reach.ledger.boundary_in_m3
                    } else {
                        reach.ledger.boundary_out_m3
                    }
                })
                .collect::<Vec<_>>();
            let child = Tensor::<R>::from_slice(&child_values, &[child_values.len()], c.device())?;
            nonnegative(c, &child, "checkpoint original child boundary history")?;
            if steps == 0 {
                reject(
                    c,
                    &c.ne(&child, &zero(c)?)?,
                    "zero-history channel boundary counters are nonzero. Restore original child accounting",
                )?;
            }
            let zero = zero(c)?;
            reject(
                c,
                &c.ne(&c.eq(&grouped.value, &zero)?, &c.eq(&child, &zero)?)?,
                "checkpoint zero boundary histories differ between channel and endpoint owners. Restore complete original counters",
            )?;
            let child_bound = child_history_bound(c, &child, factor)?;
            let allowance = add_bounds(
                c,
                &[
                    &grouped.bound,
                    &grouped_bounds.value,
                    &grouped_bounds.bound,
                    &child_bound,
                ],
            )?;
            let (difference, low) = c.two_sum(&grouped.value, &c.neg(&child)?)?;
            finite(
                c,
                &difference,
                "checkpoint channel boundary-history difference",
            )?;
            finite(
                c,
                &low,
                "checkpoint channel boundary-history difference low",
            )?;
            let used = upward_add(c, &c.abs(&difference)?, &c.abs(&low)?)?;
            reject(
                c,
                &c.gt(&used, &allowance)?,
                "checkpoint directional channel history disagrees with endpoint owners. Restore complete channel and parent counters",
            )?;
        }
        Ok(())
    }

    fn boundary_history_tree(&self) -> Result<TransferTree<R>, ChannelPoolError> {
        let connections = self.setup.connections.len();
        let count = connections
            .checked_add(self.topology.external_endpoints.len())
            .ok_or_else(|| {
                ChannelPoolError::input("checkpoint endpoint history count overflows")
            })?;
        let mut targets = Vec::with_capacity(count);
        let mut keys = Vec::with_capacity(count);
        for (index, connection) in self.setup.connections.iter().enumerate() {
            let reach = self
                .topology
                .reach_by_id
                .get(&connection.reach_id)
                .copied()
                .ok_or_else(|| {
                    ChannelPoolError::input("checkpoint connection lacks its declared reach")
                })?;
            targets.push((reach, index));
            keys.push(format!("connection/{}", connection.id));
        }
        for (index, &(reach, endpoint)) in self.topology.external_endpoints.iter().enumerate() {
            let definition = self.setup.reaches.get(reach).ok_or_else(|| {
                ChannelPoolError::input("checkpoint external endpoint lacks its declared reach")
            })?;
            targets.push((reach, connections + index));
            keys.push(format!("external/{}/{endpoint}", definition.id));
        }
        let targets = targets
            .iter()
            .zip(&keys)
            .map(|(&(reach, input), key)| (reach, input, key.as_str()))
            .collect::<Vec<_>>();
        Ok(TransferTree::new_grouped(
            &self.client,
            self.setup.reaches.len(),
            count,
            &targets,
        )?)
    }
}

fn child_history_factor(steps: usize) -> Result<f64, ChannelPoolError> {
    if steps as u128 >= (1u128 << 51) {
        return Err(ChannelPoolError::input(
            "checkpoint accepted count exceeds channel history rounding support: supply fewer than 2^51 accepted intervals",
        ));
    }
    if steps == 0 {
        return Ok(0.0);
    }
    // Two positive additions per interval give gamma_(2n). Express its bound against the stored counter.
    let delta = steps as f64 * f64::EPSILON;
    Ok((delta / (1.0 - 2.0 * delta)).next_up())
}

fn child_history_bound<R: Runtime<DType = DType>>(
    client: &R::Client,
    child: &Tensor<R>,
    factor: f64,
) -> Result<Tensor<R>, ChannelPoolError>
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
    let product = client.mul_scalar(child, factor)?;
    finite(
        client,
        &product,
        "checkpoint child boundary-history error product",
    )?;
    let rounding = rounding_with_client(client, &product)?;
    finite(
        client,
        &rounding,
        "checkpoint child boundary-history product rounding",
    )?;
    let upper = upward_add(client, &product, &rounding)?;
    let zero = zero(client)?;
    let result = if factor == 0.0 {
        client.mul_scalar(child, 0.0)?
    } else {
        client.where_cond(&client.gt(child, &zero)?, &upper, &zero)?
    };
    finite(
        client,
        &result,
        "checkpoint child boundary-history arithmetic bound",
    )?;
    Ok(result)
}
