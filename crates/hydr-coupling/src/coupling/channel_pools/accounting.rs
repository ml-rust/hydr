use super::{
    ChannelPoolError,
    accounting_math::{add_bounds, admit, expression, physical_scale, zero},
    context_types::Context,
    reductions::{reduce, upper_sum},
    state::{Accepted, NativeEvaluation, NativeIntervalAccounting, NativeWaterBalance},
};
use crate::channel::ChannelReach;
use crate::numerics::water_transfers::{AggregationKind, reject, upward_add};
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::Runtime;
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
    pub(super) fn inventory(
        &self,
        reaches: &[ChannelReach<R>],
        pools: &LevelPoolBatch<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelPoolError> {
        self.reductions.inventory(&self.client, reaches, pools)
    }

    pub(super) fn interval_accounting(
        &self,
        previous: &Accepted<R>,
        evaluation: &NativeEvaluation<R>,
    ) -> Result<NativeIntervalAccounting<R>, ChannelPoolError>
    where
        R::Client: numr::ops::TensorOps<R> + numr::ops::NextRepresentableOps<R>,
    {
        self.check_interval_pairing(evaluation)?;
        let c = &self.client;
        let (storage_before_m3, before_bound) =
            self.inventory(&previous.reaches, &previous.pools)?;
        let (storage_after_m3, after_bound) =
            self.inventory(&evaluation.reaches, &evaluation.pools)?;
        let external = &evaluation.interval.external_volume_m3;
        let (external_in_m3, external_in_bound) = reduce(
            c,
            &self.reductions.external_counters,
            &c.maximum(&c.neg(external)?, &zero(c)?)?,
        )?;
        let (external_out_m3, external_out_bound) = reduce(
            c,
            &self.reductions.external_counters,
            &c.maximum(external, &zero(c)?)?,
        )?;
        let (lateral_in_m3, lateral_out_m3, lateral_bound) =
            self.lateral_totals(&evaluation.reaches, true)?;
        let (residual_m3, arithmetic_bound) = expression(
            c,
            &[
                (&storage_after_m3, true),
                (&storage_before_m3, false),
                (&external_in_m3, false),
                (&external_out_m3, true),
                (&lateral_in_m3, false),
                (&lateral_out_m3, true),
            ],
        )?;
        let physical_charge_m3 = self.physical_charges(evaluation)?;
        let physical_bound = self.scalar_physical_charge(&physical_charge_m3)?;
        let incoming_assembly = upper_sum(
            c,
            &self.reductions.pool_counters,
            &evaluation.interval.pool_incoming_bound_m3,
        )?;
        let outgoing_assembly = upper_sum(
            c,
            &self.reductions.pool_counters,
            &evaluation.interval.pool_outgoing_bound_m3,
        )?;
        let roundoff_bound_m3 = add_bounds(
            c,
            &[
                &before_bound,
                &after_bound,
                &external_in_bound,
                &external_out_bound,
                &lateral_bound,
                &arithmetic_bound,
                &physical_bound,
                &incoming_assembly,
                &outgoing_assembly,
            ],
        )?;
        let scale = physical_scale(
            c,
            &storage_before_m3,
            &storage_after_m3,
            &external_in_m3,
            &external_out_m3,
            &lateral_in_m3,
            &lateral_out_m3,
        )?;
        admit(
            c,
            &residual_m3,
            &roundoff_bound_m3,
            &scale,
            self.setup.config.mass_absolute_m3,
            self.setup.config.mass_relative,
        )?;
        Ok(NativeIntervalAccounting {
            storage_before_m3,
            storage_after_m3,
            external_in_m3,
            external_out_m3,
            lateral_in_m3,
            lateral_out_m3,
            residual_m3,
            roundoff_bound_m3,
            physical_charge_m3,
        })
    }

    pub(super) fn water_balance(
        &self,
        accepted: &Accepted<R>,
    ) -> Result<NativeWaterBalance<R>, ChannelPoolError> {
        accepted.counters.check(
            &self.client,
            self.setup.connections.len(),
            self.topology.external_endpoints.len(),
            self.setup.pools.len(),
            accepted.pools.accepted_steps(),
        )?;
        self.check_counter_pairing(accepted)?;
        let c = &self.client;
        let (storage_m3, storage_bound) = self.inventory(&accepted.reaches, &accepted.pools)?;
        let (external_in_m3, external_in_report_bound) = reduce(
            c,
            &self.reductions.external_counters,
            &accepted.counters.external_in_m3,
        )?;
        let (external_out_m3, external_out_report_bound) = reduce(
            c,
            &self.reductions.external_counters,
            &accepted.counters.external_out_m3,
        )?;
        let external_in_counter_bound = upper_sum(
            c,
            &self.reductions.external_counters,
            &accepted.counters.external_in_bound_m3,
        )?;
        let external_out_counter_bound = upper_sum(
            c,
            &self.reductions.external_counters,
            &accepted.counters.external_out_bound_m3,
        )?;
        let (lateral_in_m3, lateral_out_m3, lateral_bound) =
            self.lateral_totals(&accepted.reaches, false)?;
        let initial_storage_m3 = self.initial_storage_m3.clone();
        let (residual_m3, arithmetic_bound) = expression(
            c,
            &[
                (&storage_m3, true),
                (&initial_storage_m3, false),
                (&external_in_m3, false),
                (&external_out_m3, true),
                (&lateral_in_m3, false),
                (&lateral_out_m3, true),
            ],
        )?;
        let physical_bound =
            self.scalar_physical_charge(&accepted.counters.historical_physical_charge_m3)?;
        let incoming_assembly = upper_sum(
            c,
            &self.reductions.pool_counters,
            &accepted.counters.pool_incoming_assembly_bound_m3,
        )?;
        let outgoing_assembly = upper_sum(
            c,
            &self.reductions.pool_counters,
            &accepted.counters.pool_outgoing_assembly_bound_m3,
        )?;
        let roundoff_bound_m3 = add_bounds(
            c,
            &[
                &self.initial_storage_bound_m3,
                &storage_bound,
                &external_in_report_bound,
                &external_out_report_bound,
                &external_in_counter_bound,
                &external_out_counter_bound,
                &lateral_bound,
                &arithmetic_bound,
                &physical_bound,
                &incoming_assembly,
                &outgoing_assembly,
            ],
        )?;
        let scale = physical_scale(
            c,
            &initial_storage_m3,
            &storage_m3,
            &external_in_m3,
            &external_out_m3,
            &lateral_in_m3,
            &lateral_out_m3,
        )?;
        admit(
            c,
            &residual_m3,
            &roundoff_bound_m3,
            &scale,
            self.setup.config.mass_absolute_m3,
            self.setup.config.mass_relative,
        )?;
        Ok(NativeWaterBalance {
            initial_storage_m3,
            storage_m3,
            external_in_m3,
            external_out_m3,
            lateral_in_m3,
            lateral_out_m3,
            residual_m3,
            roundoff_bound_m3,
        })
    }

    pub(super) fn check_counter_pairing(
        &self,
        accepted: &Accepted<R>,
    ) -> Result<(), ChannelPoolError> {
        let c = &self.client;
        let pools = accepted.pools.coupling_native_accounting();
        for (connections, connection_bounds, actual, actual_bound, assembly) in [
            (
                &accepted.counters.connection_in_m3,
                &accepted.counters.connection_in_bound_m3,
                pools.cumulative_incoming_volume_m3,
                pools.incoming_roundoff_bound_m3,
                &accepted.counters.pool_incoming_assembly_bound_m3,
            ),
            (
                &accepted.counters.connection_out_m3,
                &accepted.counters.connection_out_bound_m3,
                pools.cumulative_outgoing_volume_m3,
                pools.outgoing_roundoff_bound_m3,
                &accepted.counters.pool_outgoing_assembly_bound_m3,
            ),
        ] {
            let grouped = self
                .topology
                .pool_transfers
                .aggregate(c, connections, AggregationKind::Momentum)
                .map_err(ChannelPoolError::from_transfer)?;
            let grouped_bounds = self
                .topology
                .pool_transfers
                .aggregate(c, connection_bounds, AggregationKind::Momentum)
                .map_err(ChannelPoolError::from_transfer)?;
            let grouping_allowance = upward_add(c, &grouped_bounds.value, &grouped_bounds.bound)
                .map_err(ChannelPoolError::from_transfer)?;
            let allowance = add_bounds(
                c,
                &[&grouped.bound, &grouping_allowance, actual_bound, assembly],
            )?;
            let (difference, low) = c.two_sum(&grouped.value, &c.neg(actual)?)?;
            let used = upward_add(c, &c.abs(&difference)?, &c.abs(&low)?)
                .map_err(ChannelPoolError::from_transfer)?;
            reject(
                c,
                &c.gt(&used, &allowance)?,
                "cumulative connection transfers disagree with pool histories. Restore the complete coordinated state",
            )
            .map_err(ChannelPoolError::from_transfer)?;
        }
        Ok(())
    }
}
