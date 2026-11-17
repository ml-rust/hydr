use super::{
    ChannelPoolError,
    accounting_math::{nonnegative, zero},
    context_types::Context,
    state::{NativeInterval, NativeIntervalAccounting, ParentCounters},
};
use crate::numerics::water_transfers::{finite, reject, upward_add};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> ParentCounters<R> {
    pub(super) fn zeros(
        client: &R::Client,
        connections: usize,
        external: usize,
        pools: usize,
    ) -> Result<Self, ChannelPoolError> {
        let zeros = |count| {
            Tensor::zeros(&[count], DType::F64, client.device()).map_err(ChannelPoolError::from)
        };
        Ok(Self {
            connection_in_m3: zeros(connections)?,
            connection_out_m3: zeros(connections)?,
            connection_in_bound_m3: zeros(connections)?,
            connection_out_bound_m3: zeros(connections)?,
            external_in_m3: zeros(external)?,
            external_out_m3: zeros(external)?,
            external_in_bound_m3: zeros(external)?,
            external_out_bound_m3: zeros(external)?,
            pool_incoming_assembly_bound_m3: zeros(pools)?,
            pool_outgoing_assembly_bound_m3: zeros(pools)?,
            historical_physical_charge_m3: zeros(3)?,
        })
    }
}

impl<R: Runtime<DType = DType>> ParentCounters<R>
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
    pub(super) fn check(
        &self,
        client: &R::Client,
        connections: usize,
        external: usize,
        pools: usize,
        steps: usize,
    ) -> Result<(), ChannelPoolError> {
        for (value, count) in [
            (&self.connection_in_m3, connections),
            (&self.connection_out_m3, connections),
            (&self.connection_in_bound_m3, connections),
            (&self.connection_out_bound_m3, connections),
            (&self.external_in_m3, external),
            (&self.external_out_m3, external),
            (&self.external_in_bound_m3, external),
            (&self.external_out_bound_m3, external),
            (&self.pool_incoming_assembly_bound_m3, pools),
            (&self.pool_outgoing_assembly_bound_m3, pools),
            (&self.historical_physical_charge_m3, 3),
        ] {
            if value.shape() != [count]
                || value.dtype() != DType::F64
                || !value.device().is_same(client.device())
            {
                return Err(ChannelPoolError::input(format!(
                    "parent counter shape={:?},dtype={:?}: supply F64 [{count}] on the configured device",
                    value.shape(),
                    value.dtype()
                )));
            }
            nonnegative(client, value, "parent counter or arithmetic history")?;
            if steps == 0 {
                reject(
                    client,
                    &client.ne(value, &zero(client)?)?,
                    "zero accepted steps require zero parent counters and arithmetic history",
                )
                .map_err(ChannelPoolError::from_transfer)?;
            }
        }
        for (value, bound) in [
            (&self.connection_in_m3, &self.connection_in_bound_m3),
            (&self.connection_out_m3, &self.connection_out_bound_m3),
            (&self.external_in_m3, &self.external_in_bound_m3),
            (&self.external_out_m3, &self.external_out_bound_m3),
        ] {
            reject(
                client,
                &client.mul(
                    &client.eq(value, &zero(client)?)?,
                    &client.ne(bound, &zero(client)?)?,
                )?,
                "zero parent throughput requires zero cumulative addition bound",
            )
            .map_err(ChannelPoolError::from_transfer)?;
        }
        Ok(())
    }
}

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
    pub(super) fn advance_counters(
        &self,
        previous: &ParentCounters<R>,
        interval: &NativeInterval<R>,
        accounting: &NativeIntervalAccounting<R>,
    ) -> Result<ParentCounters<R>, ChannelPoolError> {
        let c = &self.client;
        let z = zero(c)?;
        let connection_in = c.maximum(&interval.connection_volume_m3, &z)?;
        let connection_out = c.maximum(&c.neg(&interval.connection_volume_m3)?, &z)?;
        let external_in = c.maximum(&c.neg(&interval.external_volume_m3)?, &z)?;
        let external_out = c.maximum(&interval.external_volume_m3, &z)?;
        let (connection_in_m3, connection_in_bound_m3) = accumulate(
            c,
            &previous.connection_in_m3,
            &previous.connection_in_bound_m3,
            &connection_in,
        )?;
        let (connection_out_m3, connection_out_bound_m3) = accumulate(
            c,
            &previous.connection_out_m3,
            &previous.connection_out_bound_m3,
            &connection_out,
        )?;
        let (external_in_m3, external_in_bound_m3) = accumulate(
            c,
            &previous.external_in_m3,
            &previous.external_in_bound_m3,
            &external_in,
        )?;
        let (external_out_m3, external_out_bound_m3) = accumulate(
            c,
            &previous.external_out_m3,
            &previous.external_out_bound_m3,
            &external_out,
        )?;
        Ok(ParentCounters {
            connection_in_m3,
            connection_out_m3,
            connection_in_bound_m3,
            connection_out_bound_m3,
            external_in_m3,
            external_out_m3,
            external_in_bound_m3,
            external_out_bound_m3,
            pool_incoming_assembly_bound_m3: upward_add(
                c,
                &previous.pool_incoming_assembly_bound_m3,
                &interval.pool_incoming_bound_m3,
            )
            .map_err(ChannelPoolError::from_transfer)?,
            pool_outgoing_assembly_bound_m3: upward_add(
                c,
                &previous.pool_outgoing_assembly_bound_m3,
                &interval.pool_outgoing_bound_m3,
            )
            .map_err(ChannelPoolError::from_transfer)?,
            historical_physical_charge_m3: upward_add(
                c,
                &previous.historical_physical_charge_m3,
                &accounting.physical_charge_m3,
            )
            .map_err(ChannelPoolError::from_transfer)?,
        })
    }
}

fn accumulate<R: Runtime<DType = DType>>(
    client: &R::Client,
    previous: &Tensor<R>,
    previous_bound: &Tensor<R>,
    increment: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), ChannelPoolError>
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
    nonnegative(client, increment, "parent interval counter increment")?;
    let (value, low) = client.two_sum(previous, increment)?;
    finite(client, &value, "parent cumulative transfer")
        .map_err(ChannelPoolError::from_transfer)?;
    finite(client, &low, "parent cumulative addition low")
        .map_err(ChannelPoolError::from_transfer)?;
    let bound = upward_add(client, previous_bound, &client.abs(&low)?)
        .map_err(ChannelPoolError::from_transfer)?;
    Ok((value, bound))
}
