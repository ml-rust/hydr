use super::{
    ChannelPoolError,
    accounting_lateral::pack,
    accounting_math::{add_bounds, expression, zero},
    context_types::Context,
    reductions::reduce,
    state::NativeEvaluation,
};
use crate::numerics::water_transfers::{reject, upward_add};
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
    pub(super) fn physical_charges(
        &self,
        evaluation: &NativeEvaluation<R>,
    ) -> Result<Tensor<R>, ChannelPoolError>
    where
        R::Client: numr::ops::TensorOps<R> + numr::ops::NextRepresentableOps<R>,
    {
        let c = &self.client;
        let mut faces = Vec::with_capacity(evaluation.reaches.len());
        let mut assemblies = Vec::with_capacity(evaluation.reaches.len());
        for (index, reach) in evaluation.reaches.iter().enumerate() {
            let view = reach.coupling_accounting();
            faces.push(
                view.face_transfer_underflow_bound_m3
                    .ok_or_else(|| {
                        ChannelPoolError::numerical(
                            "channel interval transfer certificate is missing",
                        )
                    })?
                    .clone(),
            );
            let assembly = reach.coupling_water_assembly_bound_m3().map_err(|source| {
                ChannelPoolError::Channel {
                    reach_id: self.setup.reaches[index].id.clone(),
                    source,
                }
            })?;
            assemblies.push(assembly.ok_or_else(|| {
                ChannelPoolError::numerical(
                    "channel interval water assembly certificate is missing",
                )
            })?);
        }
        let pools = evaluation.pools.coupling_native_interval().ok_or_else(|| {
            ChannelPoolError::numerical("pool interval water witnesses are missing")
        })?;
        let pool_charge = upward_add(
            c,
            &c.abs(pools.outgoing_witness_m3)?,
            &c.abs(pools.incoming_witness_m3)?,
        )?;
        // This checks witness evaluation, without converting its allowance into historical water loss.
        let (consistency, arithmetic) = expression(
            c,
            &[
                (pools.volume_m3, true),
                (pools.old_volume_m3, false),
                (pools.incoming_volume_m3, false),
                (pools.outgoing_volume_m3, true),
                (pools.outgoing_witness_m3, true),
                (pools.incoming_witness_m3, true),
            ],
        )?;
        let allowed = upward_add(c, pools.witness_consistency_bound_m3, &arithmetic)?;
        reject(
            c,
            &c.gt(&c.abs(&consistency)?, &allowed)?,
            "pool state witnesses disagree with their original water operands. Recalculate the complete interval",
        )?;
        let packed = pack(c, &[pack(c, &faces)?, pack(c, &assemblies)?, pool_charge])?;
        let (charges, reduction_bound) = reduce(c, &self.reductions.physical_charges, &packed)?;
        Ok(upward_add(c, &charges, &reduction_bound)?)
    }

    pub(super) fn scalar_physical_charge(
        &self,
        lanes: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelPoolError> {
        add_bounds(
            &self.client,
            &[
                &lanes.narrow(0, 0, 1)?,
                &lanes.narrow(0, 1, 1)?,
                &lanes.narrow(0, 2, 1)?,
            ],
        )
    }

    pub(super) fn check_interval_pairing(
        &self,
        evaluation: &NativeEvaluation<R>,
    ) -> Result<(), ChannelPoolError> {
        let c = &self.client;
        let z = zero(c)?;
        let incoming = c.maximum(&evaluation.interval.connection_volume_m3, &z)?;
        let outgoing = c.maximum(&c.neg(&evaluation.interval.connection_volume_m3)?, &z)?;
        let incoming = self.topology.pool_transfers.aggregate(
            c,
            &incoming,
            crate::numerics::water_transfers::AggregationKind::Water,
        )?;
        let outgoing = self.topology.pool_transfers.aggregate(
            c,
            &outgoing,
            crate::numerics::water_transfers::AggregationKind::Water,
        )?;
        let pools = evaluation.pools.coupling_native_interval().ok_or_else(|| {
            ChannelPoolError::numerical("pool interval exchange receipt is missing")
        })?;
        for (expected, actual, retained_bound, recalculated_bound) in [
            (
                &incoming.value,
                pools.incoming_volume_m3,
                &evaluation.interval.pool_incoming_bound_m3,
                &incoming.bound,
            ),
            (
                &outgoing.value,
                pools.outgoing_volume_m3,
                &evaluation.interval.pool_outgoing_bound_m3,
                &outgoing.bound,
            ),
        ] {
            reject(
                c,
                &c.ne(expected, actual)?,
                "pool interval exchanges differ from original grouped channel volumes. Preserve shared transfer operands",
            )?;
            reject(
                c,
                &c.ne(retained_bound, recalculated_bound)?,
                "pool interval exchange assembly bounds differ from retained operands",
            )?;
        }
        Ok(())
    }
}
