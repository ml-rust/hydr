use numr::runtime::Device;
use solvr::interpolate::Interp1dAlgorithms;

use super::{
    LEVEL_POOL_CONTRACT_VERSION, LEVEL_POOL_SNAPSHOT_SCHEMA, LevelPoolSnapshot,
    LevelPoolSnapshotData,
};
use crate::storage::batch::StoredStep;
use crate::storage::tensor::{TensorPoolContext, TensorPoolState};
use crate::storage::{LevelPoolBatch, LevelPoolError};
use hydr_core::contract::package_identity;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

impl<R> LevelPoolBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    /// Restore trusted complete state after exact immutable definition checks.
    pub fn restore(&mut self, snapshot: &LevelPoolSnapshot<R>) -> Result<(), LevelPoolError> {
        self.compatible(
            snapshot.batch.elements(),
            snapshot.batch.config(),
            snapshot.batch.source_policies(),
        )?;
        require(
            self.context
                .client
                .device()
                .is_same(snapshot.batch.context.client.device()),
            "trusted snapshot device differs from the target runtime client",
        )?;
        self.state = snapshot.batch.state.clone();
        self.time = snapshot.batch.time;
        self.steps = snapshot.batch.steps;
        self.last = snapshot.batch.last.clone();
        Ok(())
    }
    /// Check complete caller-owned state before atomic replacement, without replaying historical intervals.
    pub fn restore_data(&mut self, data: &LevelPoolSnapshotData) -> Result<(), LevelPoolError> {
        package_identity(
            "level pool",
            &data.schema,
            LEVEL_POOL_SNAPSHOT_SCHEMA,
            &data.contract,
            LEVEL_POOL_CONTRACT_VERSION,
        )
        .map_err(LevelPoolError::input)?;
        self.compatible(
            &data.elements,
            &data.config,
            data.source_policies.as_deref(),
        )?;
        let mut candidate = if let Some(policies) = &data.source_policies {
            Self::new_with_sources(
                self.context.client.clone(),
                data.elements.clone(),
                data.states.clone(),
                data.time,
                data.config.clone(),
                policies.clone(),
            )?
        } else {
            Self::new(
                self.context.client.clone(),
                data.elements.clone(),
                data.states.clone(),
                data.time,
                data.config.clone(),
            )?
        };
        let context = &candidate.context;
        let n = context.elements.len();
        require(
            data.element_ledgers.len() == n,
            "element ledger count differs from definitions",
        )?;
        require(
            (data.accepted_steps == 0) == data.last_step.is_none(),
            "accepted count and last interval presence disagree",
        )?;
        let state = import_state(context, data)?;
        crate::storage::numerics::accounting::check_state(context, &state, data.accepted_steps)?;
        crate::storage::numerics::curves::evaluate(context, &state.initial_volume_m3)?;
        crate::storage::accounting::totals(context, &state)?;
        let computed = crate::storage::projection::ledgers(context, &state)?;
        require(
            computed
                .iter()
                .zip(&data.element_ledgers)
                .all(|(a, b)| a.same_bits(b)),
            "stored element ledger differs from complete current accounting",
        )?;
        let mut last = None;
        if let Some(receipt) = &data.last_step {
            require(
                receipt.start.is_finite()
                    && receipt.start >= 0.0
                    && receipt.end.is_finite()
                    && receipt.end == data.time
                    && receipt.dt.is_finite()
                    && receipt.dt > 0.0
                    && receipt.dt == receipt.end - receipt.start,
                "last interval requires exact positive finite duration and accepted endpoint",
            )?;
            require(
                receipt.receipts.len() == n,
                "last receipt count differs from complete element support",
            )?;
            for (index, r) in receipt.receipts.iter().enumerate() {
                require(
                    r.element == index,
                    "last receipts must retain exact ordered element identities",
                )?;
            }
            if data.accepted_steps == 1 {
                for (r, l) in receipt.receipts.iter().zip(&data.element_ledgers) {
                    require(
                        r.old_volume_m3.to_bits() == l.initial_volume_m3.to_bits()
                            && [
                                r.cumulative_incoming_before_m3,
                                r.cumulative_outgoing_before_m3,
                                r.incoming_bound_before_m3,
                                r.outgoing_bound_before_m3,
                            ]
                            .iter()
                            .all(|v| v.to_bits() == 0),
                        "first interval must continue exact original storage and zero counters/bounds",
                    )?;
                }
            }
            let tensor = |f: fn(&crate::storage::LevelPoolReceipt) -> f64| {
                context.tensor(&receipt.receipts.iter().map(f).collect::<Vec<_>>())
            };
            let before = TensorPoolState {
                sources: super::sources::before(context, data, receipt)?,
                volume_m3: tensor(|r| r.old_volume_m3)?,
                initial_volume_m3: state.initial_volume_m3.clone(),
                cumulative_incoming_volume_m3: tensor(|r| r.cumulative_incoming_before_m3)?,
                cumulative_outgoing_volume_m3: tensor(|r| r.cumulative_outgoing_before_m3)?,
                incoming_roundoff_bound_m3: tensor(|r| r.incoming_bound_before_m3)?,
                outgoing_roundoff_bound_m3: tensor(|r| r.outgoing_bound_before_m3)?,
            };
            crate::storage::numerics::accounting::check_state(
                context,
                &before,
                data.accepted_steps - 1,
            )?;
            crate::storage::numerics::curves::evaluate(context, &before.volume_m3)?;
            crate::storage::accounting::totals(context, &before)?;
            let incoming = tensor(|r| r.incoming_volume_m3)?;
            let outgoing = tensor(|r| r.outgoing_volume_m3)?;
            let (next, step) = if let Some(input) = super::sources::inputs(context, receipt)? {
                crate::storage::numerics::advance_with_sources(
                    context,
                    &before,
                    &incoming,
                    &outgoing,
                    data.accepted_steps,
                    receipt.dt,
                    &input,
                )?
            } else {
                crate::storage::numerics::advance(
                    context,
                    &before,
                    &incoming,
                    &outgoing,
                    data.accepted_steps,
                )?
            };
            let projected = crate::storage::projection::ledgers(context, &next)?;
            require(
                projected
                    .iter()
                    .zip(&data.element_ledgers)
                    .all(|(a, b)| a.same_bits(b)),
                "last cumulative additions/bounds and final volume do not continue into accepted element ledgers",
            )?;
            let stored = StoredStep {
                start: receipt.start,
                end: receipt.end,
                step,
            };
            let checked = crate::storage::projection::step(context, &stored)?;
            require(
                checked.same_bits(receipt),
                "last receipt differs from original TwoSum operands/witnesses, stage inversion, or independent aggregate accounting",
            )?;
            last = Some(Arc::new(stored));
        }
        candidate.state = state;
        candidate.steps = data.accepted_steps;
        candidate.last = last;
        *self = candidate;
        Ok(())
    }
    fn compatible(
        &self,
        elements: &[crate::storage::LevelPoolElement],
        config: &crate::storage::LevelPoolConfig,
        source_policies: Option<&[crate::storage::LevelPoolSourcePolicy]>,
    ) -> Result<(), LevelPoolError> {
        require(
            self.source_policies() == source_policies
                && self.config().same_bits(config)
                && self.elements().len() == elements.len()
                && self
                    .elements()
                    .iter()
                    .zip(elements)
                    .all(|(a, b)| a.same_bits(b)),
            "curve/datum/provenance/assertion/support/config bits differ from the immutable target",
        )
    }
}

fn import_state<R>(
    context: &TensorPoolContext<R>,
    data: &LevelPoolSnapshotData,
) -> Result<TensorPoolState<R>, LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    for (index, (ledger, state)) in data.element_ledgers.iter().zip(&data.states).enumerate() {
        require(
            ledger.element == index && ledger.volume_m3.to_bits() == state.volume_m3.to_bits(),
            "element ledger identity/current volume differs from ordered accepted state",
        )?;
        if data.accepted_steps == 0 {
            require(
                ledger.initial_volume_m3.to_bits() == ledger.volume_m3.to_bits()
                    && [
                        ledger.cumulative_incoming_volume_m3,
                        ledger.cumulative_outgoing_volume_m3,
                        ledger.incoming_roundoff_bound_m3,
                        ledger.outgoing_roundoff_bound_m3,
                    ]
                    .iter()
                    .all(|v| v.to_bits() == 0),
                "zero-history imports require exact initial/current identity and zero counters/bounds",
            )?;
        }
    }
    let tensor = |f: fn(&crate::storage::LevelPoolElementLedger) -> f64| {
        context.tensor(&data.element_ledgers.iter().map(f).collect::<Vec<_>>())
    };
    Ok(TensorPoolState {
        sources: super::sources::state(context, data)?,
        volume_m3: tensor(|l| l.volume_m3)?,
        initial_volume_m3: tensor(|l| l.initial_volume_m3)?,
        cumulative_incoming_volume_m3: tensor(|l| l.cumulative_incoming_volume_m3)?,
        cumulative_outgoing_volume_m3: tensor(|l| l.cumulative_outgoing_volume_m3)?,
        incoming_roundoff_bound_m3: tensor(|l| l.incoming_roundoff_bound_m3)?,
        outgoing_roundoff_bound_m3: tensor(|l| l.outgoing_roundoff_bound_m3)?,
    })
}
fn require(condition: bool, message: &str) -> Result<(), LevelPoolError> {
    if condition {
        Ok(())
    } else {
        Err(LevelPoolError::input(format!(
            "level-pool checkpoint: {message}. Supply an unchanged complete same-build package"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolState,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn trusted_restore_preserves_destination_client() -> Result<(), LevelPoolError> {
        let client = CpuClient::new(CpuDevice::new());
        let elements = vec![LevelPoolElement {
            id: "pool".into(),
            curve: LevelPoolCurve::new(
                vec![
                    LevelPoolKnot {
                        volume_m3: 0.0,
                        stage_m: 0.0,
                    },
                    LevelPoolKnot {
                        volume_m3: 1.0,
                        stage_m: 1.0,
                    },
                ],
                "datum".into(),
                "measured".into(),
            )?,
            connected_pool_assertion: "connected".into(),
            storage_support_identity: "support".into(),
        }];
        let mut target = LevelPoolBatch::<CpuRuntime>::new(
            client.with_parallelism(ParallelismConfig::new(Some(1), Some(64))),
            elements.clone(),
            vec![LevelPoolState { volume_m3: 0.0 }],
            0.0,
            LevelPoolConfig::default(),
        )?;
        let source = LevelPoolBatch::<CpuRuntime>::new(
            client.with_parallelism(ParallelismConfig::new(Some(1), Some(128))),
            elements,
            vec![LevelPoolState { volume_m3: 0.5 }],
            2.0,
            LevelPoolConfig::default(),
        )?;
        let context = Arc::clone(&target.context);
        target.restore(&source.snapshot())?;
        assert!(Arc::ptr_eq(&target.context, &context));
        assert_eq!(target.context.client.parallelism().chunk_size, Some(64));
        assert_eq!(target.snapshot_data()?, source.snapshot_data()?);
        Ok(())
    }
}
