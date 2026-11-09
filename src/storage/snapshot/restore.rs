use super::{LevelPoolSnapshot, LevelPoolSnapshotData};
use crate::cpu::level_pool::{CpuPoolContext, CpuPoolState};
use crate::storage::batch::StoredStep;
use crate::storage::{LevelPoolBatch, LevelPoolError};
use std::sync::Arc;

impl LevelPoolBatch {
    /// Restore trusted complete state after exact immutable definition checks.
    pub fn restore(&mut self, snapshot: &LevelPoolSnapshot) -> Result<(), LevelPoolError> {
        self.compatible(
            snapshot.batch.elements(),
            snapshot.batch.config(),
            snapshot.batch.source_policies(),
        )?;
        *self = snapshot.batch.clone();
        Ok(())
    }
    /// Check complete caller-owned state before atomic replacement, without replaying historical intervals.
    pub fn restore_data(&mut self, data: &LevelPoolSnapshotData) -> Result<(), LevelPoolError> {
        self.compatible(
            &data.elements,
            &data.config,
            data.source_policies.as_deref(),
        )?;
        let mut candidate = if let Some(policies) = &data.source_policies {
            Self::new_with_sources(
                data.elements.clone(),
                data.states.clone(),
                data.time,
                data.config.clone(),
                policies.clone(),
            )?
        } else {
            Self::new(
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
        crate::numerics::storage::accounting::check_state(context, &state, data.accepted_steps)?;
        crate::numerics::storage::curves::evaluate(context, &state.initial_volume_m3)?;
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
            let before = CpuPoolState {
                sources: super::sources::before(context, data, receipt)?,
                volume_m3: tensor(|r| r.old_volume_m3)?,
                initial_volume_m3: state.initial_volume_m3.clone(),
                cumulative_incoming_volume_m3: tensor(|r| r.cumulative_incoming_before_m3)?,
                cumulative_outgoing_volume_m3: tensor(|r| r.cumulative_outgoing_before_m3)?,
                incoming_roundoff_bound_m3: tensor(|r| r.incoming_bound_before_m3)?,
                outgoing_roundoff_bound_m3: tensor(|r| r.outgoing_bound_before_m3)?,
            };
            crate::numerics::storage::accounting::check_state(
                context,
                &before,
                data.accepted_steps - 1,
            )?;
            crate::numerics::storage::curves::evaluate(context, &before.volume_m3)?;
            crate::storage::accounting::totals(context, &before)?;
            let incoming = tensor(|r| r.incoming_volume_m3)?;
            let outgoing = tensor(|r| r.outgoing_volume_m3)?;
            let (next, step) = if let Some(input) = super::sources::inputs(context, receipt)? {
                crate::numerics::storage::advance_with_sources(
                    context,
                    &before,
                    &incoming,
                    &outgoing,
                    data.accepted_steps,
                    receipt.dt,
                    &input,
                )?
            } else {
                crate::numerics::storage::advance(
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

fn import_state(
    context: &CpuPoolContext,
    data: &LevelPoolSnapshotData,
) -> Result<CpuPoolState, LevelPoolError> {
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
    Ok(CpuPoolState {
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
