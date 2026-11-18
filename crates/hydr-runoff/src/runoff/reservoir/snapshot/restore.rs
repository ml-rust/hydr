use numr::runtime::Device;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

use super::super::batch::StoredStep;
use super::{
    RESERVOIR_CONTRACT_VERSION, RESERVOIR_SNAPSHOT_SCHEMA, ReservoirSnapshot, ReservoirSnapshotData,
};
use crate::runoff::{ReservoirBatch, ReservoirConfig, ReservoirElement, ReservoirError};
use hydr_core::contract::package_identity;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

impl<R> ReservoirBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Restore a trusted clone after checking every immutable definition bit.
    pub fn restore(&mut self, snapshot: &ReservoirSnapshot<R>) -> Result<(), ReservoirError> {
        self.check_snapshot_definition(snapshot.batch.elements(), snapshot.batch.config())?;
        require(
            self.context
                .client
                .device()
                .is_same(snapshot.batch.context.client.device()),
            "trusted snapshot device differs from the target runtime client",
        )?;
        self.state = snapshot.batch.state.clone();
        self.time = snapshot.batch.time;
        self.ledger = snapshot.batch.ledger.clone();
        self.steps = snapshot.batch.steps;
        self.last = snapshot.batch.last.clone();
        Ok(())
    }

    /// Check and restore caller-owned complete continuation values atomically.
    ///
    /// Import checks receipt algebra without replaying the numerical root solve.
    /// It establishes continuation consistency, not historical authenticity.
    pub fn restore_data(&mut self, data: &ReservoirSnapshotData) -> Result<(), ReservoirError> {
        package_identity(
            "reservoir",
            &data.schema,
            RESERVOIR_SNAPSHOT_SCHEMA,
            &data.contract,
            RESERVOIR_CONTRACT_VERSION,
        )
        .map_err(ReservoirError::input)?;
        self.check_snapshot_definition(&data.elements, &data.config)?;
        let mut candidate = Self::new(
            self.context.client.clone(),
            data.elements.clone(),
            data.states.clone(),
            data.time,
            data.config.clone(),
        )?;
        require(
            (data.accepted_steps == 0) == data.last_step.is_none(),
            "accepted step count and final interval disagree",
        )?;
        super::super::accounting::check_ledger(&candidate.context, &data.ledger)?;
        require(
            data.ledger.storage_m3 == candidate.ledger().storage_m3,
            "accepted ledger storage differs from represented state",
        )?;
        if let Some(last) = &data.last_step {
            require(
                last.start.is_finite()
                    && last.start >= 0.0
                    && last.end.is_finite()
                    && last.dt.is_finite()
                    && last.dt > 0.0
                    && last.end == data.time
                    && last.dt == last.end - last.start,
                "last interval requires finite start/end, positive duration, and accepted endpoint",
            )?;
            let step = super::receipts::import(&candidate.context, last)?;
            for (index, (state, receipt)) in data.states.iter().zip(&last.receipts).enumerate() {
                require(
                    state.depth_m.to_bits() == receipt.depth_m.to_bits(),
                    format!("element={index}: accepted state and final receipt depth bits differ"),
                )
                .map_err(|error| error.at_element(index))?;
            }
            candidate.last = Some(Arc::new(StoredStep {
                start: last.start,
                end: last.end,
                step,
            }));
            let projected = candidate.last_step()?;
            require(
                projected.as_ref() == Some(last),
                "last diagnostics differ from checked tensor receipt reductions",
            )?;
            for (name, cumulative, interval) in [
                (
                    "incoming",
                    data.ledger.incoming_volume_m3,
                    last.incoming_volume_m3,
                ),
                (
                    "export",
                    data.ledger.export_volume_m3,
                    last.export_volume_m3,
                ),
            ] {
                require(
                    cumulative >= interval,
                    format!(
                        "{name}: cumulative={cumulative}, last={interval}: counters must not decrease"
                    ),
                )?;
            }
            require(
                last.storage_after_m3 == data.ledger.storage_m3,
                "last storage and accepted ledger storage differ",
            )?;
        } else {
            require(
                data.ledger.initial_storage_m3 == data.ledger.storage_m3
                    && data.ledger.incoming_volume_m3 == 0.0
                    && data.ledger.export_volume_m3 == 0.0
                    && data.ledger.residual_m3 == 0.0,
                "initial ledger requires original storage and zero transfers/residual",
            )?;
        }
        candidate.ledger = data.ledger.clone();
        candidate.steps = data.accepted_steps;
        *self = candidate;
        Ok(())
    }

    fn check_snapshot_definition(
        &self,
        elements: &[ReservoirElement],
        config: &ReservoirConfig,
    ) -> Result<(), ReservoirError> {
        require(
            self.config().same_bits(config)
                && self.elements().len() == elements.len()
                && self.elements().iter().zip(elements).all(|(left, right)| {
                    left.id == right.id && left.parameters.same_bits(&right.parameters)
                }),
            "definition differs: restore into an identically constructed reservoir batch",
        )
    }
}

pub(super) fn require(condition: bool, context: impl Into<String>) -> Result<(), ReservoirError> {
    if condition {
        Ok(())
    } else {
        Err(ReservoirError::input(format!(
            "reservoir checkpoint: {}. Supply an unchanged complete same-build package",
            context.into()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runoff::{ReservoirParameters, ReservoirState};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn trusted_restore_preserves_destination_client() -> Result<(), ReservoirError> {
        let client = CpuClient::new(CpuDevice::new());
        let elements = vec![ReservoirElement {
            id: "reservoir".into(),
            parameters: ReservoirParameters::from_si(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)?,
        }];
        let mut target = ReservoirBatch::<CpuRuntime>::new(
            client.with_parallelism(ParallelismConfig::new(Some(1), Some(64))),
            elements.clone(),
            vec![ReservoirState { depth_m: 0.0 }],
            0.0,
            ReservoirConfig::default(),
        )?;
        let source = ReservoirBatch::<CpuRuntime>::new(
            client.with_parallelism(ParallelismConfig::new(Some(1), Some(128))),
            elements,
            vec![ReservoirState { depth_m: 0.5 }],
            2.0,
            ReservoirConfig::default(),
        )?;
        let context = Arc::clone(&target.context);
        target.restore(&source.snapshot())?;
        assert!(Arc::ptr_eq(&target.context, &context));
        assert_eq!(target.context.client.parallelism().chunk_size, Some(64));
        assert_eq!(target.snapshot_data()?, source.snapshot_data()?);
        Ok(())
    }
}
