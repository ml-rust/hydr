use numr::runtime::Device;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::super::batch::StoredStep;
use super::{SubcatchmentSnapshot, SubcatchmentSnapshotData};
use crate::runoff::{
    Subarea, SubcatchmentBatch, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentError,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

impl<R> SubcatchmentBatch<R>
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
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Restore complete trusted state after checking immutable definition bits.
    pub fn restore(&mut self, snapshot: &SubcatchmentSnapshot<R>) -> Result<(), SubcatchmentError> {
        self.check_snapshot_definition(snapshot.batch.definitions(), snapshot.batch.config())?;
        require(
            self.context
                .client
                .device()
                .is_same(snapshot.batch.context.client.device()),
            "trusted snapshot device differs from the target runtime client",
        )?;
        let state = self.state.continuation_from(&snapshot.batch.state)?;
        self.state = state;
        self.time = snapshot.batch.time;
        self.ledger = snapshot.batch.ledger.clone();
        self.steps = snapshot.batch.steps;
        self.last = snapshot.batch.last.clone();
        Ok(())
    }

    /// Check complete caller-owned continuation values before replacing any accepted state.
    ///
    /// Native soil state support and receipt algebra are independent checks. No physical history is replayed.
    pub fn restore_data(
        &mut self,
        data: &SubcatchmentSnapshotData,
    ) -> Result<(), SubcatchmentError> {
        self.check_snapshot_definition(&data.definitions, &data.config)?;
        let mut candidate = Self::new(
            self.context.client.clone(),
            data.definitions.clone(),
            data.states.clone(),
            data.time,
            data.config.clone(),
        )?;
        require(
            (data.accepted_steps == 0) == data.last_step.is_none(),
            "accepted step count and final interval disagree",
        )?;
        super::super::accounting::check_ledger(
            &candidate.context,
            &data.ledger,
            data.accepted_steps,
        )?;
        require(
            data.ledger.storage_m3 == candidate.ledger().storage_m3,
            "ledger storage differs from complete represented accepted depths",
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
                "last interval requires nonnegative finite start, positive duration, and accepted endpoint",
            )?;
            let step = super::receipts::import(&candidate.context, last)?;
            for (index, (state, receipt)) in data.states.iter().zip(&last.receipts).enumerate() {
                for subarea in Subarea::ALL {
                    if let Some(source) = &receipt.subareas[subarea.index()] {
                        require(
                            state.depths_m[subarea.index()].to_bits()
                                == source.drainage.depth_m.to_bits(),
                            "accepted depth bits differ from the complete final drainage receipt",
                        )
                        .map_err(|e| e.at(index, Some(subarea)))?;
                    }
                }
            }
            let totals = super::super::accounting::aggregate_step(&candidate.context, &step)?;
            candidate.last = Some(Arc::new(StoredStep {
                start: last.start,
                end: last.end,
                step,
                totals,
            }));
            require(
                candidate.last_step()?.as_ref() == Some(last),
                "final diagnostics differ from shared checked source, transfer, or drainage reductions",
            )?;
            check_last_counters(data)?;
        } else {
            require(
                data.ledger == *candidate.ledger(),
                "initial ledger requires original storage, its checked scale lower bound, and zero lifetime counters",
            )?;
        }
        candidate.ledger = data.ledger.clone();
        candidate.steps = data.accepted_steps;
        *self = candidate;
        Ok(())
    }

    fn check_snapshot_definition(
        &self,
        definitions: &[SubcatchmentDefinition],
        config: &SubcatchmentConfig,
    ) -> Result<(), SubcatchmentError> {
        require(
            self.config().same_bits(config)
                && self.definitions().len() == definitions.len()
                && self
                    .definitions()
                    .iter()
                    .zip(definitions)
                    .all(|(left, right)| {
                        left.id == right.id && left.parameters.same_bits(&right.parameters)
                    }),
            "definition differs: restore into an identically constructed subcatchment batch",
        )
    }
}

fn check_last_counters(data: &SubcatchmentSnapshotData) -> Result<(), SubcatchmentError> {
    let Some(last) = &data.last_step else {
        return Ok(());
    };
    let ledger = &data.ledger;
    for (name, cumulative, interval) in [
        (
            "applied rain",
            ledger.applied_rain_volume_m3,
            last.applied_rain_volume_m3,
        ),
        (
            "external runon",
            ledger.external_runon_volume_m3,
            last.external_runon_volume_m3,
        ),
        (
            "infiltration",
            ledger.infiltration_volume_m3,
            last.infiltration_volume_m3,
        ),
        (
            "evaporation",
            ledger.evaporation_volume_m3,
            last.evaporation_volume_m3,
        ),
        ("outlet", ledger.outlet_volume_m3, last.outlet_volume_m3),
        (
            "soil adjustment",
            ledger.soil_water_adjustment_volume_m3,
            last.soil_water_adjustment_volume_m3,
        ),
        (
            "soil adjustment charge",
            ledger.soil_water_adjustment_abs_bound_m3,
            last.soil_water_adjustment_abs_charge_m3,
        ),
        (
            "internal traffic",
            ledger.internal_transfer_volume_m3,
            last.internal_transfer_volume_m3,
        ),
        (
            "transfer uncertainty charge",
            ledger.transfer_uncertainty_charge_m3,
            last.transfer_uncertainty_charge_m3,
        ),
        (
            "physical scale lower",
            ledger.physical_scale_lower_m3,
            last.physical_scale_lower_m3,
        ),
    ] {
        require(
            cumulative >= interval,
            format!(
                "{name}: cumulative={cumulative}, last={interval}: nonnegative counters must not decrease"
            ),
        )?;
    }
    require(
        ledger.storage_m3 == last.storage_after_m3,
        "final interval storage differs from accepted ledger storage",
    )
}

pub(super) fn require(
    condition: bool,
    context: impl Into<String>,
) -> Result<(), SubcatchmentError> {
    if condition {
        Ok(())
    } else {
        Err(SubcatchmentError::input(format!(
            "subcatchment checkpoint: {}. Supply an unchanged complete same-build package",
            context.into()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runoff::{
        SubcatchmentGeometry, SubcatchmentParameters, SubcatchmentRouting, SubcatchmentState,
    };
    use crate::soil::ModifiedGreenAmptParameters;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn trusted_restore_preserves_destination_client() -> Result<(), SubcatchmentError> {
        let client = CpuClient::new(CpuDevice::new());
        let parameters = SubcatchmentParameters::new(
            SubcatchmentGeometry {
                total_area_m2: 1.0,
                width_m: 0.0,
                slope: 0.0,
                impervious_fraction: 0.0,
                zero_storage_impervious_fraction: 0.0,
                impervious_manning: 1.0,
                pervious_manning: 1.0,
                impervious_depression_m: 0.0,
                pervious_depression_m: 0.0,
            },
            Some(ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?),
            SubcatchmentRouting::Direct,
        )?;
        let state = SubcatchmentState::initialized(&parameters)?;
        let definitions = vec![SubcatchmentDefinition {
            id: "catchment".into(),
            parameters,
        }];
        let mut target = SubcatchmentBatch::<CpuRuntime>::new(
            client.with_parallelism(ParallelismConfig::new(Some(1), Some(64))),
            definitions.clone(),
            vec![state],
            0.0,
            SubcatchmentConfig::default(),
        )?;
        let mut source_state = state;
        source_state.depths_m[2] = 0.25;
        if let Some(soil) = source_state.soil.as_mut() {
            soil.t_s = 25.0;
        }
        let source = SubcatchmentBatch::<CpuRuntime>::new(
            client.with_parallelism(ParallelismConfig::new(Some(1), Some(128))),
            definitions,
            vec![source_state],
            2.0,
            SubcatchmentConfig::default(),
        )?;
        let context = Arc::clone(&target.context);
        target.restore(&source.snapshot())?;
        assert!(Arc::ptr_eq(&target.context, &context));
        assert_eq!(target.context.client.parallelism().chunk_size, Some(64));
        for group in [
            target.context.phase1.as_ref(),
            target.context.phase2.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            assert_eq!(group.reservoir.client.parallelism().chunk_size, Some(64));
        }
        assert_eq!(target.snapshot_data()?, source.snapshot_data()?);
        Ok(())
    }
}
