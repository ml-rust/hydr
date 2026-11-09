use super::super::batch::StoredStep;
use super::{SubcatchmentSnapshot, SubcatchmentSnapshotData};
use crate::runoff::{
    Subarea, SubcatchmentBatch, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentError,
};
use std::sync::Arc;

impl SubcatchmentBatch {
    /// Restore complete trusted state after checking immutable definition bits.
    pub fn restore(&mut self, snapshot: &SubcatchmentSnapshot) -> Result<(), SubcatchmentError> {
        self.check_snapshot_definition(snapshot.batch.definitions(), snapshot.batch.config())?;
        *self = snapshot.batch.clone();
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
            candidate.last = Some(Arc::new(StoredStep {
                start: last.start,
                end: last.end,
                step,
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
