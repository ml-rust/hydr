use super::check::{self, require};
use super::{
    CHANNEL_CONTRACT_VERSION, CHANNEL_SNAPSHOT_SCHEMA, ChannelIntervalSnapshotData,
    ChannelSnapshotData,
};
use crate::channel::flow::coupling::ChannelCouplingTrial;
use crate::channel::flow::{
    ChannelError, ChannelInitialState, ChannelReach, conserved_point::check_parts,
};
use hydr_core::contract::package_identity;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
    },
};
use std::sync::Arc;

/// Binding path that replays the final interval of an imported package.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Replay {
    /// Standalone stepping: caller forcing rejects `surface-exchange:` inputs.
    Standalone,
    /// Coordinator binding at the exact recorded endpoint.
    Coordinator,
}

impl<R> ChannelReach<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Check a complete persisted package and replay its final interval before restoration.
    ///
    /// Historical authenticity requires application-owned hashes or authenticated history.
    pub fn import_snapshot(&self, data: &ChannelSnapshotData) -> Result<Self, ChannelError>
    where
        R::Client: TensorOps<R>
            + CorrectlyRoundedOps<R>
            + NextRepresentableOps<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedMappedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
        self.import_with(data, Replay::Standalone)
    }

    /// Imports a package whose final interval a coordinator calculated.
    ///
    /// Runs every [`ChannelReach::import_snapshot`] check. The final interval replays through
    /// the coordinator binding path at its exact endpoint, so `surface-exchange:` inputs bind.
    #[doc(hidden)]
    pub fn coupling_import_snapshot(&self, data: &ChannelSnapshotData) -> Result<Self, ChannelError>
    where
        R::Client: TensorOps<R>
            + CorrectlyRoundedOps<R>
            + NextRepresentableOps<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedMappedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
        self.import_with(data, Replay::Coordinator)
    }

    fn import_with(&self, data: &ChannelSnapshotData, replay: Replay) -> Result<Self, ChannelError>
    where
        R::Client: TensorOps<R>
            + CorrectlyRoundedOps<R>
            + NextRepresentableOps<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedMappedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
        package_identity(
            "channel",
            &data.schema,
            CHANNEL_SNAPSHOT_SCHEMA,
            &data.contract,
            CHANNEL_CONTRACT_VERSION,
        )
        .map_err(ChannelError::input)?;
        require(
            check::definition(self.definition(), &data.definition)
                && check::initial(&self.context.original_initial, &data.initial),
            "geometry, config, or original initial-state bits differ from the destination",
        )?;
        require(
            data.time_s.is_finite()
                && data.time_s >= data.initial.time_s
                && (data.accepted_steps == 0) == data.last_interval.is_none(),
            "accepted count, time, and final interval presence disagree",
        )?;
        let surveyed = self.context.surveyed();
        check_parts("volume", &data.volume_m3, &data.volume_low_m3, surveyed)?;
        check_parts(
            "momentum",
            &data.longitudinal_momentum_m4_per_s,
            &data.longitudinal_momentum_low_m4_per_s,
            surveyed,
        )?;
        let original_storage = self.accepted.ledger.initial_storage_m3;
        if data.accepted_steps == 0 {
            require(
                data.initial.time_s.to_bits() == data.time_s.to_bits()
                    && check::values(&data.initial.volume_m3, &data.volume_m3)
                    && check::values(
                        &data.initial.longitudinal_momentum_m4_per_s,
                        &data.longitudinal_momentum_m4_per_s,
                    ),
                "zero-step state differs from the original epoch",
            )?;
            let zeros = vec![0.0; data.volume_m3.len()];
            require(
                check::values(&data.volume_low_m3, &zeros)
                    && check::values(&data.longitudinal_momentum_low_m4_per_s, &zeros),
                "zero-step low parts differ from the original epoch",
            )?;
            let original = self.checked_start(&data.initial, &zeros, &zeros)?;
            require(
                check::ledger(&original.accepted.ledger, &data.ledger)
                    && check::lateral_state(
                        &original
                            .context
                            .lateral
                            .state_data(&original.context.client, &original.accepted.lateral)?,
                        &data.lateral,
                    ),
                "zero-step ledger or lateral counters differ from the original zero history",
            )?;
            return Ok(original);
        }
        let interval = data.last_interval.as_ref().ok_or_else(|| {
            ChannelError::input(
                "channel checkpoint lacks a final interval: supply complete continuation data",
            )
        })?;
        require(
            interval.previous_accepted_steps.checked_add(1) == Some(data.accepted_steps)
                && interval.previous_time_s.is_finite()
                && interval.previous_time_s >= data.initial.time_s
                && interval.previous_time_s < data.time_s,
            "final interval does not continue the previous accepted count and time",
        )?;
        let previous_initial = ChannelInitialState {
            time_s: interval.previous_time_s,
            volume_m3: interval.previous_volume_m3.clone(),
            longitudinal_momentum_m4_per_s: interval
                .previous_longitudinal_momentum_m4_per_s
                .clone(),
        };
        let starts_at_origin = check::initial(&data.initial, &previous_initial);
        let mut previous = self.checked_start(
            &previous_initial,
            &interval.previous_volume_low_m3,
            &interval.previous_longitudinal_momentum_low_m4_per_s,
        )?;
        let initial_lateral = self
            .context
            .lateral
            .state_data(&self.context.client, &previous.accepted.lateral)?;
        previous.accepted.lateral = self
            .context
            .lateral
            .restore_state(&self.context.client, &interval.previous_lateral)?;
        let totals = self
            .context
            .lateral
            .totals(&self.context.client, &previous.accepted.lateral)?;
        require(
            check::values(
                &[
                    totals.incoming_m3,
                    totals.withdrawal_m3,
                    totals.counter_abs_bound_m3,
                    totals.assembly_abs_bound_m3,
                ],
                &[
                    interval.previous_ledger.lateral_in_m3,
                    interval.previous_ledger.lateral_out_m3,
                    interval.previous_ledger.lateral_counter_abs_bound_m3,
                    interval.previous_ledger.lateral_assembly_abs_bound_m3,
                ],
            ),
            "previous lateral totals differ from the authoritative boundary counters",
        )?;
        check::ledger_admission(
            &interval.previous_ledger,
            previous.accepted.ledger.storage_m3,
            original_storage,
            self.definition(),
        )?;
        if interval.previous_accepted_steps == 0 {
            require(
                starts_at_origin
                    && check::ledger(&previous.accepted.ledger, &interval.previous_ledger)
                    && check::lateral_state(&initial_lateral, &interval.previous_lateral),
                "first interval does not start from the original state and zero counters",
            )?;
        } else {
            require(
                interval.previous_time_s > data.initial.time_s,
                "positive previous count requires time after the original epoch",
            )?;
        }
        previous.accepted.steps = interval.previous_accepted_steps;
        previous.accepted.ledger = interval.previous_ledger.clone();
        let replayed = match replay {
            Replay::Standalone => {
                previous
                    .step(interval.requested_end_s, &interval.forcing)?
                    .next
            }
            Replay::Coordinator => previous.replay_coupled(interval)?,
        };
        require(
            replayed.accepted.time.to_bits() == data.time_s.to_bits()
                && replayed.accepted.steps == data.accepted_steps
                && check::point(
                    &replayed.accepted.volume.high.try_to_vec::<f64>()?,
                    &replayed.accepted.volume.low.try_to_vec::<f64>()?,
                    &data.volume_m3,
                    &data.volume_low_m3,
                )
                && check::point(
                    &replayed.accepted.momentum.high.try_to_vec::<f64>()?,
                    &replayed.accepted.momentum.low.try_to_vec::<f64>()?,
                    &data.longitudinal_momentum_m4_per_s,
                    &data.longitudinal_momentum_low_m4_per_s,
                )
                && check::ledger(&replayed.accepted.ledger, &data.ledger),
            "accepted state or cumulative accounting differs from the replayed final interval",
        )?;
        require(
            check::lateral_state(
                &self
                    .context
                    .lateral
                    .state_data(&self.context.client, &replayed.accepted.lateral)?,
                &data.lateral,
            ),
            "accepted lateral counters or bounds differ from the replayed interval",
        )?;
        let diagnostics = replayed.last_step()?.ok_or_else(|| {
            ChannelError::input(
                "replayed channel interval lacks a receipt: inspect continuation arithmetic",
            )
        })?;
        if interval.forcing.lateral_inputs.iter().any(|input| {
            matches!(
                input.amount(),
                crate::forcing::WaterAmount::IntervalVolume { .. }
            )
        }) {
            require(
                replayed.time().to_bits() == interval.requested_end_s.to_bits()
                    && interval.forcing.end_s.to_bits() == interval.requested_end_s.to_bits()
                    && diagnostics.retries == 0,
                "exact lateral volumes do not retain their requested span without internal retries",
            )?;
        }
        require(
            check::diagnostics(&diagnostics, &interval.diagnostics)
                && check::forcing(&replayed.accepted.last.as_ref().ok_or_else(|| {
                    ChannelError::input("replayed channel interval lacks original forcing: inspect continuation state")
                })?.forcing.project(&self.context.client)?, &interval.forcing),
            "final receipt operands or numerical bounds differ from the replayed interval",
        )?;
        Ok(replayed)
    }

    /// Recalculates a coordinator interval from this start state at its exact endpoint.
    fn replay_coupled(&self, interval: &ChannelIntervalSnapshotData) -> Result<Self, ChannelError>
    where
        R::Client: TensorOps<R>
            + CorrectlyRoundedOps<R>
            + NextRepresentableOps<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedMappedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
        let forcing = &interval.forcing;
        let bound = self.bind_coupling_host(
            forcing.start_s,
            forcing.end_s,
            Some(&forcing.left),
            Some(&forcing.right),
            &forcing.lateral_inputs,
        )?;
        match self.trial_coupled(interval.requested_end_s, &bound, None, None)? {
            ChannelCouplingTrial::Accepted(next) => Ok(*next),
            ChannelCouplingTrial::Shorter {
                suggested_end_s,
                reason,
            } => Err(ChannelError::input(format!(
                "replayed coordinator interval needs end={suggested_end_s}: {reason}: supply an unchanged complete same-build package"
            ))),
        }
    }

    fn checked_start(
        &self,
        initial: &ChannelInitialState,
        volume_low: &[f64],
        momentum_low: &[f64],
    ) -> Result<Self, ChannelError>
    where
        R::Client: TensorOps<R> + NextRepresentableOps<R>,
    {
        Self::from_parts(Arc::clone(&self.context), initial, volume_low, momentum_low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::snapshot::fixture::{advanced, walled_reach};
    use crate::channel::{ChannelConfig, ChannelReachDefinition, RectangularChannelCell};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn cold_restore_uploads_with_a_separately_constructed_cpu_client() -> Result<(), ChannelError> {
        let definition = ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "exact rectangular device fixture".into(),
            lateral_bindings: Vec::new(),
            geometry: crate::channel::ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 1.0,
                    width_m: 2.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                }],
            },
            config: ChannelConfig::default(),
        };
        let initial = ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![2.0],
            longitudinal_momentum_m4_per_s: vec![0.0],
        };
        let source = ChannelReach::<CpuRuntime>::new(
            CpuClient::new(CpuDevice::new()),
            definition.clone(),
            initial.clone(),
        )?;
        let destination =
            ChannelReach::<CpuRuntime>::new(CpuClient::new(CpuDevice::new()), definition, initial)?;
        let context = Arc::clone(&destination.context);
        let restored = destination.import_snapshot(&source.snapshot_data()?)?;
        assert!(Arc::ptr_eq(&restored.context, &context));
        assert_eq!(restored.snapshot_data()?, source.snapshot_data()?);
        Ok(())
    }

    #[test]
    fn triangular_restoration_rejects_a_different_reconstruction() -> Result<(), ChannelError> {
        use crate::channel::{
            ChannelGeometry, ChannelTriangularReconstruction, TriangularChannelCell,
        };
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let definition = ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "triangular reconstruction continuation".into(),
            geometry: ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v: 1.0,
                right_bank_slope_h_per_v: 1.0,
                bed_elevation_m: 0.0,
                cells: vec![TriangularChannelCell {
                    id: "cell".into(),
                    length_m: 2.0,
                    maximum_depth_m: 3.0,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: vec![],
            config: ChannelConfig::default(),
        };
        let initial = ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![2.0],
            longitudinal_momentum_m4_per_s: vec![0.0],
        };
        let source =
            ChannelReach::<CpuRuntime>::new(client.clone(), definition.clone(), initial.clone())?;
        let mut changed = definition;
        changed.config.triangular_reconstruction =
            ChannelTriangularReconstruction::FirstOrderReference;
        assert!(!source.definition().same_bits(&changed));
        let destination = ChannelReach::<CpuRuntime>::new(client, changed, initial)?;
        assert!(destination.restore(&source.snapshot()).is_err());
        assert!(
            destination
                .import_snapshot(&source.snapshot_data()?)
                .is_err()
        );
        assert!(source.restore(&destination.snapshot()).is_err());
        assert!(
            source
                .import_snapshot(&destination.snapshot_data()?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn unnormalized_low_is_rejected_on_import() -> Result<(), ChannelError> {
        let origin = walled_reach()?;
        let data = advanced(&origin)?.snapshot_data()?;
        let mut current = data.clone();
        current.volume_low_m3[0] = 0.5;
        assert_eq!(
            origin
                .import_snapshot(&current)
                .err()
                .and_then(|e| e.cell()),
            Some(0)
        );
        let mut previous = data.clone();
        if let Some(interval) = previous.last_interval.as_mut() {
            interval.previous_longitudinal_momentum_low_m4_per_s[0] = f64::NAN;
        }
        assert!(origin.import_snapshot(&previous).is_err());
        let mut zero_step = origin.snapshot_data()?;
        zero_step.longitudinal_momentum_low_m4_per_s[0] = 3.0;
        assert!(origin.import_snapshot(&zero_step).is_err());
        Ok(())
    }

    #[test]
    fn rectangular_reach_rejects_a_nonzero_normalized_low() -> Result<(), ChannelError> {
        let origin = walled_reach()?;
        let data = advanced(&origin)?.snapshot_data()?;
        let mut current = data.clone();
        current.volume_low_m3[0] = 1e-20;
        assert_eq!(
            origin
                .import_snapshot(&current)
                .err()
                .and_then(|e| e.cell()),
            Some(0)
        );
        let mut previous = data;
        if let Some(interval) = previous.last_interval.as_mut() {
            interval.previous_volume_low_m3[0] = 1e-20;
        }
        assert!(origin.import_snapshot(&previous).is_err());
        Ok(())
    }
}
