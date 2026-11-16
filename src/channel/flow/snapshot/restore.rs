use super::check::{self, require};
use super::{ChannelSnapshot, ChannelSnapshotData};
use crate::channel::flow::{
    ChannelError, ChannelInitialState, ChannelReach, conserved_point::check_parts,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
    },
};
use std::sync::Arc;

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
    /// Restore trusted same-device state while retaining this reach's configured client.
    pub fn restore(&self, snapshot: &ChannelSnapshot<R>) -> Result<Self, ChannelError> {
        require(
            check::definition(self.definition(), snapshot.reach.definition())
                && check::initial(
                    &self.context.original_initial,
                    &snapshot.reach.context.original_initial,
                ),
            "geometry, config, or original initial-state bits differ from the destination",
        )?;
        require(
            self.context
                .client
                .device()
                .is_same(snapshot.reach.context.client.device()),
            "trusted tensor device differs from the destination client",
        )?;
        Ok(Self {
            context: Arc::clone(&self.context),
            accepted: snapshot.reach.accepted.clone(),
        })
    }

    /// Check a complete persisted package and replay its final interval before restoration.
    ///
    /// Historical authenticity requires application-owned hashes or authenticated history.
    pub fn import_snapshot(&self, data: &ChannelSnapshotData) -> Result<Self, ChannelError>
    where
        R::Client: TensorOps<R>
            + NextRepresentableOps<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedMappedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
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
        let replayed = previous
            .step(interval.requested_end_s, &interval.forcing)?
            .next;
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
