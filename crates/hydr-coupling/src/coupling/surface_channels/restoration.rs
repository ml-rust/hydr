//! Trusted restoration, checked package import, and resume of the coordinator.
//!
//! Import order: package identity, nested identities, setup and synthesized reach definitions,
//! then child restoration through the coordinator paths. Without a last interval the current
//! children restore directly. Otherwise the start frame restores, the interval recalculates once,
//! and the result replaces the current children after a bitwise comparison with the package.

use super::batch::{SurfaceChannelBatch, order_reach_states};
use super::context::Context;
use super::frames::{Children, Parts, checked, rejected};
use super::identity;
use super::snapshot::{
    SurfaceChannelSetupData, SurfaceChannelSnapshot, SurfaceChannelSnapshotData,
};
use super::{SurfaceChannelDefinition, SurfaceChannelError};
use crate::channel::{ChannelReach, ChannelSnapshotData};
use crate::solver::{ForcingSchedule, SnapshotData, Solver};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::sync::Arc;

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    /// Restores a trusted same-device snapshot into this batch's definition and client.
    ///
    /// The surface restores through its coordinator path, so it keeps exchange water.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] for another device, setup, or original state.
    /// Returns every surface and reach restoration rejection.
    pub fn restore(
        &self,
        snapshot: &SurfaceChannelSnapshot<R>,
    ) -> Result<Self, SurfaceChannelError> {
        if !self
            .context
            .client
            .device()
            .is_same(snapshot.context.client.device())
        {
            return Err(rejected(
                "device",
                "differs from the destination client device",
            ));
        }
        checked(
            identity::setup(
                &SurfaceChannelSetupData::of(snapshot.definition.setup()),
                &SurfaceChannelSetupData::of(self.setup()),
            ),
            "differs from the destination setup",
        )?;
        if snapshot.definition.structures != self.definition.structures {
            return Err(rejected(
                "structures",
                "differs from the destination structure links",
            ));
        }
        checked(
            identity::initial(&snapshot.initial, &self.initial),
            "differs from the destination original state",
        )?;
        let mut surface = self.surface.coupling_clone();
        surface.coupling_restore(&snapshot.surface)?;
        let reaches = self
            .reaches
            .iter()
            .zip(&snapshot.reaches)
            .zip(&self.setup().reaches)
            .map(|((reach, trusted), definition)| {
                reach
                    .restore(trusted)
                    .map_err(|source| SurfaceChannelError::Channel {
                        reach_id: definition.id.clone(),
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            definition: Arc::clone(&self.definition),
            context: Arc::clone(&self.context),
            initial: Arc::clone(&self.initial),
            time_s: snapshot.time_s,
            surface,
            reaches,
            accepted_steps: snapshot.accepted_steps,
            counters: snapshot.counters.clone(),
            last: snapshot.last.as_ref().map(Arc::clone),
        })
    }

    /// Checks a caller-owned package against this batch and recalculates its last interval.
    ///
    /// This batch's definition holds the synthesized reach definitions of the package setup,
    /// since both setups must match bit for bit. The result shares this batch's client.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] naming the first differing or foreign field.
    /// Returns every surface, reach, and replay rejection.
    pub fn import_snapshot(
        &self,
        data: &SurfaceChannelSnapshotData,
    ) -> Result<Self, SurfaceChannelError> {
        data.check_identities()
            .map_err(SurfaceChannelError::Input)?;
        checked(
            identity::setup(&data.setup, &SurfaceChannelSetupData::of(self.setup())),
            "differs from the destination setup",
        )?;
        if !self.definition.structures.is_empty() {
            return Err(rejected(
                "structures",
                "holds no structure links of the destination",
            ));
        }
        checked(
            identity::initial(&data.initial, &self.initial),
            "differs from the destination original state",
        )?;
        check_definitions(&self.definition, data)?;
        let (prefix, surface, reaches) = start_frame(data);
        let frame = Children::Destination(self).children(prefix, surface, reaches)?;
        let parts = Parts {
            definition: Arc::clone(&self.definition),
            context: Arc::clone(&self.context),
            initial: Arc::clone(&self.initial),
        };
        Self::replay(&parts, data, frame)
    }

    /// Builds a batch on `client` from a caller-owned package, then continues the surface forcing.
    ///
    /// The surface definition comes from the nested surface package. Pass
    /// `data.surface.setup.forcing.clone()` as `surface_forcing` for a plain restart.
    /// See [`SurfaceChannelBatch::continue_surface_forcing`] for the splice rule.
    ///
    /// # Errors
    /// Returns every [`SurfaceChannelDefinition::new`] and
    /// [`SurfaceChannelBatch::import_snapshot`] rejection, and every continuation rejection.
    pub fn resume(
        client: R::Client,
        data: &SurfaceChannelSnapshotData,
        surface_forcing: ForcingSchedule,
    ) -> Result<Self, SurfaceChannelError> {
        data.check_identities()
            .map_err(SurfaceChannelError::Input)?;
        let definition = Arc::new(SurfaceChannelDefinition::new(
            data.setup.with_surface(data.surface.setup.clone()),
        )?);
        check_definitions(&definition, data)?;
        let initial = &data.initial;
        if initial.surface.time.to_bits() != initial.time_s.to_bits() {
            return Err(rejected(
                "initial.surface.time",
                "differs from initial.time_s",
            ));
        }
        let states =
            order_reach_states(definition.setup(), initial.reaches.clone(), initial.time_s)?;
        let (prefix, surface, reaches) = start_frame(data);
        let frame = Children::Fresh {
            client: &client,
            definition: &definition,
            states,
        }
        .children(prefix, surface, reaches)?;
        check_epoch(&frame.0, data)?;
        let parts = Parts {
            context: Arc::new(Context::new(client.clone(), &definition, &frame.0)?),
            definition: Arc::clone(&definition),
            initial: Arc::new(initial.clone()),
        };
        Self::replay(&parts, data, frame)?.continue_surface_forcing(surface_forcing)
    }

    /// Assembles the restored start frame, then recalculates the last interval from it.
    ///
    /// Without a last interval the start frame is the current frame. Otherwise the recalculated
    /// batch becomes the current state once every result matches the package bit for bit.
    fn replay(
        parts: &Parts<R>,
        data: &SurfaceChannelSnapshotData,
        frame: (Solver<R>, Vec<ChannelReach<R>>),
    ) -> Result<Self, SurfaceChannelError> {
        let steps = data.counters.accepted_steps;
        if (steps == 0) != data.last_interval.is_none() {
            return Err(rejected(
                "last_interval",
                "presence disagrees with counters.accepted_steps",
            ));
        }
        let Some(last) = &data.last_interval else {
            return Self::assemble(
                parts,
                frame,
                data.surface.accepted_steps,
                &data.counters,
                "",
            );
        };
        let previous = Self::assemble(
            parts,
            frame,
            last.previous.surface.accepted_steps,
            &last.previous.counters,
            PREVIOUS,
        )?;
        if previous.accepted_steps.checked_add(1) != Some(steps) {
            return Err(rejected(
                "last_interval.previous.counters.accepted_steps",
                "does not precede counters.accepted_steps by one interval",
            ));
        }
        if last.forcing.start_s.to_bits() != previous.time_s.to_bits() {
            return Err(rejected(
                "last_interval.forcing.start_s",
                "differs from the start frame time",
            ));
        }
        let replayed = previous
            .trial_to(last.diagnostics.requested_end_s, &last.forcing)?
            .into_next();
        checked(
            identity::replayed(&replayed.snapshot_data()?, data),
            "differs from the recalculated last interval",
        )?;
        Ok(replayed)
    }
}

/// Path prefix of the last-interval start frame.
const PREVIOUS: &str = "last_interval.previous.";

/// The frame an import restores: the last-interval start frame, else the current frame.
fn start_frame(
    data: &SurfaceChannelSnapshotData,
) -> (
    &'static str,
    &SnapshotData,
    &[(String, ChannelSnapshotData)],
) {
    match &data.last_interval {
        Some(last) => (PREVIOUS, &last.previous.surface, &last.previous.reaches),
        None => ("", &data.surface, &data.reaches),
    }
}

/// Checks the original surface state against the epoch the nested surface package records.
///
/// The represented storage of `initial.surface` must equal `surface.ledger.initial_storage`
/// bit for bit. A zero-history package must hold `initial.surface` as its accepted state.
fn check_epoch<R: Runtime<DType = DType>>(
    surface: &Solver<R>,
    data: &SurfaceChannelSnapshotData,
) -> Result<(), SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let path = "initial.surface";
    let storage = surface
        .coupling_epoch_storage(&data.initial.surface)
        .map_err(|error| {
            rejected(
                path,
                &format!("does not fit the surface definition: {error}"),
            )
        })?;
    let recorded = data.surface.ledger.initial_storage;
    if storage.to_bits() != recorded.to_bits() {
        return Err(rejected(
            path,
            &format!("storage={storage} differs from surface.ledger.initial_storage={recorded}"),
        ));
    }
    if data.surface.accepted_steps == 0 {
        checked(
            identity::surface_epoch(path, &data.initial.surface, &data.surface.accepted),
            "differs from the accepted state of the zero-history surface package",
        )?;
    }
    Ok(())
}

/// Checks the reach packages of every frame against the synthesized reach definitions.
fn check_definitions(
    definition: &SurfaceChannelDefinition,
    data: &SurfaceChannelSnapshotData,
) -> Result<(), SurfaceChannelError> {
    let detail = "differs from the synthesized coordinator reach definition";
    checked(
        identity::reach_definitions("", definition, &data.reaches),
        detail,
    )?;
    if let Some(last) = &data.last_interval {
        checked(
            identity::reach_definitions(
                "last_interval.previous.",
                definition,
                &last.previous.reaches,
            ),
            detail,
        )?;
    }
    Ok(())
}
