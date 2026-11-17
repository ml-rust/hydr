//! Snapshot capture, host projection, and forecast continuation of the surface forcing.

use super::SurfaceChannelError;
use super::batch::SurfaceChannelBatch;
use super::snapshot::{
    SURFACE_CHANNEL_CONTRACT_VERSION, SURFACE_CHANNEL_SNAPSHOT_SCHEMA, SurfaceChannelCounterData,
    SurfaceChannelIntervalSnapshotData, SurfaceChannelPreviousData, SurfaceChannelSetupData,
    SurfaceChannelSnapshot, SurfaceChannelSnapshotData,
};
use super::state::{LinkCounters, PreviousFrame, StoredStep};
use crate::channel::{ChannelError, ChannelSnapshotData};
use crate::solver::{ForcingSchedule, Solver};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, HypotOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};
use std::sync::Arc;

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
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
        + BatchedNewtonAlgorithms<R>,
{
    /// Retains complete native continuation without host readback.
    pub fn snapshot(&self) -> SurfaceChannelSnapshot<R> {
        SurfaceChannelSnapshot {
            definition: Arc::clone(&self.definition),
            context: Arc::clone(&self.context),
            initial: Arc::clone(&self.initial),
            time_s: self.time_s,
            surface: self.surface.snapshot(),
            reaches: self.reaches.iter().map(|reach| reach.snapshot()).collect(),
            accepted_steps: self.accepted_steps,
            counters: self.counters.clone(),
            last: self.last.as_ref().map(Arc::clone),
        }
    }

    /// Projects complete continuation values for caller-owned persistence.
    ///
    /// # Errors
    /// Returns child projection and runtime transfer errors.
    pub fn snapshot_data(&self) -> Result<SurfaceChannelSnapshotData, SurfaceChannelError> {
        self.snapshot().data()
    }

    /// Returns this batch with `forcing` spliced into the surface schedule after the accepted time.
    ///
    /// The surface engine applies [`Solver::continue_with_forcing`]. Every interval that starts
    /// before the accepted time stays frozen bit for bit. The start frame of the last interval
    /// continues under the same window, so checkpoints stay importable.
    /// The result's `setup().surface.forcing` holds the surface engine's new window.
    /// This batch stays unchanged.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Surface`] for every surface continuation rejection.
    pub fn continue_surface_forcing(
        &self,
        forcing: ForcingSchedule,
    ) -> Result<Self, SurfaceChannelError> {
        let mut surface = self.surface.coupling_clone();
        surface.continue_with_forcing(forcing)?;
        let last = match &self.last {
            None => None,
            Some(step) => {
                let mut previous = step.previous.surface.coupling_clone();
                previous.continue_with_forcing(surface.coupling_forcing().clone())?;
                Some(Arc::new(step.with_previous_surface(previous)))
            }
        };
        let window = surface.coupling_forcing().clone();
        Ok(Self {
            definition: Arc::new(self.definition.with_surface_forcing(window)),
            context: Arc::clone(&self.context),
            initial: Arc::clone(&self.initial),
            time_s: self.time_s,
            surface,
            reaches: self.reaches.clone(),
            accepted_steps: self.accepted_steps,
            counters: self.counters.clone(),
            last,
        })
    }
}

impl<R: Runtime<DType = DType>> StoredStep<R> {
    /// This receipt with `surface` as the start-frame surface.
    fn with_previous_surface(&self, surface: Solver<R>) -> Self {
        Self {
            previous: PreviousFrame {
                surface,
                reaches: self.previous.reaches.clone(),
                counters: self.previous.counters.clone(),
            },
            forcing: self.forcing.clone(),
            start_s: self.start_s,
            end_s: self.end_s,
            requested_end_s: self.requested_end_s,
            retries: self.retries,
            iterations: self.iterations,
            head_residual_m: self.head_residual_m,
            transfer_residual_m3: self.transfer_residual_m3,
            totals: self.totals,
            signed_m3: self.signed_m3.clone(),
            weir_m3: self.weir_m3.clone(),
            factor: self.factor.clone(),
            regime: self.regime.clone(),
            midpoint_surface_m: self.midpoint_surface_m.clone(),
            midpoint_channel_m: self.midpoint_channel_m.clone(),
        }
    }
}

impl<R: Runtime<DType = DType>> SurfaceChannelSnapshot<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Projects complete owned values, including the start frame of the last interval.
    ///
    /// # Errors
    /// Returns child projection and runtime transfer errors.
    pub fn data(&self) -> Result<SurfaceChannelSnapshotData, SurfaceChannelError> {
        let setup = self.definition.setup();
        let link_ids = &self.context.link_ids;
        let reaches = self
            .reaches
            .iter()
            .map(|reach| reach.data())
            .collect::<Vec<_>>();
        let last_interval = match &self.last {
            None => None,
            Some(step) => {
                let steps = self.accepted_steps.checked_sub(1).ok_or_else(|| {
                    SurfaceChannelError::Numerical(
                        "a last interval without an accepted interval: inspect accepted state"
                            .into(),
                    )
                })?;
                let previous = step
                    .previous
                    .reaches
                    .iter()
                    .map(|reach| reach.snapshot_data())
                    .collect::<Vec<_>>();
                Some(SurfaceChannelIntervalSnapshotData {
                    previous: SurfaceChannelPreviousData {
                        surface: step.previous.surface.snapshot().data()?,
                        reaches: named(setup, previous)?,
                        counters: counter_data(
                            &step.previous.counters,
                            link_ids,
                            step.start_s,
                            steps,
                        )?,
                    },
                    forcing: step.forcing.clone(),
                    diagnostics: step.diagnostics(link_ids)?,
                })
            }
        };
        Ok(SurfaceChannelSnapshotData {
            schema: SURFACE_CHANNEL_SNAPSHOT_SCHEMA.into(),
            contract: SURFACE_CHANNEL_CONTRACT_VERSION.into(),
            setup: SurfaceChannelSetupData::of(setup),
            initial: (*self.initial).clone(),
            surface: self.surface.data()?,
            reaches: named(setup, reaches)?,
            counters: counter_data(&self.counters, link_ids, self.time_s, self.accepted_steps)?,
            last_interval,
        })
    }
}

/// Pairs each projected reach package with its setup identity.
fn named(
    setup: &super::SurfaceChannelSetup,
    packages: Vec<Result<ChannelSnapshotData, ChannelError>>,
) -> Result<Vec<(String, ChannelSnapshotData)>, SurfaceChannelError> {
    setup
        .reaches
        .iter()
        .zip(packages)
        .map(|(reach, package)| {
            package
                .map(|data| (reach.id.clone(), data))
                .map_err(|source| SurfaceChannelError::Channel {
                    reach_id: reach.id.clone(),
                    source,
                })
        })
        .collect()
}

/// Host counters at `time_s` after `accepted_steps` intervals.
fn counter_data<R: Runtime<DType = DType>>(
    counters: &LinkCounters<R>,
    link_ids: &[String],
    time_s: f64,
    accepted_steps: usize,
) -> Result<SurfaceChannelCounterData, SurfaceChannelError> {
    Ok(SurfaceChannelCounterData {
        time_s,
        accepted_steps,
        links: counters.ledgers(link_ids)?,
    })
}
