//! Child engines and coordinator state of one checkpoint frame.

use super::SurfaceChannelError;
use super::accounting::within;
use super::batch::SurfaceChannelBatch;
use super::context::Context;
use super::identity::{self, Check};
use super::snapshot::SurfaceChannelCounterData;
use super::state::LinkCounters;
use super::{SurfaceChannelDefinition, SurfaceChannelInitialState};
use crate::channel::{ChannelInitialState, ChannelReach, ChannelSnapshotData};
use crate::solver::{SnapshotData, Solver};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, HypotOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::sync::Arc;

/// Rejects checkpoint field `path` with `detail`.
pub(super) fn rejected(path: &str, detail: &str) -> SurfaceChannelError {
    SurfaceChannelError::input(format!(
        "checkpoint {path} {detail}: supply an unchanged complete same-build package"
    ))
}

/// Maps a bitwise comparison onto a rejection that names the differing field.
pub(super) fn checked(check: Check, detail: &str) -> Result<(), SurfaceChannelError> {
    check.map_err(|path| rejected(&path, detail))
}

fn require(condition: bool, path: &str, detail: &str) -> Result<(), SurfaceChannelError> {
    if condition {
        Ok(())
    } else {
        Err(rejected(path, detail))
    }
}

/// Shared immutable parts of every frame of one import.
pub(super) struct Parts<R: Runtime<DType = DType>> {
    pub definition: Arc<SurfaceChannelDefinition>,
    pub context: Arc<Context<R>>,
    pub initial: Arc<SurfaceChannelInitialState>,
}

/// Source of the child engines of a frame.
pub(super) enum Children<'a, R: Runtime<DType = DType>> {
    /// Restore into copies of a destination batch's children.
    Destination(&'a SurfaceChannelBatch<R>),
    /// Resume the surface on `client` and build coordinated reaches at their original states.
    Fresh {
        client: &'a R::Client,
        definition: &'a SurfaceChannelDefinition,
        /// Original reach states in setup order.
        states: Vec<ChannelInitialState>,
    },
}

impl<R: Runtime<DType = DType>> Children<'_, R>
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
        + BatchedNewtonAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    /// Restores the surface and every reach package through their coordinator paths.
    ///
    /// The caller has checked the reach count, identities, and definitions of the frame.
    pub fn children(
        &self,
        prefix: &str,
        surface: &SnapshotData,
        reaches: &[(String, ChannelSnapshotData)],
    ) -> Result<(Solver<R>, Vec<ChannelReach<R>>), SurfaceChannelError> {
        let channel = |id: &str| {
            let reach_id = id.to_string();
            move |source| SurfaceChannelError::Channel { reach_id, source }
        };
        match self {
            Self::Destination(batch) => {
                let mut engine = batch.surface.coupling_clone();
                engine.coupling_restore_data(surface)?;
                let reaches = reaches
                    .iter()
                    .zip(&batch.reaches)
                    .map(|((id, data), reach)| {
                        reach.coupling_import_snapshot(data).map_err(channel(id))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((engine, reaches))
            }
            Self::Fresh {
                client,
                definition,
                states,
            } => {
                checked(
                    identity::surface_definition(
                        prefix,
                        &surface.setup,
                        &definition.setup().surface,
                    ),
                    "differs from the current surface definition",
                )?;
                let engine = Solver::<R>::coupling_resume(
                    (*client).clone(),
                    surface,
                    surface.setup.forcing.clone(),
                )?;
                let reaches = reaches
                    .iter()
                    .zip(definition.coordinated.iter().zip(states))
                    .map(|((id, data), (coordinated, state))| {
                        ChannelReach::<R>::new_coordinated(
                            (*client).clone(),
                            coordinated.clone(),
                            state.clone(),
                        )
                        .and_then(|reach| reach.coupling_import_snapshot(data))
                        .map_err(channel(id))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((engine, reaches))
            }
        }
    }
}

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
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    /// Builds a batch without a last-interval receipt from restored children and `counters`.
    ///
    /// Checks the common time and count of every child, the original epoch of a zero-history
    /// frame, and the composite lifetime water balance.
    pub(super) fn assemble(
        parts: &Parts<R>,
        children: (Solver<R>, Vec<ChannelReach<R>>),
        surface_steps: usize,
        counters: &SurfaceChannelCounterData,
        prefix: &str,
    ) -> Result<Self, SurfaceChannelError> {
        let (surface, reaches) = children;
        let time = counters.time_s;
        let steps = counters.accepted_steps;
        let path = |field: &str| format!("{prefix}{field}");
        require(
            surface.time().to_bits() == time.to_bits() && surface_steps == steps,
            &path("surface"),
            "time or accepted count differs from the coordinator counters",
        )?;
        for (index, (reach, definition)) in reaches
            .iter()
            .zip(&parts.definition.setup().reaches)
            .enumerate()
        {
            require(
                reach.time().to_bits() == time.to_bits() && reach.accepted_steps() == steps,
                &path(&format!("reaches[{index}] ({:?})", definition.id)),
                "time or accepted count differs from the coordinator counters",
            )?;
        }
        require(
            steps > 0 || time.to_bits() == parts.initial.time_s.to_bits(),
            &path("counters.time_s"),
            "differs from the original epoch before the first interval",
        )?;
        let batch = Self {
            definition: Arc::clone(&parts.definition),
            context: Arc::clone(&parts.context),
            initial: Arc::clone(&parts.initial),
            time_s: time,
            surface,
            reaches,
            accepted_steps: steps,
            counters: import_counters(&parts.context, counters, prefix)?,
            last: None,
        };
        let balance = batch.ledger()?;
        require(
            within(balance.residual_m3, balance.bound_m3),
            &path("counters"),
            &format!(
                "leave composite water residual={} above bound={}",
                balance.residual_m3, balance.bound_m3
            ),
        )?;
        Ok(batch)
    }
}

/// Device link counters from host values in ascending link identity.
///
/// Every value is finite and nonnegative. Before the first interval every value is `+0.0`.
fn import_counters<R: Runtime<DType = DType>>(
    context: &Context<R>,
    data: &SurfaceChannelCounterData,
    prefix: &str,
) -> Result<LinkCounters<R>, SurfaceChannelError> {
    require(
        data.links.len() == context.link_ids.len()
            && data
                .links
                .iter()
                .zip(&context.link_ids)
                .all(|(link, id)| link.link_id == *id),
        &format!("{prefix}counters.links"),
        "differs from the links in ascending identity",
    )?;
    for (index, link) in data.links.iter().enumerate() {
        let values = [
            link.surface_to_channel_m3,
            link.channel_to_surface_m3,
            link.roundoff_bound_m3,
        ];
        require(
            values
                .iter()
                .all(|value| value.is_finite() && value.is_sign_positive())
                && (data.accepted_steps > 0 || values.iter().all(|value| value.to_bits() == 0)),
            &format!("{prefix}counters.links[{index}] ({:?})", link.link_id),
            "holds a negative or nonfinite value, or a nonzero value before the first interval",
        )?;
    }
    let device = context.client.device();
    let tensor = |values: Vec<f64>| Tensor::<R>::from_slice(&values, &[values.len()], device);
    Ok(LinkCounters {
        to_channel_m3: tensor(data.links.iter().map(|l| l.surface_to_channel_m3).collect())?,
        to_surface_m3: tensor(data.links.iter().map(|l| l.channel_to_surface_m3).collect())?,
        roundoff_bound_m3: tensor(data.links.iter().map(|l| l.roundoff_bound_m3).collect())?,
    })
}
