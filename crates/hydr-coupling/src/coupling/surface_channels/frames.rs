//! Child engines and coordinator state of one checkpoint frame.

use super::SurfaceChannelError;
use super::accounting::within;
use super::batch::SurfaceChannelBatch;
use super::context::Context;
use super::identity::{self, Check};
use super::operating::OperatingChildren;
use super::pool_ends::PoolEndCounters;
use super::pools::PoolChild;
use super::ports::{PortCounters, PortTopology};
use super::runoff::RunoffCounters;
use super::snapshot::SurfaceChannelCounterData;
use super::state::LinkCounters;
use super::{SurfaceChannelDefinition, SurfaceChannelInitialState, SurfaceChannelOperatingData};
use crate::channel::{ChannelInitialState, ChannelReach, ChannelSnapshotData};
use crate::runoff::SubcatchmentBatch;
use crate::solver::{SnapshotData, Solver};
use crate::storage::{LevelPoolBatch, LevelPoolSnapshotData};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
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

/// Restored children of one checkpoint frame, before its counters.
pub(super) struct Frame<R: Runtime<DType = DType>> {
    pub surface: Solver<R>,
    /// Reaches in setup order.
    pub reaches: Vec<ChannelReach<R>>,
    /// Pool storage without its inflow ledgers, or `None` without pools.
    pub pools: Option<LevelPoolBatch<R>>,
    pub operating: Option<OperatingChildren<R>>,
    /// Runoff child, or `None` without active runoff.
    pub runoff: Option<SubcatchmentBatch<R>>,
}

/// Pool and operating packages of one frame.
pub(super) type NodePackages<'a> = (
    Option<&'a LevelPoolSnapshotData>,
    Option<&'a SurfaceChannelOperatingData>,
);

/// Restored pool storage and operating children of one frame.
pub(super) type NodeChildren<R> = (Option<LevelPoolBatch<R>>, Option<OperatingChildren<R>>);

/// Restores the pool and operating packages of one frame into copies of `bases`.
///
/// Each base holds the immutable model the package must match: the destination's children,
/// or fresh children of a resumed definition.
pub(super) fn node_children<R: Runtime<DType = DType>>(
    context: &Context<R>,
    prefix: &str,
    (pool_base, operating_base): (Option<&LevelPoolBatch<R>>, Option<&OperatingChildren<R>>),
    (pools, operating): NodePackages<'_>,
) -> Result<NodeChildren<R>, SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let detail = "presence differs from the destination children";
    let pools = match (pool_base, pools) {
        (Some(base), Some(data)) => {
            let restored = context.pool_topology()?.fns.restored;
            Some(restored(base, data).map_err(|error| {
                rejected(&format!("{prefix}pools"), &format!("rejects: {error}"))
            })?)
        }
        (None, None) => None,
        _ => return Err(rejected(&format!("{prefix}pools"), detail)),
    };
    let operating = match (operating_base, operating) {
        (Some(base), Some(data)) => Some(base.restored(&format!("{prefix}operating"), data)?),
        (None, None) => None,
        _ => return Err(rejected(&format!("{prefix}operating"), detail)),
    };
    Ok((pools, operating))
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
                let ports = batch.context.ports.as_ref().map(|ports| &ports.surface);
                engine.coupling_restore_data(surface, ports)?;
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
                // Port faces depend on the surface definition alone. An engine of the package's
                // definition builds them, so the resumed engine admits their port water.
                let ports = if definition.ports.is_empty() {
                    None
                } else {
                    let frame = Solver::<R>::coupling_frame((*client).clone(), surface)?;
                    PortTopology::new(definition, &frame)?
                };
                let engine = Solver::<R>::coupling_resume(
                    (*client).clone(),
                    surface,
                    surface.setup.forcing.clone(),
                    ports.as_ref().map(|ports| &ports.surface),
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
    /// Builds a batch without a last-interval receipt from restored children and `counters`.
    ///
    /// Checks the common time and count of every child, the original epoch of a zero-history
    /// frame, and the composite lifetime water balance.
    pub(super) fn assemble(
        parts: &Parts<R>,
        frame: Frame<R>,
        surface_steps: usize,
        counters: &SurfaceChannelCounterData,
        prefix: &str,
    ) -> Result<Self, SurfaceChannelError> {
        let Frame {
            surface,
            reaches,
            pools,
            operating,
            runoff,
        } = frame;
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
        if let Some(pools) = &pools {
            require(
                pools.time().to_bits() == time.to_bits() && pools.accepted_steps() == steps,
                &path("pools"),
                "time or accepted count differs from the coordinator counters",
            )?;
        }
        for (child, child_time, child_steps) in operating.iter().flat_map(|o| o.clocks()) {
            require(
                child_time.to_bits() == time.to_bits() && child_steps == steps,
                &path(child),
                "time or accepted count differs from the coordinator counters",
            )?;
        }
        require(
            steps > 0 || time.to_bits() == parts.initial.time_s.to_bits(),
            &path("counters.time_s"),
            "differs from the original epoch before the first interval",
        )?;
        let context = &parts.context;
        let pools = match pools {
            Some(batch) => {
                let external = context
                    .pool_topology()?
                    .inflows
                    .import_state(&context.client, &counters.pool_inflows, steps)
                    .map_err(|error| {
                        rejected(&path("counters.pool_inflows"), &format!("rejects: {error}"))
                    })?;
                Some(PoolChild { batch, external })
            }
            None => {
                require(
                    counters.pool_inflows.is_empty(),
                    &path("counters.pool_inflows"),
                    "holds ledgers without pools",
                )?;
                None
            }
        };
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
            pools,
            operating,
            runoff,
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

/// Device link counters and host port and pool-end counters from host values in ascending
/// identity.
///
/// Bank and structure ledgers together fill the link table. Every volume is finite and
/// nonnegative. Before the first interval every value is `+0.0`.
fn import_counters<R: Runtime<DType = DType>>(
    context: &Context<R>,
    data: &SurfaceChannelCounterData,
    prefix: &str,
) -> Result<LinkCounters<R>, SurfaceChannelError> {
    let advanced = data.accepted_steps > 0;
    let valid = |values: [f64; 3]| {
        values
            .iter()
            .all(|value| value.is_finite() && value.is_sign_positive())
            && (advanced || values.iter().all(|value| value.to_bits() == 0))
    };
    let invalid =
        "holds a negative or nonfinite value, or a nonzero value before the first interval";
    let count = context.link_count;
    let mut table = vec![[0.0; 3]; count];
    let bank: Vec<(&str, [f64; 3])> = data
        .links
        .iter()
        .map(|l| {
            let values = [
                l.surface_to_channel_m3,
                l.channel_to_surface_m3,
                l.roundoff_bound_m3,
            ];
            (l.link_id.as_str(), values)
        })
        .collect();
    let structures: Vec<(&str, [f64; 3])> = data
        .structures
        .iter()
        .map(|s| {
            let values = [s.a_to_b_m3, s.b_to_a_m3, s.roundoff_bound_m3];
            (s.structure_id.as_str(), values)
        })
        .collect();
    for (field, ledgers, positions) in [
        ("links", bank, context.bank.clone()),
        (
            "structures",
            structures,
            context.structure_positions().collect::<Vec<_>>(),
        ),
    ] {
        require(
            ledgers.len() == positions.len()
                && ledgers
                    .iter()
                    .zip(&positions)
                    .all(|((id, _), &index)| *id == context.link_ids[index]),
            &format!("{prefix}counters.{field}"),
            "differs from the links of its kind in ascending identity",
        )?;
        for (index, ((id, values), &position)) in ledgers.iter().zip(&positions).enumerate() {
            require(
                valid(*values),
                &format!("{prefix}counters.{field}[{index}] ({id:?})"),
                invalid,
            )?;
            table[position] = *values;
        }
    }
    let ports = PortCounters::import(&data.ports, context.port_ids(), advanced).map_err(|field| {
        rejected(
            &format!("{prefix}counters.ports.{field}"),
            "differs from the ports in ascending identity, holds a negative or nonfinite value, or holds a nonzero value before the first interval",
        )
    })?;
    let ids = context.pool_end_ids();
    require(
        data.pool_ends.len() == ids.len()
            && data
                .pool_ends
                .iter()
                .zip(&ids)
                .all(|(ledger, id)| ledger.pool_end_id == *id),
        &format!("{prefix}counters.pool_ends"),
        "differs from the pool ends in ascending identity",
    )?;
    let mut pool_ends = PoolEndCounters::zero(ids.len());
    for (index, ledger) in data.pool_ends.iter().enumerate() {
        let values = [
            ledger.reach_to_pool_m3,
            ledger.pool_to_reach_m3,
            ledger.roundoff_bound_m3,
        ];
        require(
            valid(values),
            &format!(
                "{prefix}counters.pool_ends[{index}] ({:?})",
                ledger.pool_end_id
            ),
            invalid,
        )?;
        pool_ends.to_pool_m3[index] = values[0];
        pool_ends.to_reach_m3[index] = values[1];
        pool_ends.roundoff_bound_m3[index] = values[2];
    }
    let device = context.client.device();
    let tensor = |lane: usize| {
        let values: Vec<f64> = table.iter().map(|values| values[lane]).collect();
        Tensor::<R>::from_slice(&values, &[values.len()], device)
    };
    Ok(LinkCounters {
        to_channel_m3: tensor(0)?,
        to_surface_m3: tensor(1)?,
        roundoff_bound_m3: tensor(2)?,
        ports,
        pool_ends,
        runoff: context
            .runoff
            .as_ref()
            .map(|topology| RunoffCounters::zero(&context.client, topology))
            .transpose()?,
    })
}
