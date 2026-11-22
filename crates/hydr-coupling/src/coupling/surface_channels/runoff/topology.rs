//! Static delivery operands: lane gathers, normalized weights, and receiving bindings.

use super::admission::RunoffTarget;
use super::surface::SurfaceRunoff;
use crate::channel::{ChannelTensorLateralInput, ChannelTensorWaterAmount};
use crate::coupling::surface_channels::{
    SurfaceChannelDefinition, SurfaceChannelError, SurfaceChannelNode,
};
use crate::runoff::Subarea;
use crate::solver::Solver;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// One runoff lateral binding of a reach: its identity and its delivery lane.
#[derive(Clone, Debug)]
pub(in super::super) struct RunoffLateral {
    pub boundary_id: String,
    pub lane: usize,
}

/// Delivery operands built once from the checked runoff.
///
/// Delivery lanes follow the checked runoff lanes. Outlet entries follow the runoff child's
/// flattened `[definition, subarea]` export.
pub(in super::super) struct RunoffTopology<R: Runtime<DType = DType>> {
    /// Outlet entry of every delivery lane, I64 `[N]`.
    pub gather: Tensor<R>,
    /// Normalized weight of every delivery lane, `[N]`, or `None` when every weight is one.
    pub weights: Option<Tensor<R>>,
    /// Delivery lanes of every outlet entry, in lane order. An entry without lanes exports zero.
    pub entries: Vec<Vec<usize>>,
    /// Runoff lateral bindings of every setup reach, in binding order.
    pub reaches: Vec<Vec<RunoffLateral>>,
    /// Position of each setup reach's first runoff binding among its coordinated bindings.
    pub offsets: Vec<usize>,
    /// Delivery lane of every runoff pool binding, in binding order.
    pub pool_lanes: Vec<usize>,
    /// [`Self::pool_lanes`] as I64 `[Np]`, or `None` without pool lanes.
    pub pool_gather: Option<Tensor<R>>,
    /// Receiving node of every delivery lane.
    pub targets: Vec<RunoffTarget>,
    /// Receiving node of every delivery lane, as receipts name it.
    pub nodes: Vec<SurfaceChannelNode>,
    /// Outlet identity of every delivery lane.
    pub outlets: Vec<String>,
    /// Subarea of every delivery lane.
    pub subareas: Vec<Subarea>,
    /// Free-outlet invert of every delivery lane, in metres.
    pub inverts: Vec<f64>,
    /// Surface delivery operands, or `None` without a surface target.
    pub surface: Option<SurfaceRunoff<R>>,
    /// Zero volume and velocity operand, `[1]`.
    pub zero: Tensor<R>,
}

impl<R: Runtime<DType = DType>> RunoffTopology<R> {
    /// Delivery operands of `definition` on `client` over `surface`, or `None` without active
    /// runoff.
    ///
    /// Active runoff without a potential subarea holds no delivery lane.
    pub fn new(
        client: &R::Client,
        definition: &SurfaceChannelDefinition,
        surface: &Solver<R>,
    ) -> Result<Option<Self>, SurfaceChannelError> {
        let Some(runoff) = &definition.runoff else {
            return Ok(None);
        };
        if runoff.plan.active_definitions().is_empty() {
            return Ok(None);
        }
        let device = client.device();
        let indices = |values: &[usize]| -> Result<Tensor<R>, SurfaceChannelError> {
            let values = values
                .iter()
                .map(|&v| {
                    i64::try_from(v)
                        .map_err(|_| SurfaceChannelError::input("runoff lanes exceed I64 indexing"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Tensor::<R>::from_slice(&values, &[values.len()], device).map_err(Into::into)
        };
        let setup = definition.setup();
        let declared = setup
            .runoff
            .as_ref()
            .map_or(&[][..], |r| r.outlets.as_slice());
        let entry_count = 3 * runoff.plan.active_definitions().len();
        let mut entries = vec![Vec::new(); entry_count];
        let mut gather = Vec::with_capacity(runoff.lanes.len());
        let mut reaches = vec![Vec::new(); setup.reaches.len()];
        let mut pool_lanes = Vec::new();
        let mut inverts = Vec::with_capacity(runoff.lanes.len());
        for (index, lane) in runoff.lanes.iter().enumerate() {
            let entry = 3 * lane.definition + lane.subarea.index();
            entries
                .get_mut(entry)
                .ok_or_else(|| {
                    SurfaceChannelError::Numerical(
                        "a runoff lane names no active definition".into(),
                    )
                })?
                .push(index);
            gather.push(entry);
            inverts.push(
                declared
                    .get(lane.outlet)
                    .map_or(f64::NAN, |outlet| outlet.free_outlet.invert_elevation_m),
            );
            match (lane.target, &lane.binding) {
                (RunoffTarget::Channel { reach, .. }, Some(boundary_id)) => {
                    if let Some(bindings) = reaches.get_mut(reach) {
                        bindings.push(RunoffLateral {
                            boundary_id: boundary_id.clone(),
                            lane: index,
                        });
                    }
                }
                (RunoffTarget::Pool { .. }, Some(_)) => pool_lanes.push(index),
                _ => {}
            }
        }
        let offsets = definition
            .coordinated
            .iter()
            .zip(&reaches)
            .map(|(coordinated, own)| coordinated.lateral_bindings.len().saturating_sub(own.len()))
            .collect();
        let weights: Vec<f64> = runoff.lanes.iter().map(|lane| lane.weight).collect();
        let unit = weights.iter().all(|w| w.to_bits() == 1.0_f64.to_bits());
        let pool_ids = definition
            .pools
            .as_ref()
            .map_or(&[][..], |p| p.ids.as_slice());
        let nodes = runoff
            .lanes
            .iter()
            .map(|lane| match lane.target {
                RunoffTarget::Surface { cell } => SurfaceChannelNode::SurfaceCell(cell),
                RunoffTarget::Channel { reach, cell } => SurfaceChannelNode::ChannelCell {
                    reach_id: setup
                        .reaches
                        .get(reach)
                        .map_or_else(String::new, |r| r.id.clone()),
                    cell,
                },
                RunoffTarget::Pool { position } => {
                    SurfaceChannelNode::Pool(pool_ids.get(position).cloned().unwrap_or_default())
                }
            })
            .collect();
        let targets: Vec<RunoffTarget> = runoff.lanes.iter().map(|lane| lane.target).collect();
        let surface = SurfaceRunoff::new(client, surface, definition.surface_cells(), &targets)?;
        Ok(Some(Self {
            surface,
            nodes,
            outlets: runoff
                .lanes
                .iter()
                .map(|lane| {
                    declared
                        .get(lane.outlet)
                        .map_or_else(String::new, |outlet| outlet.outlet_id.clone())
                })
                .collect(),
            subareas: runoff.lanes.iter().map(|lane| lane.subarea).collect(),
            gather: indices(&gather)?,
            weights: if unit {
                None
            } else {
                Some(Tensor::<R>::from_slice(&weights, &[weights.len()], device)?)
            },
            entries,
            reaches,
            offsets,
            pool_gather: if pool_lanes.is_empty() {
                None
            } else {
                Some(indices(&pool_lanes)?)
            },
            pool_lanes,
            targets,
            inverts,
            zero: Tensor::<R>::zeros(&[1], DType::F64, device)?,
        }))
    }

    /// Delivery lane count `N`.
    pub fn lane_count(&self) -> usize {
        self.targets.len()
    }

    /// Runoff inflow inputs of setup reach `reach`, each delivering `volume(lane)`.
    ///
    /// Runoff enters with zero longitudinal velocity.
    pub fn laterals(
        &self,
        reach: usize,
        volume: impl Fn(usize) -> Result<Tensor<R>, SurfaceChannelError>,
    ) -> Result<Vec<ChannelTensorLateralInput<R>>, SurfaceChannelError> {
        self.reaches
            .get(reach)
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .map(|binding| {
                Ok(ChannelTensorLateralInput::Inflow {
                    boundary_id: binding.boundary_id.clone(),
                    amount: ChannelTensorWaterAmount::IntervalVolume {
                        volume_m3: volume(binding.lane)?,
                    },
                    longitudinal_velocity_m_per_s: self.zero.clone(),
                })
            })
            .collect()
    }
}
