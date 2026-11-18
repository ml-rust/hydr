//! Accepted-start ghost states of every port, built once per common trial.

use super::super::SurfaceChannelError;
use super::super::context::{Context, Ops};
use crate::channel::ChannelEnd;
use crate::channel::ChannelReach;
use crate::solver::Solver;
use hydr_solver::internal::PortGhosts;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Accepted-start state of one ported reach end cell, on the host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in super::super) struct PortEnd {
    /// Depth `A / b`, in m.
    pub depth_m: f64,
    /// Velocity along the reach's outward normal, in m/s. Zero when dry.
    pub outward_velocity_m_per_s: f64,
}

/// Ghost states and host end states of every port, from the accepted reaches.
pub(in super::super) struct PortStart<R: Runtime<DType = DType>> {
    pub ghosts: PortGhosts<R>,
    /// End state of every port, in port order.
    pub ends: Vec<PortEnd>,
}

/// Outward normal sign of a reach end: `-1` left, `+1` right.
pub(in super::super) fn outward(end: ChannelEnd) -> f64 {
    match end {
        ChannelEnd::Left => -1.0,
        ChannelEnd::Right => 1.0,
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Ghost states of every port from the accepted `reaches`, or `None` without ports.
    ///
    /// Each ghost holds its reach end cell's depth `A / b` and discharge per width `Q / b`.
    pub fn port_start(
        &self,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
    ) -> Result<Option<PortStart<R>>, SurfaceChannelError> {
        let Some(topology) = &self.ports else {
            return Ok(None);
        };
        let c = &self.client;
        let mut depth = Vec::with_capacity(topology.ports.len());
        let mut discharge = Vec::with_capacity(topology.ports.len());
        let mut area = Vec::with_capacity(topology.ports.len());
        let mut flow = Vec::with_capacity(topology.ports.len());
        for port in &topology.ports {
            let state = reaches[port.reach_index]
                .coupling_end_state(port.end)
                .map_err(|source| SurfaceChannelError::Channel {
                    reach_id: reach_ids[port.reach_index].to_string(),
                    source,
                })?;
            depth.push(state.depth_m);
            discharge.push(state.discharge_per_width_m2_per_s);
            area.push(state.area_m2);
            flow.push(state.discharge_m3_per_s);
        }
        let join = |parts: &[Tensor<R>]| c.cat(&parts.iter().collect::<Vec<_>>(), 0);
        let depth = join(&depth)?;
        let discharge = join(&discharge)?;
        let ghosts = surface.coupling_port_ghosts(&topology.surface, &depth, &discharge)?;
        let count = topology.ports.len();
        let host = c
            .cat(&[&depth, &join(&area)?, &join(&flow)?], 0)?
            .try_to_vec::<f64>()?;
        let ends = topology
            .ports
            .iter()
            .enumerate()
            .map(|(index, port)| {
                let read = |offset: usize| {
                    host.get(offset * count + index).copied().ok_or_else(|| {
                        SurfaceChannelError::Numerical("port end read lacks a value".into())
                    })
                };
                let (depth_m, area_m2, discharge) = (read(0)?, read(1)?, read(2)?);
                let velocity = if area_m2 > 0.0 {
                    discharge / area_m2
                } else {
                    0.0
                };
                Ok(PortEnd {
                    depth_m,
                    outward_velocity_m_per_s: outward(port.end) * velocity,
                })
            })
            .collect::<Result<_, SurfaceChannelError>>()?;
        Ok(Some(PortStart { ghosts, ends }))
    }
}
