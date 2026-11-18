//! Static port topology on the configured device, built once like the surface links.

use super::super::{SurfaceChannelDefinition, SurfaceChannelError};
use super::faces::FACE_OWNERSHIP;
use super::{CheckedPort, end_slot};
use crate::solver::Solver;
use hydr_channel::internal::ChannelCouplingEnd;
use hydr_solver::internal::{PortReachEnd, SurfacePortSpec, SurfacePorts};
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;

/// Admitted ports of one coordinator, in ascending port identity order.
pub(in super::super) struct PortTopology<R: Runtime<DType = DType>> {
    /// Surface port faces. Port `p` here is port `p` of `ports`.
    pub surface: SurfacePorts<R>,
    /// Port identities in ascending order.
    pub ids: Vec<String>,
    /// Reach, end cell, width, ghost bed, and ghost side of every port.
    pub ports: Vec<CheckedPort>,
    /// Per setup reach: the port at its `[left, right]` end.
    pub reach_ports: Vec<[Option<usize>; 2]>,
}

impl<R: Runtime<DType = DType>> PortTopology<R>
where
    R::Client: IndexingOps<R>,
{
    /// Builds the port topology of `definition` on `surface`, or `None` without ports.
    pub fn new(
        definition: &SurfaceChannelDefinition,
        surface: &Solver<R>,
    ) -> Result<Option<Self>, SurfaceChannelError> {
        if definition.ports.is_empty() {
            return Ok(None);
        }
        let specs: Vec<SurfacePortSpec> = definition
            .ports
            .iter()
            .map(|port| SurfacePortSpec {
                faces: port.faces.clone(),
                ghost_bed_m: port.ghost_bed_m,
                reach_end: match port.end {
                    ChannelCouplingEnd::Left => PortReachEnd::Left,
                    ChannelCouplingEnd::Right => PortReachEnd::Right,
                },
            })
            .collect();
        let ids: Vec<String> = definition
            .ports
            .iter()
            .map(|port| definition.port_declarations[port.setup_index].id.clone())
            .collect();
        let surface = surface.coupling_ports(&specs).map_err(|error| {
            SurfaceChannelError::input(format!("{FACE_OWNERSHIP}: ports={ids:?}: {error}"))
        })?;
        let mut reach_ports = vec![[None; 2]; definition.setup.reaches.len()];
        for (position, port) in definition.ports.iter().enumerate() {
            reach_ports[port.reach_index][end_slot(port.end)] = Some(position);
        }
        Ok(Some(Self {
            surface,
            ids,
            ports: definition.ports.clone(),
            reach_ports,
        }))
    }
}

impl<R: Runtime<DType = DType>> PortTopology<R> {
    /// Identity of the port at `end` of setup reach `reach`, if any.
    pub fn port_at(&self, reach: usize, end: ChannelCouplingEnd) -> Option<&str> {
        let position = self.reach_ports.get(reach)?[end_slot(end)]?;
        self.ids.get(position).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::forcing::{BoundForcing, PortedEnds};
    use super::super::super::{
        SurfaceChannelBatch, SurfaceChannelError, SurfaceChannelForcingInterval,
        SurfaceChannelReachForcing,
    };
    use super::super::fixture::{ALPHA, BETA, TestResult, case, contains, four_ports};
    use super::*;
    use crate::channel::ChannelBoundary;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn build(ports: bool) -> TestResult<SurfaceChannelBatch<CpuRuntime>> {
        let (setup, initial) = case(&[])?;
        let ports = if ports { four_ports() } else { Vec::new() };
        let definition = SurfaceChannelDefinition::with_ports(setup, ports)?;
        Ok(SurfaceChannelBatch::from_definition(
            CpuClient::new(CpuDevice::new()),
            definition,
            initial,
        )?)
    }

    #[test]
    fn the_batch_builds_surface_ports_once_in_identity_order() -> TestResult {
        let batch = build(true)?;
        let Some(topology) = &batch.context.ports else {
            return Err("a ported batch holds a port topology".into());
        };
        assert_eq!(
            topology.ids,
            [
                "p-alpha-left",
                "p-alpha-right",
                "p-beta-left",
                "p-beta-right"
            ]
        );
        assert_eq!(topology.surface.len(), 4);
        assert_eq!(topology.surface.face_lanes(), 8);
        for (position, port) in topology.ports.iter().enumerate() {
            assert_eq!(
                topology.surface.faces(position),
                Some(port.faces.as_slice())
            );
            assert_eq!(topology.surface.ghost_left(position), Some(port.ghost_left));
        }
        assert_eq!(
            topology.reach_ports,
            [[Some(0), Some(1)], [Some(2), Some(3)]]
        );
        assert_eq!(
            batch.context.port_at(1, ChannelCouplingEnd::Right),
            Some("p-beta-right")
        );
        assert!(build(false)?.context.ports.is_none());
        Ok(())
    }

    #[test]
    fn caller_forcing_at_a_ported_end_rejects() -> TestResult {
        let batch = build(true)?;
        let reach = |id: &str| SurfaceChannelReachForcing {
            reach_id: id.into(),
            left: ChannelBoundary::Wall,
            right: ChannelBoundary::Wall,
            lateral_inputs: Vec::new(),
        };
        let forcing = SurfaceChannelForcingInterval {
            start_s: 0.0,
            end_s: 1.0,
            reaches: vec![reach(ALPHA), reach(BETA)],
        };
        let ids = [ALPHA, BETA];
        let bind = |batch: &SurfaceChannelBatch<CpuRuntime>, ported| {
            BoundForcing::bind(&batch.context, &ids, (0.0, 1.0), &forcing, ported)
        };
        let Err(SurfaceChannelError::Input(message)) = bind(&batch, PortedEnds::Caller) else {
            return Err("a caller policy at a ported end must reject".into());
        };
        contains(
            &message,
            &[
                "port-endpoint-policy",
                "reach=\"alpha\" left=Wall sits at port=\"p-alpha-left\"",
            ],
        )?;
        // The coordinator owns every ported end, so no end keeps a caller policy.
        let owned = bind(&batch, PortedEnds::Coordinator)?;
        assert!(
            owned
                .reaches
                .iter()
                .all(|reach| reach.left.is_none() && reach.right.is_none())
        );
        // Without ports every end takes its caller policy.
        let unported = build(false)?;
        let caller = bind(&unported, PortedEnds::Coordinator)?;
        assert!(
            caller
                .reaches
                .iter()
                .all(|reach| reach.left.is_some() && reach.right.is_some())
        );
        Ok(())
    }
}
