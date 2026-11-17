//! Endpoints and the boundary kinds each one admits.

/// Place where a component meets a boundary or another component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EndpointId {
    /// Face on the exterior of a 2D grid.
    ExteriorFace,
    /// Left end of a channel reach.
    ReachLeft,
    /// Right end of a channel reach.
    ReachRight,
    /// Connection between two pools.
    PoolConnection,
    /// End of a hydraulic structure.
    StructureEndpoint,
    /// Outlet of a pool or subcatchment.
    Outlet,
    /// Lateral exchange along a reach or pool.
    Lateral,
}

impl EndpointId {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::ExteriorFace => "exterior-face",
            Self::ReachLeft => "reach-left",
            Self::ReachRight => "reach-right",
            Self::PoolConnection => "pool-connection",
            Self::StructureEndpoint => "structure-endpoint",
            Self::Outlet => "outlet",
            Self::Lateral => "lateral",
        }
    }
}

/// Boundary condition or link that an endpoint accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BoundaryKind {
    /// Closed wall with zero normal flux.
    Wall,
    /// Port that exchanges flux with another component.
    FluxPort,
    /// Prescribed water-surface stage.
    Stage,
    /// Prescribed discharge.
    Discharge,
    /// Exterior state supplied outside the model.
    Exterior,
    /// Transmissive outflow that copies the interior state.
    Transmissive,
    /// Reservoir at rest with a prescribed stage.
    QuiescentReservoir,
    /// Link to a pool connection.
    PoolConnection,
    /// Submerged orifice.
    SubmergedOrifice,
    /// Rectangular weir without end contractions.
    SuppressedRectangularWeir,
    /// Stopped pump that isolates its two sides.
    PumpStoppedIsolated,
    /// Running pump.
    PumpRunning,
    /// Gate held at a fixed opening.
    GateFixedOpening,
    /// Prescribed inflow rate.
    PrescribedInflowRate,
    /// Prescribed inflow volume over an interval.
    PrescribedInflowVolume,
    /// Runoff outlet that receives no feedback from downstream.
    RunoffOutletNoFeedback,
}

impl BoundaryKind {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::Wall => "wall",
            Self::FluxPort => "flux-port",
            Self::Stage => "stage",
            Self::Discharge => "discharge",
            Self::Exterior => "exterior",
            Self::Transmissive => "transmissive",
            Self::QuiescentReservoir => "quiescent-reservoir",
            Self::PoolConnection => "pool-connection",
            Self::SubmergedOrifice => "submerged-orifice",
            Self::SuppressedRectangularWeir => "suppressed-rectangular-weir",
            Self::PumpStoppedIsolated => "pump-stopped-isolated",
            Self::PumpRunning => "pump-running",
            Self::GateFixedOpening => "gate-fixed-opening",
            Self::PrescribedInflowRate => "prescribed-inflow-rate",
            Self::PrescribedInflowVolume => "prescribed-inflow-volume",
            Self::RunoffOutletNoFeedback => "runoff-outlet-no-feedback",
        }
    }
}

/// Endpoint and the boundary kinds that admission accepts there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    /// Endpoint identity.
    pub id: EndpointId,
    /// Boundary kinds accepted at the endpoint.
    pub admitted: &'static [BoundaryKind],
}

impl Endpoint {
    /// Returns `true` when the endpoint admits `kind`.
    pub fn admits(&self, kind: BoundaryKind) -> bool {
        self.admitted.contains(&kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::id_check::assert_ids;

    #[test]
    fn endpoint_ids_are_unique_kebab_case() {
        let all = [
            EndpointId::ExteriorFace,
            EndpointId::ReachLeft,
            EndpointId::ReachRight,
            EndpointId::PoolConnection,
            EndpointId::StructureEndpoint,
            EndpointId::Outlet,
            EndpointId::Lateral,
        ];
        assert_ids(&all.map(|e| e.id()));
    }

    #[test]
    fn boundary_ids_are_unique_kebab_case() {
        let all = [
            BoundaryKind::Wall,
            BoundaryKind::FluxPort,
            BoundaryKind::Stage,
            BoundaryKind::Discharge,
            BoundaryKind::Exterior,
            BoundaryKind::Transmissive,
            BoundaryKind::QuiescentReservoir,
            BoundaryKind::PoolConnection,
            BoundaryKind::SubmergedOrifice,
            BoundaryKind::SuppressedRectangularWeir,
            BoundaryKind::PumpStoppedIsolated,
            BoundaryKind::PumpRunning,
            BoundaryKind::GateFixedOpening,
            BoundaryKind::PrescribedInflowRate,
            BoundaryKind::PrescribedInflowVolume,
            BoundaryKind::RunoffOutletNoFeedback,
        ];
        assert_ids(&all.map(|b| b.id()));
    }

    #[test]
    fn endpoint_admits_only_listed_kinds() {
        let end = Endpoint {
            id: EndpointId::ReachLeft,
            admitted: &[BoundaryKind::Wall, BoundaryKind::Stage],
        };
        assert!(end.admits(BoundaryKind::Wall));
        assert!(end.admits(BoundaryKind::Stage));
        assert!(!end.admits(BoundaryKind::Discharge));
    }
}
