//! Identity of each component that declares capabilities.

/// Component that declares capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ComponentId {
    /// 2D shallow-water surface solver.
    SurfaceSolver,
    /// Channel reach with a rectangular cross-section.
    ChannelReachRectangular,
    /// Channel reach with a triangular cross-section.
    ChannelReachTriangular,
    /// Channel reach with a surveyed cross-section.
    ChannelReachSurveyed,
    /// Channel reaches joined through pools.
    ChannelPools,
    /// Single level-pool storage.
    LevelPool,
    /// Level pools joined through connections.
    CoupledLevelPools,
    /// Runoff routed into pools.
    RunoffPools,
    /// Reservoir storage.
    Reservoir,
    /// Subcatchment runoff generator.
    Subcatchment,
    /// Soil infiltration model.
    Soil,
    /// Control logic that sets structure operation.
    OperatingControl,
    /// Actuator that moves a gate opening.
    GateActuator,
    /// Gate under operating control.
    ControlledGate,
    /// Pump under operating control.
    ControlledPump,
    /// 2D surface coupled to channel reaches through lateral bank weirs.
    SurfaceChannels,
    /// Closed-conduit culvert barrel geometry for circular and box barrels.
    CulvertBarrel,
    /// HDS-5 inlet-control headwater of circular and box culverts.
    CulvertInlet,
    /// Outlet-control headwater and hydraulic jump of circular and box culverts.
    CulvertOutlet,
    /// HDS-5 culvert structure law: discharge from the heads at both ends.
    Culvert,
    /// Pinned EPA SWMM v5.2.4 piped drainage network solver.
    SwmmDrainage,
    /// 2D surface coupled to a pinned EPA SWMM v5.2.4 piped drainage network.
    SurfaceDrainage,
}

impl ComponentId {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::SurfaceSolver => "surface-solver",
            Self::ChannelReachRectangular => "channel-reach-rectangular",
            Self::ChannelReachTriangular => "channel-reach-triangular",
            Self::ChannelReachSurveyed => "channel-reach-surveyed",
            Self::ChannelPools => "channel-pools",
            Self::LevelPool => "level-pool",
            Self::CoupledLevelPools => "coupled-level-pools",
            Self::RunoffPools => "runoff-pools",
            Self::Reservoir => "reservoir",
            Self::Subcatchment => "subcatchment",
            Self::Soil => "soil",
            Self::OperatingControl => "operating-control",
            Self::GateActuator => "gate-actuator",
            Self::ControlledGate => "controlled-gate",
            Self::ControlledPump => "controlled-pump",
            Self::SurfaceChannels => "surface-channels",
            Self::CulvertBarrel => "culvert-barrel",
            Self::CulvertInlet => "culvert-inlet",
            Self::CulvertOutlet => "culvert-outlet",
            Self::Culvert => "culvert",
            Self::SwmmDrainage => "swmm-drainage",
            Self::SurfaceDrainage => "surface-drainage",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::id_check::assert_ids;

    #[test]
    fn ids_are_unique_kebab_case() {
        let all = [
            ComponentId::SurfaceSolver,
            ComponentId::ChannelReachRectangular,
            ComponentId::ChannelReachTriangular,
            ComponentId::ChannelReachSurveyed,
            ComponentId::ChannelPools,
            ComponentId::LevelPool,
            ComponentId::CoupledLevelPools,
            ComponentId::RunoffPools,
            ComponentId::Reservoir,
            ComponentId::Subcatchment,
            ComponentId::Soil,
            ComponentId::OperatingControl,
            ComponentId::GateActuator,
            ComponentId::ControlledGate,
            ComponentId::ControlledPump,
            ComponentId::SurfaceChannels,
            ComponentId::CulvertBarrel,
            ComponentId::CulvertInlet,
            ComponentId::CulvertOutlet,
            ComponentId::Culvert,
            ComponentId::SwmmDrainage,
            ComponentId::SurfaceDrainage,
        ];
        assert_ids(&all.map(|c| c.id()));
    }
}
