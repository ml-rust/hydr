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
        ];
        assert_ids(&all.map(|c| c.id()));
    }
}
