//! Every component declaration with the identity constants it must carry.

use hydr::capability::ComponentCapabilities;

/// One component's declaration with the identity constants it must carry.
pub struct Declared {
    pub capabilities: &'static ComponentCapabilities,
    pub contract: Option<&'static str>,
    pub snapshot_schema: Option<&'static str>,
}

pub fn declarations() -> Vec<Declared> {
    let channel = |capabilities| Declared {
        capabilities,
        contract: Some(hydr::channel::CHANNEL_CONTRACT_VERSION),
        snapshot_schema: Some(hydr::channel::CHANNEL_SNAPSHOT_SCHEMA),
    };
    vec![
        Declared {
            capabilities: hydr::solver::surface_solver_capabilities(),
            contract: Some(hydr::solver::CONTRACT_VERSION),
            snapshot_schema: Some(hydr::solver::SNAPSHOT_SCHEMA),
        },
        channel(hydr::channel::rectangular_reach_capabilities()),
        channel(hydr::channel::triangular_reach_capabilities()),
        channel(hydr::channel::surveyed_reach_capabilities()),
        Declared {
            capabilities: hydr::storage::level_pool_capabilities(),
            contract: Some(hydr::storage::LEVEL_POOL_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::storage::LEVEL_POOL_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::coupling::coupled_level_pool_capabilities(),
            contract: Some(hydr::coupling::COUPLED_LEVEL_POOL_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::coupling::COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::coupling::channel_pool_capabilities(),
            contract: Some(hydr::coupling::CHANNEL_POOL_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::coupling::CHANNEL_POOL_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::soil::soil_capabilities(),
            contract: None,
            snapshot_schema: None,
        },
        Declared {
            capabilities: hydr::runoff::reservoir_capabilities(),
            contract: Some(hydr::runoff::RESERVOIR_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::runoff::RESERVOIR_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::runoff::subcatchment_capabilities(),
            contract: Some(hydr::runoff::SUBCATCHMENT_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::runoff::SUBCATCHMENT_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::coupling::runoff_pool_capabilities(),
            contract: Some(hydr::coupling::RUNOFF_POOL_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::coupling::RUNOFF_POOL_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::structures::operating_control_capabilities(),
            contract: Some(hydr::structures::OPERATING_CONTROL_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::structures::OPERATING_CONTROL_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::structures::gate_actuator_capabilities(),
            contract: Some(hydr::structures::GATE_ACTUATOR_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::structures::GATE_ACTUATOR_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::structures::controlled_gate_capabilities(),
            contract: Some(hydr::structures::CONTROLLED_GATE_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::structures::CONTROLLED_GATE_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::structures::controlled_pump_capabilities(),
            contract: Some(hydr::structures::CONTROLLED_PUMP_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::structures::CONTROLLED_PUMP_SNAPSHOT_SCHEMA),
        },
        Declared {
            capabilities: hydr::coupling::surface_channel_capabilities(),
            contract: Some(hydr::coupling::SURFACE_CHANNEL_CONTRACT_VERSION),
            snapshot_schema: Some(hydr::coupling::SURFACE_CHANNEL_SNAPSHOT_SCHEMA),
        },
    ]
}
