//! Public pool variants mapped to declared capability ids.
//!
//! Each mapping is an exhaustive `match` or a complete destructuring, so a new public variant
//! or field fails to compile until it is classified.

use crate::pools_fixture::{
    Kind, TestResult, channel_pool_case, laws, operating_points, surveyed_geometry,
};
use hydr::capability::{BoundaryKind, ComponentId, EndpointId, SourceKind, UnsupportedReason};
use hydr::channel::{ChannelGeometry, ChannelLateralDirection};
use hydr::coupling::ChannelPoolEndpoint;
use hydr::forcing::WaterAmount;
use hydr::storage::{LevelPoolInput, LevelPoolSources};
use hydr::structures::{StructureLaw, StructureOperatingPoint};

pub const CHANNEL_POOL_ENDPOINTS: [ChannelPoolEndpoint; 2] =
    [ChannelPoolEndpoint::Left, ChannelPoolEndpoint::Right];

/// Endpoint id of each public channel/pool endpoint.
pub fn endpoint_id(endpoint: ChannelPoolEndpoint) -> EndpointId {
    match endpoint {
        ChannelPoolEndpoint::Left => EndpointId::ReachLeft,
        ChannelPoolEndpoint::Right => EndpointId::ReachRight,
    }
}

/// Boundary kind of a law under an operating point, or the declared rejection id.
pub fn structure_kind(
    law: &StructureLaw,
    operating_point: StructureOperatingPoint,
) -> Result<BoundaryKind, &'static str> {
    use StructureOperatingPoint::{GateOpening, Passive, PumpRunning, PumpStoppedIsolated};
    const MISMATCH: &str = "mismatched-structure-operation";
    match law {
        StructureLaw::HydrostaticRectangularGate(_) => match operating_point {
            GateOpening { .. } => Ok(BoundaryKind::GateFixedOpening),
            Passive | PumpStoppedIsolated | PumpRunning { .. } => Err(MISMATCH),
        },
        StructureLaw::Pump(_) => match operating_point {
            PumpStoppedIsolated => Ok(BoundaryKind::PumpStoppedIsolated),
            PumpRunning { .. } => Ok(BoundaryKind::PumpRunning),
            Passive | GateOpening { .. } => Err("implicit-pump-operation"),
        },
        StructureLaw::SubmergedOrifice(_) => match operating_point {
            Passive => Ok(BoundaryKind::SubmergedOrifice),
            PumpStoppedIsolated | PumpRunning { .. } | GateOpening { .. } => Err(MISMATCH),
        },
        StructureLaw::SuppressedRectangularWeir(_) => match operating_point {
            Passive => Ok(BoundaryKind::SuppressedRectangularWeir),
            PumpStoppedIsolated | PumpRunning { .. } | GateOpening { .. } => Err(MISMATCH),
        },
    }
}

/// Message fragment of the operation rejection for `law`.
pub fn operation_fragment(law: &StructureLaw) -> &'static str {
    match law {
        StructureLaw::HydrostaticRectangularGate(_) => {
            "gate requires an explicit identity-bound opening operation"
        }
        StructureLaw::Pump(_) => "pump requires explicit running or stopped-isolated operation",
        StructureLaw::SubmergedOrifice(_) | StructureLaw::SuppressedRectangularWeir(_) => {
            "passive law requires an explicit Passive operation"
        }
    }
}

/// Boundary kind of each prescribed inflow value.
pub fn inflow_kind(value: WaterAmount) -> BoundaryKind {
    match value {
        WaterAmount::Rate { .. } => BoundaryKind::PrescribedInflowRate,
        WaterAmount::IntervalVolume { .. } => BoundaryKind::PrescribedInflowVolume,
    }
}

/// One sample of each prescribed inflow value.
pub fn inflow_samples() -> [WaterAmount; 2] {
    [
        WaterAmount::Rate { rate_m3_per_s: 0.5 },
        WaterAmount::IntervalVolume { volume_m3: 0.125 },
    ]
}

/// Source kind of each local pool rate.
pub fn pool_source_kinds(sources: LevelPoolSources) -> [(SourceKind, f64); 3] {
    let LevelPoolSources {
        rain_rate_m_per_s,
        potential_evaporation_rate_m_per_s,
        bottom_loss_rate_m3_per_s,
    } = sources;
    [
        (SourceKind::LocalPoolRain, rain_rate_m_per_s),
        (
            SourceKind::LocalPoolLoss,
            potential_evaporation_rate_m_per_s,
        ),
        (SourceKind::LocalPoolLoss, bottom_loss_rate_m3_per_s),
    ]
}

/// Source kind of each original exchange volume.
pub fn exchange_kinds(input: LevelPoolInput) -> [(SourceKind, f64); 2] {
    let LevelPoolInput {
        incoming_volume_m3,
        outgoing_volume_m3,
    } = input;
    [
        (SourceKind::ExchangeVolume, incoming_volume_m3),
        (SourceKind::ExchangeVolume, outgoing_volume_m3),
    ]
}

/// Source kind of each channel lateral direction.
pub fn lateral_kind(direction: ChannelLateralDirection) -> SourceKind {
    match direction {
        ChannelLateralDirection::Inflow => SourceKind::LateralInflow,
        ChannelLateralDirection::Withdrawal => SourceKind::LateralWithdrawal,
    }
}

/// Nested reach of a channel geometry.
pub fn geometry_component(geometry: &ChannelGeometry) -> ComponentId {
    match geometry {
        ChannelGeometry::Rectangular { .. } => ComponentId::ChannelReachRectangular,
        ChannelGeometry::PrismaticTriangular { .. } => ComponentId::ChannelReachTriangular,
        ChannelGeometry::PrismaticSurveyed { .. } => ComponentId::ChannelReachSurveyed,
    }
}

/// Asserts that `id` names a declared `RejectedByAdmission` entry of `kind`.
fn assert_refused(kind: Kind, id: &str) -> TestResult {
    let entry = kind
        .capabilities()
        .unsupported_by_id(id)
        .ok_or(format!("{kind:?}: `{id}` is undeclared"))?;
    assert_eq!(entry.reason, UnsupportedReason::RejectedByAdmission, "{id}");
    Ok(())
}

#[test]
fn pool_public_variants_are_classified() -> TestResult {
    let coupled = Kind::Coupled.capabilities();
    let mut structures = Vec::new();
    for law in laws()? {
        for operating_point in operating_points() {
            match structure_kind(&law, operating_point) {
                Ok(boundary) => {
                    assert!(
                        coupled.admits(EndpointId::PoolConnection, boundary),
                        "`{}` is not admitted",
                        boundary.id()
                    );
                    structures.push(boundary);
                }
                Err(id) => assert_refused(Kind::Coupled, id)?,
            }
        }
    }
    let inflows = inflow_samples().map(inflow_kind);
    for (endpoint, reachable) in [
        (EndpointId::PoolConnection, structures.as_slice()),
        (EndpointId::Lateral, inflows.as_slice()),
    ] {
        let declared = coupled
            .endpoint(endpoint)
            .ok_or(format!("coupled pools declare no `{}`", endpoint.id()))?;
        for admitted in declared.admitted {
            assert!(
                reachable.contains(admitted),
                "`{}` has no public variant",
                admitted.id()
            );
        }
        for boundary in reachable {
            assert!(declared.admits(*boundary), "`{}` undeclared", boundary.id());
        }
    }
    assert_eq!(coupled.endpoints.len(), 2, "coupled endpoints");

    let channel_pools = Kind::ChannelPools.capabilities();
    let ids = CHANNEL_POOL_ENDPOINTS.map(endpoint_id);
    for id in ids {
        assert!(channel_pools.admits(id, BoundaryKind::PoolConnection));
    }
    for endpoint in channel_pools.endpoints {
        assert!(ids.contains(&endpoint.id), "`{}`", endpoint.id.id());
    }
    let (rectangular, _) = channel_pool_case(false, ChannelPoolEndpoint::Right)?;
    let (triangular, _) = channel_pool_case(true, ChannelPoolEndpoint::Right)?;
    for geometry in [
        rectangular.reaches[0].definition.geometry.clone(),
        triangular.reaches[0].definition.geometry.clone(),
        surveyed_geometry("datum"),
    ] {
        assert!(
            channel_pools
                .nested
                .contains(&geometry_component(&geometry))
        );
    }
    for direction in [
        ChannelLateralDirection::Inflow,
        ChannelLateralDirection::Withdrawal,
    ] {
        assert!(channel_pools.sources.contains(&lateral_kind(direction)));
    }

    let rates = LevelPoolSources {
        rain_rate_m_per_s: 1.0,
        potential_evaporation_rate_m_per_s: 1.0,
        bottom_loss_rate_m3_per_s: 1.0,
    };
    for (source, _) in pool_source_kinds(rates) {
        for kind in [Kind::LevelPool, Kind::Coupled] {
            assert!(kind.capabilities().sources.contains(&source), "{kind:?}");
        }
    }
    let exchange = LevelPoolInput {
        incoming_volume_m3: 1.0,
        outgoing_volume_m3: 1.0,
    };
    for (source, _) in exchange_kinds(exchange) {
        assert!(Kind::LevelPool.capabilities().sources.contains(&source));
    }
    Ok(())
}
