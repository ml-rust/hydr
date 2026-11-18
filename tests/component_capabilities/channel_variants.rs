//! Public channel variants mapped to declared capability ids.
//!
//! Each mapping is an exhaustive `match`, so a new public variant fails to compile until
//! it is classified.

use crate::channel_fixture::{KINDS, Kind, TestResult, definition, exterior};
use hydr::capability::{BoundaryKind, EndpointId, SourceKind, UnsupportedReason};
use hydr::channel::{
    ChannelBoundary, ChannelError, ChannelFluxPort, ChannelLateralDirection, ChannelTensorBoundary,
    ChannelTensorFluxPort, ChannelTriangularReconstruction,
};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::error::Error;

pub const ENDPOINTS: [EndpointId; 2] = [EndpointId::ReachLeft, EndpointId::ReachRight];

/// Boundary kind of each public host endpoint policy.
pub fn boundary_kind(boundary: &ChannelBoundary) -> BoundaryKind {
    match boundary {
        ChannelBoundary::Wall => BoundaryKind::Wall,
        ChannelBoundary::Exterior { .. } => BoundaryKind::Exterior,
        ChannelBoundary::Stage { .. } => BoundaryKind::Stage,
        ChannelBoundary::Discharge { .. } => BoundaryKind::Discharge,
        ChannelBoundary::QuiescentReservoir { .. } => BoundaryKind::QuiescentReservoir,
        ChannelBoundary::FluxPort(_) => BoundaryKind::FluxPort,
    }
}

/// Boundary kind of each public tensor endpoint policy.
fn tensor_boundary_kind<R: Runtime<DType = DType>>(
    boundary: &ChannelTensorBoundary<R>,
) -> BoundaryKind {
    match boundary {
        ChannelTensorBoundary::Wall => BoundaryKind::Wall,
        ChannelTensorBoundary::Exterior { .. } => BoundaryKind::Exterior,
        ChannelTensorBoundary::Stage { .. } => BoundaryKind::Stage,
        ChannelTensorBoundary::Discharge { .. } => BoundaryKind::Discharge,
        ChannelTensorBoundary::QuiescentReservoir { .. } => BoundaryKind::QuiescentReservoir,
        ChannelTensorBoundary::FluxPort(_) => BoundaryKind::FluxPort,
    }
}

/// Signal speed in m/s of the sample port. It bounds `|u| + c` of every fixture at unit depth.
pub const PORT_SPEED: f64 = 5.0;

/// A port with the given outward rates, speed and exterior invariant.
pub fn port(
    water: f64,
    momentum: f64,
    speed: f64,
    invariant: Option<f64>,
) -> Result<ChannelBoundary, ChannelError> {
    ChannelFluxPort::new(water, momentum, speed, invariant, "capability fixture port")
        .map(ChannelBoundary::FluxPort)
}

/// Option value id of each reconstruction.
pub fn reconstruction_id(reconstruction: ChannelTriangularReconstruction) -> &'static str {
    match reconstruction {
        ChannelTriangularReconstruction::FirstOrderReference => "first-order-reference",
        ChannelTriangularReconstruction::LimitedLinear => "limited-linear",
    }
}

pub const RECONSTRUCTIONS: [ChannelTriangularReconstruction; 2] = [
    ChannelTriangularReconstruction::FirstOrderReference,
    ChannelTriangularReconstruction::LimitedLinear,
];

/// Source kind of each lateral direction.
pub fn source_kind(direction: ChannelLateralDirection) -> SourceKind {
    match direction {
        ChannelLateralDirection::Inflow => SourceKind::LateralInflow,
        ChannelLateralDirection::Withdrawal => SourceKind::LateralWithdrawal,
    }
}

/// One resting-water policy of each public variant at unit depth on a zero bed.
///
/// The port moves no water. Its speed bounds every fixture's `|u| + c` at unit depth.
pub fn samples() -> Result<[ChannelBoundary; 6], ChannelError> {
    Ok([
        ChannelBoundary::Wall,
        exterior(1.0, 0.0),
        ChannelBoundary::Stage { elevation_m: 1.0 },
        ChannelBoundary::Discharge {
            discharge_m3_per_s: 0.0,
        },
        ChannelBoundary::QuiescentReservoir {
            elevation_m: 1.0,
            entrance_loss_coefficient: 0.0,
        },
        port(0.0, 0.0, PORT_SPEED, Some(PORT_SPEED))?,
    ])
}

/// The tensor form of each sample in `samples`, in the same order.
fn tensor_samples() -> Result<[ChannelTensorBoundary<crate::runtime::Runtime>; 6], Box<dyn Error>> {
    let client = crate::runtime::client();
    let value = |v: f64| Tensor::from_slice(&[v], &[1], client.device());
    Ok([
        ChannelTensorBoundary::Wall,
        ChannelTensorBoundary::Exterior {
            depth_m: value(1.0)?,
            velocity_m_per_s: value(0.0)?,
        },
        ChannelTensorBoundary::Stage {
            elevation_m: value(1.0)?,
        },
        ChannelTensorBoundary::Discharge {
            discharge_m3_per_s: value(0.0)?,
        },
        ChannelTensorBoundary::QuiescentReservoir {
            elevation_m: value(1.0)?,
            entrance_loss_coefficient: value(0.0)?,
        },
        ChannelTensorBoundary::FluxPort(Box::new(ChannelTensorFluxPort::new(
            value(0.0)?,
            value(0.0)?,
            value(PORT_SPEED)?,
            Some(value(PORT_SPEED)?),
            "capability fixture port",
        )?)),
    ])
}

/// Declared rejection that covers an unadmitted boundary kind.
pub fn refusal(kind: Kind, boundary: BoundaryKind) -> Option<&'static str> {
    match (kind, boundary) {
        (
            Kind::Surveyed,
            BoundaryKind::Stage | BoundaryKind::Discharge | BoundaryKind::QuiescentReservoir,
        ) => Some("surveyed-scalar-boundaries"),
        (Kind::Surveyed, BoundaryKind::FluxPort) => Some("surveyed-flux-port"),
        _ => None,
    }
}

#[test]
fn channel_public_variants_are_classified() -> TestResult {
    for kind in KINDS {
        let capabilities = kind.capabilities();
        assert_eq!(Kind::of(&definition(kind, 1, false).geometry), kind);
        for boundary in samples()? {
            let class = boundary_kind(&boundary);
            for id in ENDPOINTS {
                if capabilities.admits(id, class) {
                    continue;
                }
                let refused = refusal(kind, class).ok_or(format!(
                    "{kind:?}: `{}` is neither admitted nor refused",
                    class.id()
                ))?;
                let entry = capabilities
                    .unsupported_by_id(refused)
                    .ok_or(format!("{kind:?}: `{refused}` is undeclared"))?;
                assert_eq!(entry.reason, UnsupportedReason::RejectedByAdmission);
            }
        }
        for endpoint in capabilities.endpoints {
            assert!(
                ENDPOINTS.contains(&endpoint.id),
                "{kind:?} `{}`",
                endpoint.id.id()
            );
            for admitted in endpoint.admitted {
                assert!(
                    samples()?.iter().any(|b| boundary_kind(b) == *admitted),
                    "{kind:?} admits `{}` without a public variant",
                    admitted.id()
                );
            }
        }
    }
    for (host, tensor) in samples()?.iter().zip(&tensor_samples()?) {
        assert_eq!(boundary_kind(host), tensor_boundary_kind(tensor));
    }
    let option = Kind::Triangular
        .capabilities()
        .option("triangular-reconstruction")
        .ok_or("triangular reach declares no reconstruction option")?;
    for reconstruction in RECONSTRUCTIONS {
        let id = reconstruction_id(reconstruction);
        assert!(option.values.contains(&id), "`{id}` is undeclared");
    }
    Ok(())
}
