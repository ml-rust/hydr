//! Public surface/channel variants mapped to declared capability ids.
//!
//! Each mapping is an exhaustive `match`, so a new public variant fails to compile until it is
//! classified.

use hydr::capability::{BoundaryKind, ComponentId, Regime, RegimeScope, SourceKind};
use hydr::channel::{ChannelGeometry, ChannelLateralDirection};
use hydr::coupling::{
    SurfaceChannelError, SurfaceChannelPortRegime, SurfaceChannelWeirRegime,
    surface_channel_capabilities,
};
use hydr::structures::StructureRegime;

/// Rejection kind of each public coordinator error variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Input,
    Surface,
    Channel,
    Runtime,
    Numerical,
    Recompute,
    Unresolved,
}

/// Rejection kind and message of `error`.
pub fn variant(error: &SurfaceChannelError) -> (Variant, String) {
    match error {
        SurfaceChannelError::Input(message) => (Variant::Input, message.clone()),
        SurfaceChannelError::Surface(source) => (Variant::Surface, source.to_string()),
        SurfaceChannelError::Channel { source, .. } => (Variant::Channel, source.to_string()),
        SurfaceChannelError::Runtime(source) => (Variant::Runtime, source.to_string()),
        SurfaceChannelError::Numerical(message) => (Variant::Numerical, message.clone()),
        SurfaceChannelError::RecomputeInterval { .. } => (Variant::Recompute, error.to_string()),
        SurfaceChannelError::Unresolved { reason, .. } => (Variant::Unresolved, reason.clone()),
    }
}

/// Declared regime of each public weir regime.
///
/// Equal heads above the crest are the zero-flow limit of submerged flow.
pub fn weir_regime(regime: SurfaceChannelWeirRegime) -> Regime {
    match regime {
        SurfaceChannelWeirRegime::BelowCrest => Regime::BelowCrest,
        SurfaceChannelWeirRegime::Free => Regime::FreeWeirFlow,
        SurfaceChannelWeirRegime::Submerged | SurfaceChannelWeirRegime::EqualHeads => {
            Regime::SubmergedWeirFlow
        }
    }
}

/// Every public weir regime.
pub const WEIR_REGIMES: [SurfaceChannelWeirRegime; 4] = [
    SurfaceChannelWeirRegime::BelowCrest,
    SurfaceChannelWeirRegime::Free,
    SurfaceChannelWeirRegime::Submerged,
    SurfaceChannelWeirRegime::EqualHeads,
];

/// Declared regime of each public port regime.
///
/// Both supercritical directions are the supercritical regime. A reversal is the
/// `reversed` flag of a receipt, whatever its regime.
pub fn port_regime(regime: SurfaceChannelPortRegime) -> Regime {
    match regime {
        SurfaceChannelPortRegime::Subcritical => Regime::WetSubcritical,
        SurfaceChannelPortRegime::SupercriticalToSurface
        | SurfaceChannelPortRegime::SupercriticalToReach => Regime::Supercritical,
        SurfaceChannelPortRegime::Transcritical => Regime::Critical,
        SurfaceChannelPortRegime::WetDryFront => Regime::WetDryFront,
        SurfaceChannelPortRegime::Dry => Regime::Dry,
    }
}

/// Every public port regime.
pub const PORT_REGIMES: [SurfaceChannelPortRegime; 6] = [
    SurfaceChannelPortRegime::Subcritical,
    SurfaceChannelPortRegime::SupercriticalToSurface,
    SurfaceChannelPortRegime::SupercriticalToReach,
    SurfaceChannelPortRegime::Transcritical,
    SurfaceChannelPortRegime::WetDryFront,
    SurfaceChannelPortRegime::Dry,
];

/// Declared regime of each public structure regime, with the law volume of the interval.
///
/// A running pump at zero law volume sits at shutoff. A gate below its sill and a weir below
/// its crest carry no water and claim no flow regime. Culvert regimes claim none, because the
/// coordinator rejects culvert laws at admission.
pub fn structure_regime(regime: StructureRegime, law_volume_m3: f64) -> Option<Regime> {
    Some(match regime {
        StructureRegime::SubmergedOrifice => Regime::SubmergedOrificeFlow,
        StructureRegime::FreeWeirOverflow => Regime::FreeWeirFlow,
        StructureRegime::SubmergedWeirOverflow | StructureRegime::EqualHeads => {
            Regime::SubmergedWeirFlow
        }
        StructureRegime::BelowCrest | StructureRegime::GateBelowSill => Regime::BelowCrest,
        StructureRegime::FreeGateAperture => Regime::FreeGateAperture,
        StructureRegime::MixedGateAperture => Regime::MixedGateAperture,
        StructureRegime::SubmergedGateAperture => Regime::SubmergedGateAperture,
        StructureRegime::GateClosed => Regime::GateClosed,
        StructureRegime::PumpCurve if law_volume_m3 == 0.0 => Regime::PumpShutoff,
        StructureRegime::PumpCurve => Regime::PumpCurve,
        StructureRegime::PumpIsolated => Regime::PumpIsolated,
        StructureRegime::CulvertInletControl
        | StructureRegime::CulvertOutletControl
        | StructureRegime::CulvertControlTie
        | StructureRegime::CulvertBelowInvert
        | StructureRegime::CulvertUndeclaredEnd => return None,
    })
}

/// Every public structure regime.
pub const STRUCTURE_REGIMES: [StructureRegime; 17] = [
    StructureRegime::GateClosed,
    StructureRegime::GateBelowSill,
    StructureRegime::FreeGateAperture,
    StructureRegime::MixedGateAperture,
    StructureRegime::SubmergedGateAperture,
    StructureRegime::PumpCurve,
    StructureRegime::PumpIsolated,
    StructureRegime::SubmergedOrifice,
    StructureRegime::FreeWeirOverflow,
    StructureRegime::SubmergedWeirOverflow,
    StructureRegime::BelowCrest,
    StructureRegime::EqualHeads,
    StructureRegime::CulvertInletControl,
    StructureRegime::CulvertOutletControl,
    StructureRegime::CulvertControlTie,
    StructureRegime::CulvertBelowInvert,
    StructureRegime::CulvertUndeclaredEnd,
];

/// Nested reach of a linked geometry.
pub fn geometry_component(geometry: &ChannelGeometry) -> ComponentId {
    match geometry {
        ChannelGeometry::Rectangular { .. } => ComponentId::ChannelReachRectangular,
        ChannelGeometry::PrismaticTriangular { .. } => ComponentId::ChannelReachTriangular,
        ChannelGeometry::PrismaticSurveyed { .. } => ComponentId::ChannelReachSurveyed,
    }
}

/// Source kind of each caller lateral direction.
pub fn lateral_kind(direction: ChannelLateralDirection) -> SourceKind {
    match direction {
        ChannelLateralDirection::Inflow => SourceKind::LateralInflow,
        ChannelLateralDirection::Withdrawal => SourceKind::LateralWithdrawal,
    }
}

#[test]
fn surface_channel_public_variants_are_classified() -> Result<(), String> {
    let declared = surface_channel_capabilities();
    for regime in WEIR_REGIMES {
        let claimed = weir_regime(regime);
        if !declared.regimes.iter().any(|r| r.regime == claimed) {
            return Err(format!("{regime:?} maps to undeclared `{}`", claimed.id()));
        }
    }
    for regime in PORT_REGIMES {
        let claimed = port_regime(regime);
        if !declared.regimes.iter().any(|r| {
            r.regime == claimed && r.scope == RegimeScope::Boundary(BoundaryKind::FluxPort)
        }) {
            return Err(format!("{regime:?} maps to undeclared `{}`", claimed.id()));
        }
    }
    for claim in declared.regimes {
        let public = match claim.scope {
            RegimeScope::Boundary(BoundaryKind::SuppressedRectangularWeir) => {
                WEIR_REGIMES.iter().any(|r| weir_regime(*r) == claim.regime)
            }
            RegimeScope::Boundary(BoundaryKind::FluxPort) => {
                claim.regime == Regime::ReversalInBranch
                    || PORT_REGIMES.iter().any(|r| port_regime(*r) == claim.regime)
            }
            // A pool end runs only wet subcritical; drier or faster ends reject.
            RegimeScope::Boundary(BoundaryKind::QuiescentReservoir) => {
                claim.regime == Regime::WetSubcritical
            }
            RegimeScope::Boundary(_) => STRUCTURE_REGIMES.iter().any(|r| {
                [0.0, 1.0]
                    .iter()
                    .any(|v| structure_regime(*r, *v) == Some(claim.regime))
            }),
            _ => false,
        };
        if !public {
            return Err(format!(
                "`{}` at {:?} has no public regime",
                claim.regime.id(),
                claim.scope
            ));
        }
    }
    for direction in [
        ChannelLateralDirection::Inflow,
        ChannelLateralDirection::Withdrawal,
    ] {
        let kind = lateral_kind(direction);
        if !declared.sources.contains(&kind) {
            return Err(format!("`{}` is undeclared", kind.id()));
        }
    }
    Ok(())
}
