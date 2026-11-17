//! Public surface/channel variants mapped to declared capability ids.
//!
//! Each mapping is an exhaustive `match`, so a new public variant fails to compile until it is
//! classified.

use hydr::capability::{ComponentId, Regime, SourceKind};
use hydr::channel::{ChannelGeometry, ChannelLateralDirection};
use hydr::coupling::{SurfaceChannelError, SurfaceChannelWeirRegime, surface_channel_capabilities};

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
    for claim in declared.regimes {
        if !WEIR_REGIMES.iter().any(|r| weir_regime(*r) == claim.regime) {
            return Err(format!("`{}` has no public weir regime", claim.regime.id()));
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
