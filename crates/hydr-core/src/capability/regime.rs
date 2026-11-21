//! Flow regimes and where a component supports them.

use super::BoundaryKind;

/// Flow regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Regime {
    /// Wet flow with Froude number below one.
    WetSubcritical,
    /// Flow with Froude number above one.
    Supercritical,
    /// Flow at Froude number one.
    Critical,
    /// Dry cells or sections.
    Dry,
    /// Moving front between wet and dry.
    WetDryFront,
    /// Flow that reverses direction within one branch.
    ReversalInBranch,
    /// Weir flow with only the upstream side above the crest.
    FreeWeirFlow,
    /// Weir flow with both sides above the crest, including equal heads.
    SubmergedWeirFlow,
    /// Both sides of a weir at or below the crest, with no flow.
    BelowCrest,
    /// Orifice flow with both sides above the crown.
    SubmergedOrificeFlow,
    /// Gate flow with the downstream side at or below the sill.
    FreeGateAperture,
    /// Gate flow with the downstream side between the sill and the opening top.
    MixedGateAperture,
    /// Gate flow with both sides above the opening top.
    SubmergedGateAperture,
    /// Gate at zero opening, with no flow.
    GateClosed,
    /// Running pump on its curve, with positive flow.
    PumpCurve,
    /// Running pump at a lift where its curve gives zero flow.
    PumpShutoff,
    /// Stopped pump that isolates its two sides, with no flow.
    PumpIsolated,
    /// Culvert flow whose inlet-control headwater governs.
    CulvertInletControl,
    /// Culvert flow whose outlet-control headwater governs.
    CulvertOutletControl,
}

impl Regime {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::WetSubcritical => "wet-subcritical",
            Self::Supercritical => "supercritical",
            Self::Critical => "critical",
            Self::Dry => "dry",
            Self::WetDryFront => "wet-dry-front",
            Self::ReversalInBranch => "reversal-in-branch",
            Self::FreeWeirFlow => "free-weir-flow",
            Self::SubmergedWeirFlow => "submerged-weir-flow",
            Self::BelowCrest => "below-crest",
            Self::SubmergedOrificeFlow => "submerged-orifice-flow",
            Self::FreeGateAperture => "free-gate-aperture",
            Self::MixedGateAperture => "mixed-gate-aperture",
            Self::SubmergedGateAperture => "submerged-gate-aperture",
            Self::GateClosed => "gate-closed",
            Self::PumpCurve => "pump-curve",
            Self::PumpShutoff => "pump-shutoff",
            Self::PumpIsolated => "pump-isolated",
            Self::CulvertInletControl => "culvert-inlet-control",
            Self::CulvertOutletControl => "culvert-outlet-control",
        }
    }
}

/// Part of a component where a regime claim holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegimeScope {
    /// At boundaries of this kind.
    Boundary(BoundaryKind),
    /// In the interior, away from boundaries.
    Interior,
}

impl RegimeScope {
    /// Returns the stable kebab-case id of the scope kind. Persisted decisions store it, so it
    /// never changes. A `Boundary` id omits the boundary kind, which has its own id.
    pub fn id(&self) -> &'static str {
        match self {
            Self::Boundary(_) => "boundary",
            Self::Interior => "interior",
        }
    }
}

/// Regime that a component supports within one scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegimeSupport {
    /// Where the claim holds.
    pub scope: RegimeScope,
    /// Supported regime.
    pub regime: Regime,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::id_check::assert_ids;

    #[test]
    fn regime_ids_are_unique_kebab_case() {
        let all = [
            Regime::WetSubcritical,
            Regime::Supercritical,
            Regime::Critical,
            Regime::Dry,
            Regime::WetDryFront,
            Regime::ReversalInBranch,
            Regime::FreeWeirFlow,
            Regime::SubmergedWeirFlow,
            Regime::BelowCrest,
            Regime::SubmergedOrificeFlow,
            Regime::FreeGateAperture,
            Regime::MixedGateAperture,
            Regime::SubmergedGateAperture,
            Regime::GateClosed,
            Regime::PumpCurve,
            Regime::PumpShutoff,
            Regime::PumpIsolated,
            Regime::CulvertInletControl,
            Regime::CulvertOutletControl,
        ];
        assert_ids(&all.map(|r| r.id()));
    }

    #[test]
    fn structure_regime_ids_are_frozen() {
        for (regime, id) in [
            (Regime::SubmergedOrificeFlow, "submerged-orifice-flow"),
            (Regime::FreeGateAperture, "free-gate-aperture"),
            (Regime::MixedGateAperture, "mixed-gate-aperture"),
            (Regime::SubmergedGateAperture, "submerged-gate-aperture"),
            (Regime::GateClosed, "gate-closed"),
            (Regime::PumpCurve, "pump-curve"),
            (Regime::PumpShutoff, "pump-shutoff"),
            (Regime::PumpIsolated, "pump-isolated"),
            (Regime::CulvertInletControl, "culvert-inlet-control"),
            (Regime::CulvertOutletControl, "culvert-outlet-control"),
        ] {
            assert_eq!(regime.id(), id);
        }
    }

    #[test]
    fn scope_ids_are_unique_kebab_case() {
        let all = [
            RegimeScope::Boundary(BoundaryKind::Wall),
            RegimeScope::Interior,
        ];
        assert_ids(&all.map(|s| s.id()));
    }
}
