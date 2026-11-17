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
        ];
        assert_ids(&all.map(|r| r.id()));
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
