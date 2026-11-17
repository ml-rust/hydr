//! Local water sources and sinks.

/// Local source or sink that a component admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SourceKind {
    /// Rain on the surface.
    Rain,
    /// Runoff from impervious land.
    ImperviousLand,
    /// Infiltration by the modified Green-Ampt method.
    ModifiedGreenAmpt,
    /// Evaporation loss.
    Evaporation,
    /// Prescribed withdrawal.
    Withdrawal,
    /// Lateral inflow along a reach.
    LateralInflow,
    /// Lateral withdrawal along a reach.
    LateralWithdrawal,
    /// Volume exchanged with a coupled component.
    ExchangeVolume,
    /// Rain directly on a pool.
    LocalPoolRain,
    /// Local loss from a pool.
    LocalPoolLoss,
}

impl SourceKind {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::Rain => "rain",
            Self::ImperviousLand => "impervious-land",
            Self::ModifiedGreenAmpt => "modified-green-ampt",
            Self::Evaporation => "evaporation",
            Self::Withdrawal => "withdrawal",
            Self::LateralInflow => "lateral-inflow",
            Self::LateralWithdrawal => "lateral-withdrawal",
            Self::ExchangeVolume => "exchange-volume",
            Self::LocalPoolRain => "local-pool-rain",
            Self::LocalPoolLoss => "local-pool-loss",
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
            SourceKind::Rain,
            SourceKind::ImperviousLand,
            SourceKind::ModifiedGreenAmpt,
            SourceKind::Evaporation,
            SourceKind::Withdrawal,
            SourceKind::LateralInflow,
            SourceKind::LateralWithdrawal,
            SourceKind::ExchangeVolume,
            SourceKind::LocalPoolRain,
            SourceKind::LocalPoolLoss,
        ];
        assert_ids(&all.map(|s| s.id()));
    }
}
