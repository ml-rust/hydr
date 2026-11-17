//! Complete source, physical-cap, and cumulative addition evidence.

/// Complete LevelPoolDemandCapReceipt values.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolDemandCapReceipt {
    /// Represented potential volume m3.
    pub potential_volume_m3: f64,
    /// Represented actual volume m3.
    pub actual_volume_m3: f64,
    /// Represented unmet volume m3.
    pub unmet_volume_m3: f64,
    /// Represented unmet witness m3.
    pub unmet_witness_m3: f64,
    /// Represented reconciliation residual m3.
    pub reconciliation_residual_m3: f64,
    /// Represented reconciliation consistency m3.
    pub reconciliation_consistency_m3: f64,
    /// Represented reconciliation bound m3.
    pub reconciliation_bound_m3: f64,
    /// Potential exceeds preceding available water. Equality records false.
    pub availability_limited: bool,
}
impl LevelPoolDemandCapReceipt {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.potential_volume_m3.to_bits() == other.potential_volume_m3.to_bits()
            && self.actual_volume_m3.to_bits() == other.actual_volume_m3.to_bits()
            && self.unmet_volume_m3.to_bits() == other.unmet_volume_m3.to_bits()
            && self.unmet_witness_m3.to_bits() == other.unmet_witness_m3.to_bits()
            && self.reconciliation_residual_m3.to_bits()
                == other.reconciliation_residual_m3.to_bits()
            && self.reconciliation_consistency_m3.to_bits()
                == other.reconciliation_consistency_m3.to_bits()
            && self.reconciliation_bound_m3.to_bits() == other.reconciliation_bound_m3.to_bits()
            && self.availability_limited == other.availability_limited
    }
}
/// Complete LevelPoolCounterReceipt values.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolCounterReceipt {
    /// Represented before m3.
    pub before_m3: f64,
    /// Represented after m3.
    pub after_m3: f64,
    /// Represented witness m3.
    pub witness_m3: f64,
    /// Represented bound before m3.
    pub bound_before_m3: f64,
    /// Represented bound after m3.
    pub bound_after_m3: f64,
}
impl LevelPoolCounterReceipt {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.before_m3.to_bits() == other.before_m3.to_bits()
            && self.after_m3.to_bits() == other.after_m3.to_bits()
            && self.witness_m3.to_bits() == other.witness_m3.to_bits()
            && self.bound_before_m3.to_bits() == other.bound_before_m3.to_bits()
            && self.bound_after_m3.to_bits() == other.bound_after_m3.to_bits()
    }
}
/// Complete LevelPoolSourceLedger values.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolSourceLedger {
    /// Represented cumulative rain volume m3.
    pub cumulative_rain_volume_m3: f64,
    /// Represented cumulative evaporation volume m3.
    pub cumulative_evaporation_volume_m3: f64,
    /// Represented cumulative bottom loss volume m3.
    pub cumulative_bottom_loss_volume_m3: f64,
    /// Represented cumulative bottom loss demand m3.
    pub cumulative_bottom_loss_demand_m3: f64,
    /// Represented rain roundoff bound m3.
    pub rain_roundoff_bound_m3: f64,
    /// Represented evaporation roundoff bound m3.
    pub evaporation_roundoff_bound_m3: f64,
    /// Represented bottom loss roundoff bound m3.
    pub bottom_loss_roundoff_bound_m3: f64,
    /// Represented bottom loss demand roundoff bound m3.
    pub bottom_loss_demand_roundoff_bound_m3: f64,
}
impl LevelPoolSourceLedger {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.cumulative_rain_volume_m3.to_bits() == other.cumulative_rain_volume_m3.to_bits()
            && self.cumulative_evaporation_volume_m3.to_bits()
                == other.cumulative_evaporation_volume_m3.to_bits()
            && self.cumulative_bottom_loss_volume_m3.to_bits()
                == other.cumulative_bottom_loss_volume_m3.to_bits()
            && self.cumulative_bottom_loss_demand_m3.to_bits()
                == other.cumulative_bottom_loss_demand_m3.to_bits()
            && self.rain_roundoff_bound_m3.to_bits() == other.rain_roundoff_bound_m3.to_bits()
            && self.evaporation_roundoff_bound_m3.to_bits()
                == other.evaporation_roundoff_bound_m3.to_bits()
            && self.bottom_loss_roundoff_bound_m3.to_bits()
                == other.bottom_loss_roundoff_bound_m3.to_bits()
            && self.bottom_loss_demand_roundoff_bound_m3.to_bits()
                == other.bottom_loss_demand_roundoff_bound_m3.to_bits()
    }
}
/// Complete LevelPoolSourceReceipt values.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolSourceReceipt {
    /// Represented frozen area m2.
    pub frozen_area_m2: f64,
    /// Represented rain rate m per s.
    pub rain_rate_m_per_s: f64,
    /// Represented potential evaporation rate m per s.
    pub potential_evaporation_rate_m_per_s: f64,
    /// Represented bottom loss rate m3 per s.
    pub bottom_loss_rate_m3_per_s: f64,
    /// Represented rain rate volume m3 per s.
    pub rain_rate_volume_m3_per_s: f64,
    /// Represented rain volume m3.
    pub rain_volume_m3: f64,
    /// Represented potential evaporation rate volume m3 per s.
    pub potential_evaporation_rate_volume_m3_per_s: f64,
    /// Represented potential evaporation volume m3.
    pub potential_evaporation_volume_m3: f64,
    /// Represented after rain volume m3.
    pub after_rain_volume_m3: f64,
    /// Represented after bottom loss volume m3.
    pub after_bottom_loss_volume_m3: f64,
    /// Represented post source volume m3.
    pub post_source_volume_m3: f64,
    /// Represented rain witness m3.
    pub rain_witness_m3: f64,
    /// Represented bottom loss witness m3.
    pub bottom_loss_witness_m3: f64,
    /// Represented evaporation witness m3.
    pub evaporation_witness_m3: f64,
    /// Represented post source reconstructed volume m3.
    pub post_source_reconstructed_volume_m3: f64,
    /// Represented post source inverse residual m3.
    pub post_source_inverse_residual_m3: f64,
    /// Frozen right-sided segment index, using the last segment at maximum volume.
    pub segment: usize,
    /// Physical post-source stage, absent at dry storage.
    pub post_source_stage_m: Option<f64>,
    /// Original bottom potential, actual/unmet cap, and reconciliation.
    pub bottom_loss_cap: LevelPoolDemandCapReceipt,
    /// Original evaporation potential, actual/unmet cap, and reconciliation.
    pub evaporation_cap: LevelPoolDemandCapReceipt,
    /// Complete source counter operands, witnesses, and bounds.
    pub counters: LevelPoolSourceCounterReceipt,
}
impl LevelPoolSourceReceipt {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.frozen_area_m2.to_bits() == other.frozen_area_m2.to_bits()
            && self.rain_rate_m_per_s.to_bits() == other.rain_rate_m_per_s.to_bits()
            && self.potential_evaporation_rate_m_per_s.to_bits()
                == other.potential_evaporation_rate_m_per_s.to_bits()
            && self.bottom_loss_rate_m3_per_s.to_bits() == other.bottom_loss_rate_m3_per_s.to_bits()
            && self.rain_rate_volume_m3_per_s.to_bits() == other.rain_rate_volume_m3_per_s.to_bits()
            && self.rain_volume_m3.to_bits() == other.rain_volume_m3.to_bits()
            && self.potential_evaporation_rate_volume_m3_per_s.to_bits()
                == other.potential_evaporation_rate_volume_m3_per_s.to_bits()
            && self.potential_evaporation_volume_m3.to_bits()
                == other.potential_evaporation_volume_m3.to_bits()
            && self.after_rain_volume_m3.to_bits() == other.after_rain_volume_m3.to_bits()
            && self.after_bottom_loss_volume_m3.to_bits()
                == other.after_bottom_loss_volume_m3.to_bits()
            && self.post_source_volume_m3.to_bits() == other.post_source_volume_m3.to_bits()
            && self.rain_witness_m3.to_bits() == other.rain_witness_m3.to_bits()
            && self.bottom_loss_witness_m3.to_bits() == other.bottom_loss_witness_m3.to_bits()
            && self.evaporation_witness_m3.to_bits() == other.evaporation_witness_m3.to_bits()
            && self.post_source_reconstructed_volume_m3.to_bits()
                == other.post_source_reconstructed_volume_m3.to_bits()
            && self.post_source_inverse_residual_m3.to_bits()
                == other.post_source_inverse_residual_m3.to_bits()
            && self.segment == other.segment
            && match (self.post_source_stage_m, other.post_source_stage_m) {
                (None, None) => true,
                (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
                _ => false,
            }
            && self.bottom_loss_cap.same_bits(&other.bottom_loss_cap)
            && self.evaporation_cap.same_bits(&other.evaporation_cap)
            && self.counters.same_bits(&other.counters)
    }
}
/// Complete source-process cumulative additions for the original interval.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolSourceCounterReceipt {
    /// Cumulative actual rain addition.
    pub rain: LevelPoolCounterReceipt,
    /// Cumulative actual evaporation addition.
    pub evaporation: LevelPoolCounterReceipt,
    /// Cumulative actual bottom-loss addition.
    pub bottom_loss: LevelPoolCounterReceipt,
    /// Cumulative potential bottom demand, excluded from physical scales.
    pub bottom_loss_demand: LevelPoolCounterReceipt,
}
impl LevelPoolSourceCounterReceipt {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.rain.same_bits(&other.rain)
            && self.evaporation.same_bits(&other.evaporation)
            && self.bottom_loss.same_bits(&other.bottom_loss)
            && self.bottom_loss_demand.same_bits(&other.bottom_loss_demand)
    }
}
