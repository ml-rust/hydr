//! Explicit volume, witness, and independent accounting projections.

/// Complete checked LevelPoolOutput projection.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolOutput {
    /// Ordered numerical element index.
    pub element: usize,
    /// Authoritative accepted volume.
    pub volume_m3: f64,
    /// Physical stage, absent at exact dry volume.
    pub stage_m: Option<f64>,
}

/// Complete checked LevelPoolElementLedger projection.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolElementLedger {
    /// Ordered numerical element index.
    pub element: usize,
    /// Original authoritative storage.
    pub initial_volume_m3: f64,
    /// Current authoritative storage.
    pub volume_m3: f64,
    /// Cumulative original incoming volumes.
    pub cumulative_incoming_volume_m3: f64,
    /// Cumulative original outgoing volumes.
    pub cumulative_outgoing_volume_m3: f64,
    /// Conservative absolute cumulative incoming addition error.
    pub incoming_roundoff_bound_m3: f64,
    /// Conservative absolute cumulative outgoing addition error.
    pub outgoing_roundoff_bound_m3: f64,
    /// Independent cumulative physical residual.
    pub residual_m3: f64,
}

/// Complete checked LevelPoolTotals projection.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolTotals {
    /// Original authoritative storage.
    pub initial_volume_m3: f64,
    /// Current authoritative storage.
    pub volume_m3: f64,
    /// Cumulative original incoming volumes.
    pub cumulative_incoming_volume_m3: f64,
    /// Cumulative original outgoing volumes.
    pub cumulative_outgoing_volume_m3: f64,
    /// Conservative absolute cumulative incoming addition error.
    pub incoming_roundoff_bound_m3: f64,
    /// Conservative absolute cumulative outgoing addition error.
    pub outgoing_roundoff_bound_m3: f64,
    /// Independent cumulative physical residual.
    pub residual_m3: f64,
    /// Actual aggregate storage and throughput scale.
    pub physical_scale_m3: f64,
    /// Conservative aggregate reduction uncertainty, separate from physical drift.
    pub reduction_bound_m3: f64,
}

/// Complete checked LevelPoolReceipt projection.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolReceipt {
    /// Ordered numerical element index.
    pub element: usize,
    /// Original accepted donor volume.
    pub old_volume_m3: f64,
    /// Original authoritative incoming volume.
    pub incoming_volume_m3: f64,
    /// Original authoritative outgoing volume.
    pub outgoing_volume_m3: f64,
    /// Rounded donor remainder before inflow.
    pub remaining_volume_m3: f64,
    /// Accepted authoritative volume.
    pub volume_m3: f64,
    /// Signed withdrawal TwoSum low component.
    pub outgoing_witness_m3: f64,
    /// Signed incoming TwoSum low component.
    pub incoming_witness_m3: f64,
    /// Independent physical interval residual.
    pub water_residual_m3: f64,
    /// Independent residual plus both storage witnesses.
    pub witness_consistency_m3: f64,
    /// Outward arithmetic bound for witness consistency.
    pub witness_consistency_bound_m3: f64,
    /// Actual storage and exchange tolerance scale.
    pub physical_scale_m3: f64,
    /// Physical stage, absent at exact dry volume.
    pub stage_m: Option<f64>,
    /// Inverse interpolation of reported internal stage.
    pub reconstructed_volume_m3: f64,
    /// Reconstructed minus authoritative volume.
    pub inverse_residual_m3: f64,
    /// Original cumulative incoming counter.
    pub cumulative_incoming_before_m3: f64,
    /// Candidate cumulative incoming counter.
    pub cumulative_incoming_after_m3: f64,
    /// Signed cumulative incoming addition witness.
    pub cumulative_incoming_witness_m3: f64,
    /// Original cumulative outgoing counter.
    pub cumulative_outgoing_before_m3: f64,
    /// Candidate cumulative outgoing counter.
    pub cumulative_outgoing_after_m3: f64,
    /// Signed cumulative outgoing addition witness.
    pub cumulative_outgoing_witness_m3: f64,
    /// Original conservative incoming addition-error bound.
    pub incoming_bound_before_m3: f64,
    /// Candidate conservative incoming addition-error bound.
    pub incoming_bound_after_m3: f64,
    /// Original conservative outgoing addition-error bound.
    pub outgoing_bound_before_m3: f64,
    /// Candidate conservative outgoing addition-error bound.
    pub outgoing_bound_after_m3: f64,
}

/// Complete checked LevelPoolStepDiagnostics projection.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolStepDiagnostics {
    /// Original elapsed interval start in seconds.
    pub start: f64,
    /// Accepted elapsed interval end in seconds.
    pub end: f64,
    /// Exact represented interval duration in seconds.
    pub dt: f64,
    /// Aggregate original storage.
    pub volume_before_m3: f64,
    /// Aggregate accepted storage.
    pub volume_after_m3: f64,
    /// Aggregate authoritative incoming exchange.
    pub incoming_volume_m3: f64,
    /// Aggregate authoritative outgoing exchange.
    pub outgoing_volume_m3: f64,
    /// Independent aggregate interval residual.
    pub water_residual_m3: f64,
    /// Actual aggregate interval tolerance scale.
    pub physical_scale_m3: f64,
    /// Conservative aggregate reduction uncertainty.
    pub reduction_bound_m3: f64,
    /// Ordered complete element arithmetic and stage witnesses.
    pub receipts: Vec<LevelPoolReceipt>,
}

impl LevelPoolReceipt {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.element == other.element
            && self.old_volume_m3.to_bits() == other.old_volume_m3.to_bits()
            && self.incoming_volume_m3.to_bits() == other.incoming_volume_m3.to_bits()
            && self.outgoing_volume_m3.to_bits() == other.outgoing_volume_m3.to_bits()
            && self.remaining_volume_m3.to_bits() == other.remaining_volume_m3.to_bits()
            && self.volume_m3.to_bits() == other.volume_m3.to_bits()
            && self.outgoing_witness_m3.to_bits() == other.outgoing_witness_m3.to_bits()
            && self.incoming_witness_m3.to_bits() == other.incoming_witness_m3.to_bits()
            && self.water_residual_m3.to_bits() == other.water_residual_m3.to_bits()
            && self.witness_consistency_m3.to_bits() == other.witness_consistency_m3.to_bits()
            && self.witness_consistency_bound_m3.to_bits()
                == other.witness_consistency_bound_m3.to_bits()
            && self.physical_scale_m3.to_bits() == other.physical_scale_m3.to_bits()
            && match (self.stage_m, other.stage_m) {
                (None, None) => true,
                (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
                _ => false,
            }
            && self.reconstructed_volume_m3.to_bits() == other.reconstructed_volume_m3.to_bits()
            && self.inverse_residual_m3.to_bits() == other.inverse_residual_m3.to_bits()
            && self.cumulative_incoming_before_m3.to_bits()
                == other.cumulative_incoming_before_m3.to_bits()
            && self.cumulative_incoming_after_m3.to_bits()
                == other.cumulative_incoming_after_m3.to_bits()
            && self.cumulative_incoming_witness_m3.to_bits()
                == other.cumulative_incoming_witness_m3.to_bits()
            && self.cumulative_outgoing_before_m3.to_bits()
                == other.cumulative_outgoing_before_m3.to_bits()
            && self.cumulative_outgoing_after_m3.to_bits()
                == other.cumulative_outgoing_after_m3.to_bits()
            && self.cumulative_outgoing_witness_m3.to_bits()
                == other.cumulative_outgoing_witness_m3.to_bits()
            && self.incoming_bound_before_m3.to_bits() == other.incoming_bound_before_m3.to_bits()
            && self.incoming_bound_after_m3.to_bits() == other.incoming_bound_after_m3.to_bits()
            && self.outgoing_bound_before_m3.to_bits() == other.outgoing_bound_before_m3.to_bits()
            && self.outgoing_bound_after_m3.to_bits() == other.outgoing_bound_after_m3.to_bits()
    }
}
impl LevelPoolStepDiagnostics {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.start.to_bits() == other.start.to_bits()
            && self.end.to_bits() == other.end.to_bits()
            && self.dt.to_bits() == other.dt.to_bits()
            && self.volume_before_m3.to_bits() == other.volume_before_m3.to_bits()
            && self.volume_after_m3.to_bits() == other.volume_after_m3.to_bits()
            && self.incoming_volume_m3.to_bits() == other.incoming_volume_m3.to_bits()
            && self.outgoing_volume_m3.to_bits() == other.outgoing_volume_m3.to_bits()
            && self.water_residual_m3.to_bits() == other.water_residual_m3.to_bits()
            && self.physical_scale_m3.to_bits() == other.physical_scale_m3.to_bits()
            && self.reduction_bound_m3.to_bits() == other.reduction_bound_m3.to_bits()
            && self.receipts.len() == other.receipts.len()
            && self
                .receipts
                .iter()
                .zip(&other.receipts)
                .all(|(a, b)| a.same_bits(b))
    }
}

impl LevelPoolElementLedger {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.element == other.element
            && self.initial_volume_m3.to_bits() == other.initial_volume_m3.to_bits()
            && self.volume_m3.to_bits() == other.volume_m3.to_bits()
            && self.cumulative_incoming_volume_m3.to_bits()
                == other.cumulative_incoming_volume_m3.to_bits()
            && self.cumulative_outgoing_volume_m3.to_bits()
                == other.cumulative_outgoing_volume_m3.to_bits()
            && self.incoming_roundoff_bound_m3.to_bits()
                == other.incoming_roundoff_bound_m3.to_bits()
            && self.outgoing_roundoff_bound_m3.to_bits()
                == other.outgoing_roundoff_bound_m3.to_bits()
            && self.residual_m3.to_bits() == other.residual_m3.to_bits()
    }
}
