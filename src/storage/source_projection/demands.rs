//! Explicit cap/counter projection helpers shared by complete source receipts.

use crate::cpu::level_pool::{CpuPoolCap, CpuPoolCounter};
use crate::storage::accounting::read;
use crate::storage::{LevelPoolCounterReceipt, LevelPoolDemandCapReceipt, LevelPoolError};
pub(super) struct CapVectors {
    potential_volume_m3: Vec<f64>,
    actual_volume_m3: Vec<f64>,
    unmet_volume_m3: Vec<f64>,
    unmet_witness_m3: Vec<f64>,
    reconciliation_residual_m3: Vec<f64>,
    reconciliation_consistency_m3: Vec<f64>,
    reconciliation_bound_m3: Vec<f64>,
    availability_limited: Vec<f64>,
}
impl CapVectors {
    pub(super) fn new(value: &CpuPoolCap, n: usize) -> Result<Self, LevelPoolError> {
        Ok(Self {
            potential_volume_m3: read(&value.potential_volume_m3, n)?,
            actual_volume_m3: read(&value.actual_volume_m3, n)?,
            unmet_volume_m3: read(&value.unmet_volume_m3, n)?,
            unmet_witness_m3: read(&value.unmet_witness_m3, n)?,
            reconciliation_residual_m3: read(&value.reconciliation_residual_m3, n)?,
            reconciliation_consistency_m3: read(&value.reconciliation_consistency_m3, n)?,
            reconciliation_bound_m3: read(&value.reconciliation_bound_m3, n)?,
            availability_limited: read(&value.availability_limited, n)?,
        })
    }
    pub(super) fn at(&self, i: usize) -> LevelPoolDemandCapReceipt {
        LevelPoolDemandCapReceipt {
            potential_volume_m3: self.potential_volume_m3[i],
            actual_volume_m3: self.actual_volume_m3[i],
            unmet_volume_m3: self.unmet_volume_m3[i],
            unmet_witness_m3: self.unmet_witness_m3[i],
            reconciliation_residual_m3: self.reconciliation_residual_m3[i],
            reconciliation_consistency_m3: self.reconciliation_consistency_m3[i],
            reconciliation_bound_m3: self.reconciliation_bound_m3[i],
            availability_limited: self.availability_limited[i] != 0.0,
        }
    }
}
pub(super) struct CounterVectors {
    before_m3: Vec<f64>,
    after_m3: Vec<f64>,
    witness_m3: Vec<f64>,
    bound_before_m3: Vec<f64>,
    bound_after_m3: Vec<f64>,
}
impl CounterVectors {
    pub(super) fn new(value: &CpuPoolCounter, n: usize) -> Result<Self, LevelPoolError> {
        Ok(Self {
            before_m3: read(&value.before_m3, n)?,
            after_m3: read(&value.after_m3, n)?,
            witness_m3: read(&value.witness_m3, n)?,
            bound_before_m3: read(&value.bound_before_m3, n)?,
            bound_after_m3: read(&value.bound_after_m3, n)?,
        })
    }
    pub(super) fn at(&self, i: usize) -> LevelPoolCounterReceipt {
        LevelPoolCounterReceipt {
            before_m3: self.before_m3[i],
            after_m3: self.after_m3[i],
            witness_m3: self.witness_m3[i],
            bound_before_m3: self.bound_before_m3[i],
            bound_after_m3: self.bound_after_m3[i],
        }
    }
}
