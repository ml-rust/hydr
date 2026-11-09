//! Real surface forcing construction with immutable plan-instance checks.

use super::{AdmittedInterval, SupportError, SupportPlan};
use crate::solver::ForcingSchedule;
use std::sync::Arc;

impl SupportPlan {
    /// Construct the actual checked surface schedule from this plan's admitted packets.
    pub fn surface_schedule(
        &self,
        packets: &[AdmittedInterval],
    ) -> Result<ForcingSchedule, SupportError> {
        for packet in packets {
            self.check_packet(packet)?;
        }
        Ok(ForcingSchedule::new(
            &self.inner.domain,
            &self.inner.boundaries,
            packets.iter().map(|p| p.forcing.clone()).collect(),
        )?)
    }

    pub(super) fn check_packet(&self, packet: &AdmittedInterval) -> Result<(), SupportError> {
        if !Arc::ptr_eq(&self.inner, &packet.plan) {
            return Err(SupportError::identity(
                "packet belongs to another immutable plan instance. Admit fresh inputs through this plan",
            ));
        }
        Ok(())
    }
}
