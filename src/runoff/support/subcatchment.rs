//! Actual ordered immutable numerical trials without host-state advancement.

use super::{AdmittedInterval, SupportError, SupportErrorKind, SupportPlan};
use crate::runoff::{SubcatchmentBatch, SubcatchmentTrial};

impl SupportPlan {
    /// Trial the actual selected numerical component against the exact original interval.
    ///
    /// The batch retains its independently checked numerical config. The plan binds definitions and time.
    pub fn trial_subcatchments(
        &self,
        packet: &AdmittedInterval,
        batch: Option<&SubcatchmentBatch>,
    ) -> Result<Option<SubcatchmentTrial>, SupportError> {
        self.check_packet(packet)?;
        if self.inner.definitions.is_empty() {
            if batch.is_some() {
                return Err(SupportError::new(
                    SupportErrorKind::Coverage,
                    "no active modelled definitions: supply batch=None without fabricated zero inputs",
                ));
            }
            return Ok(None);
        }
        let batch = batch.ok_or_else(|| {
            SupportError::new(
                SupportErrorKind::Coverage,
                "active modelled definitions require their actual numerical batch",
            )
        })?;
        if batch.definitions().len() != self.inner.definitions.len()
            || !batch
                .definitions()
                .iter()
                .zip(&self.inner.definitions)
                .all(|(a, b)| a.id == b.id && a.parameters.same_bits(&b.parameters))
        {
            return Err(SupportError::identity(
                "batch definition IDs/order/parameter bits differ from selected plan definitions. Construct the exact active batch",
            ));
        }
        if batch.time() != packet.start() {
            return Err(SupportError::new(
                SupportErrorKind::Coverage,
                format!(
                    "batch time={},packet start={}: supply a fresh exact packet at accepted time",
                    batch.time(),
                    packet.start()
                ),
            ));
        }
        Ok(Some(batch.trial_to(packet.end(), &packet.active_inputs)?))
    }
}
