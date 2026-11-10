//! Actual ordered immutable numerical trials without host-state advancement.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::{AdmittedInterval, SupportError, SupportErrorKind, SupportPlan};
use crate::runoff::{SubcatchmentBatch, SubcatchmentTrial};

impl SupportPlan {
    /// Trial the actual selected numerical component against the exact original interval.
    ///
    /// The batch retains its independently checked numerical config. The plan binds definitions and time.
    pub fn trial_subcatchments<R>(
        &self,
        packet: &AdmittedInterval,
        batch: Option<&SubcatchmentBatch<R>>,
    ) -> Result<Option<SubcatchmentTrial<R>>, SupportError>
    where
        R: Runtime<DType = DType>,
        R::Client: BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + TypeConversionOps<R>
            + UnaryOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + TwoSumOps<R>
            + BatchedNewtonAlgorithms<R>
            + BatchedBisectionAlgorithms<R>,
    {
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
