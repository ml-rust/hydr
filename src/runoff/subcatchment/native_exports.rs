//! Original native export operands and on-demand scalar accounting summaries.

use super::accounting::StepTotals;
use super::accounting_bounds::{self, SubcatchmentAccountingBounds};
use super::{SubcatchmentBatch, SubcatchmentError, SubcatchmentTrial};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SubcatchmentIntervalSummary {
    pub start_s: f64,
    pub end_s: f64,
    pub totals: StepTotals,
    pub transfer_uncertainty_charge_m3: f64,
    pub bounds: SubcatchmentAccountingBounds,
}

impl<R: Runtime<DType = DType>> SubcatchmentBatch<R> {
    /// Borrow original flattened `[catchment, subarea]` exports without numerical projection.
    pub(crate) fn original_outlet(&self) -> Result<&Tensor<R>, SubcatchmentError> {
        Ok(&self
            .last
            .as_ref()
            .ok_or_else(|| {
                SubcatchmentError::numerical(
                    "native runoff export requires a retained accepted interval",
                )
            })?
            .step
            .totals
            .outlet)
    }
    /// Reuse the exact scalars retained when the native interval first passes admission.
    pub(crate) fn interval_summary(
        &self,
    ) -> Result<SubcatchmentIntervalSummary, SubcatchmentError> {
        let stored = self.last.as_ref().ok_or_else(|| {
            SubcatchmentError::numerical(
                "native runoff summary requires a retained accepted interval",
            )
        })?;
        let totals = &stored.step.totals;
        Ok(SubcatchmentIntervalSummary {
            start_s: stored.start,
            end_s: stored.end,
            totals: stored.totals,
            transfer_uncertainty_charge_m3: totals.uncertainty_charge,
            bounds: accounting_bounds::interval_bounds(
                &self.context,
                &stored.totals,
                totals.adjustment_charge,
            )?,
        })
    }
    pub(crate) fn accounting_bounds(
        &self,
    ) -> Result<SubcatchmentAccountingBounds, SubcatchmentError> {
        accounting_bounds::lifetime_bounds(&self.context, &self.ledger, self.steps)
    }
}
impl<R: Runtime<DType = DType>> SubcatchmentTrial<R> {
    pub(crate) fn original_outlet(&self) -> Result<&Tensor<R>, SubcatchmentError> {
        self.next.original_outlet()
    }
}
