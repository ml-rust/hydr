//! Modelled rate records admitted once and trialled over any contained sub-span.
//!
//! Admission rejects positive runon, so every admitted value is a constant rate over its record.
//! The same rate holds over every sub-span of the record. A trial to a shorter end recomputes
//! runoff from the accepted state with those rates. It never slices or scales a volume.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::admission::{interval, span};
use super::definition::CompiledPlan;
use super::{SubcatchmentContributionInput, SupportError, SupportErrorKind, SupportPlan};
use crate::runoff::{SubcatchmentBatch, SubcatchmentInput, SubcatchmentTrial};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Checked active modelled rates over one exact record span, bound to one plan instance.
#[derive(Clone, Debug)]
pub struct AdmittedRates {
    plan: Arc<CompiledPlan>,
    start: f64,
    end: f64,
    inputs: Vec<SubcatchmentInput>,
}

impl AdmittedRates {
    /// Exact record start, in seconds.
    pub fn start(&self) -> f64 {
        self.start
    }
    /// Exact record end, in seconds.
    pub fn end(&self) -> f64 {
        self.end
    }
    /// Active modelled inputs in selected definition order. Every runon volume is zero.
    pub fn inputs(&self) -> &[SubcatchmentInput] {
        &self.inputs
    }
}

impl SupportPlan {
    /// Admit one rate record per active modelled source over exactly `[start, end]`.
    ///
    /// # Errors
    /// Returns [`SupportError::Admission`] for:
    /// - a start that is not finite and nonnegative, or a duration that is not finite and positive
    /// - a record whose span differs from `[start, end]` bit for bit
    /// - a missing, repeated, unknown, or suppressed source, or a changed input identity
    /// - a rain or evaporation rate that is not finite and nonnegative
    /// - a runon volume that is not finite and nonnegative, or one that is positive
    pub fn admit_rates(
        &self,
        start: f64,
        end: f64,
        records: &[SubcatchmentContributionInput],
    ) -> Result<AdmittedRates, SupportError> {
        interval(start, end)?;
        for record in records {
            span(
                start,
                end,
                record.start,
                record.end,
                &record.contribution_id,
            )?;
        }
        let inputs = self.select_active(records.iter().cloned())?;
        Ok(AdmittedRates {
            plan: Arc::clone(&self.inner),
            start,
            end,
            inputs,
        })
    }

    /// Trial the active batch from its accepted time to `end` with the admitted rates.
    ///
    /// The span `[batch.time(), end]` must lie inside the record span. Each call recomputes
    /// runoff from the accepted state, so a shorter end gives the exact shorter trial.
    ///
    /// # Errors
    /// Returns [`SupportError::Admission`] for:
    /// - rates admitted by another plan instance
    /// - an end outside `(rates.start(), rates.end()]`
    /// - a batch present without active definitions, or absent with them
    /// - batch definitions that differ from the active definitions in ID, order, or bits
    /// - a batch time outside `[rates.start(), end)`
    ///
    /// Returns [`SupportError::Subcatchment`] when the trial rejects.
    pub fn trial_rates_to<R>(
        &self,
        rates: &AdmittedRates,
        batch: Option<&SubcatchmentBatch<R>>,
        end: f64,
    ) -> Result<Option<SubcatchmentTrial<R>>, SupportError>
    where
        R: Runtime<DType = DType>,
        R::Client: BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + CorrectlyRoundedOps<R>
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
        if !Arc::ptr_eq(&self.inner, &rates.plan) {
            return Err(SupportError::identity(
                "rates belong to another immutable plan instance. Admit fresh rates through this plan",
            ));
        }
        if !end.is_finite() || end <= rates.start || end > rates.end {
            return Err(SupportError::new(
                SupportErrorKind::Coverage,
                format!(
                    "end={end},rates=[{},{}]: choose an end inside the record span",
                    rates.start, rates.end
                ),
            ));
        }
        let Some(batch) = self.check_batch(batch)? else {
            return Ok(None);
        };
        let time = batch.time();
        if time < rates.start || time >= end {
            return Err(SupportError::new(
                SupportErrorKind::Coverage,
                format!(
                    "batch time={time},end={end},rates=[{},{}]: trial from an accepted time inside the record span",
                    rates.start, rates.end
                ),
            ));
        }
        Ok(Some(batch.trial_to(end, &rates.inputs)?))
    }

    /// Exactly one checked request per active source, in selected definition order.
    pub(super) fn select_active(
        &self,
        requests: impl IntoIterator<Item = SubcatchmentContributionInput>,
    ) -> Result<Vec<SubcatchmentInput>, SupportError> {
        let mut models = BTreeMap::new();
        for request in requests {
            if let Some(previous) = models.insert(request.contribution_id.clone(), request) {
                return Err(err(
                    SupportErrorKind::Coverage,
                    format!("duplicate model input={:?}", previous.contribution_id),
                ));
            }
        }
        let mut active_inputs = Vec::with_capacity(self.inner.active_contributions.len());
        for source in self.active_contributions() {
            let request = models.remove(&source.id).ok_or_else(|| {
                err(
                    SupportErrorKind::Coverage,
                    format!(
                        "missing active input={:?}: supply exact interval without fallback",
                        source.id
                    ),
                )
            })?;
            if request.input_identity != source.input_identity {
                return Err(SupportError::identity(format!(
                    "model={:?},identity={:?},expected={:?}",
                    source.id, request.input_identity, source.input_identity
                )));
            }
            let forcing = request.input;
            if forcing
                .external_runon_volumes_m3
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
                || !forcing.rain_rate_m_per_s.is_finite()
                || forcing.rain_rate_m_per_s < 0.0
                || !forcing.potential_evaporation_m_per_s.is_finite()
                || forcing.potential_evaporation_m_per_s < 0.0
            {
                return Err(err(
                    SupportErrorKind::Coverage,
                    format!(
                        "model={:?},input={forcing:?}: require finite nonnegative physical inputs",
                        source.id
                    ),
                ));
            }
            if forcing.external_runon_volumes_m3.iter().any(|v| *v > 0.0) {
                return Err(err(
                    SupportErrorKind::Composition,
                    format!(
                        "model={:?}: external runon requires donor provenance and a paired coordinator. Supply zero runon in this ownership unit",
                        source.id
                    ),
                ));
            }
            active_inputs.push(forcing);
        }
        if !models.is_empty() {
            return Err(err(
                SupportErrorKind::Coverage,
                format!(
                    "unknown or suppressed model requests={:?}: provide active sources only",
                    models.keys().collect::<Vec<_>>()
                ),
            ));
        }
        Ok(active_inputs)
    }
}

fn err(kind: SupportErrorKind, context: impl Into<String>) -> SupportError {
    SupportError::new(kind, context)
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{Result, batch, fixture, kind, record};
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn admitted_rates_hold_active_inputs_in_definition_order() -> Result {
        let plan = fixture()?.plan()?;
        let rates = plan.admit_rates(0.0, 4.0, &[record(0.0, 4.0, 1e-5)])?;
        assert_eq!(rates.start().to_bits(), 0.0_f64.to_bits());
        assert_eq!(rates.end().to_bits(), 4.0_f64.to_bits());
        assert_eq!(rates.inputs(), &[record(0.0, 4.0, 1e-5).input]);
        Ok(())
    }

    #[test]
    fn admit_rates_rejects_a_record_off_the_exact_span() -> Result {
        let plan = fixture()?.plan()?;
        let shifted = plan.admit_rates(0.0, 4.0, &[record(0.0, 2.0, 1e-5)]);
        assert_eq!(kind(shifted), Some(SupportErrorKind::Coverage));
        Ok(())
    }

    #[test]
    fn admit_rates_rejects_positive_runon() -> Result {
        let plan = fixture()?.plan()?;
        let mut runon = record(0.0, 4.0, 0.0);
        runon.input.external_runon_volumes_m3[0] = 1e-3;
        assert_eq!(
            kind(plan.admit_rates(0.0, 4.0, &[runon])),
            Some(SupportErrorKind::Composition)
        );
        Ok(())
    }

    #[test]
    fn admit_rates_rejects_missing_repeated_unknown_and_renamed_records() -> Result {
        let plan = fixture()?.plan()?;
        assert_eq!(
            kind(plan.admit_rates(0.0, 4.0, &[])),
            Some(SupportErrorKind::Coverage)
        );
        let once = record(0.0, 4.0, 0.0);
        assert_eq!(
            kind(plan.admit_rates(0.0, 4.0, &[once.clone(), once.clone()])),
            Some(SupportErrorKind::Coverage)
        );
        let mut unknown = once.clone();
        unknown.contribution_id = "elsewhere".into();
        assert_eq!(
            kind(plan.admit_rates(0.0, 4.0, &[once.clone(), unknown])),
            Some(SupportErrorKind::Coverage)
        );
        let mut renamed = once;
        renamed.input_identity = "other-input".into();
        assert_eq!(
            kind(plan.admit_rates(0.0, 4.0, &[renamed])),
            Some(SupportErrorKind::Identity)
        );
        Ok(())
    }

    #[test]
    fn sub_span_trial_equals_the_direct_trial_bit_for_bit() -> Result {
        let plan = fixture()?.plan()?;
        let rates = plan.admit_rates(0.0, 4.0, &[record(0.0, 4.0, 1e-5)])?;
        let accepted = batch(&plan, 0.0)?;
        for end in [1.0, 2.5, 4.0] {
            let Some(trial) = plan.trial_rates_to(&rates, Some(&accepted), end)? else {
                return Err("active plan returned no trial".into());
            };
            let direct = accepted.trial_to(end, rates.inputs())?;
            let (trial, direct) = (trial.into_next(), direct.into_next());
            assert_eq!(trial.time().to_bits(), end.to_bits());
            assert_eq!(
                format!("{:?} {:?}", trial.states()?, trial.ledger()),
                format!("{:?} {:?}", direct.states()?, direct.ledger())
            );
        }
        Ok(())
    }

    #[test]
    fn shorter_trial_recomputes_rather_than_scales() -> Result {
        let plan = fixture()?.plan()?;
        let rates = plan.admit_rates(0.0, 4.0, &[record(0.0, 4.0, 0.0)])?;
        let accepted = batch(&plan, 0.0)?;
        let outlet = |end: f64| -> Result<f64> {
            let Some(trial) = plan.trial_rates_to(&rates, Some(&accepted), end)? else {
                return Err("active plan returned no trial".into());
            };
            Ok(trial.into_next().ledger().outlet_volume_m3)
        };
        let (whole, half) = (outlet(4.0)?, outlet(2.0)?);
        assert!(half > 0.0 && whole > half, "whole={whole},half={half}");
        assert_ne!(half.to_bits(), (whole / 2.0).to_bits());
        Ok(())
    }

    #[test]
    fn trial_rates_to_rejects_spans_outside_the_record() -> Result {
        let plan = fixture()?.plan()?;
        let rates = plan.admit_rates(1.0, 4.0, &[record(1.0, 4.0, 0.0)])?;
        let accepted = batch(&plan, 1.0)?;
        for end in [1.0, 4.5, f64::NAN] {
            assert_eq!(
                kind(plan.trial_rates_to(&rates, Some(&accepted), end)),
                Some(SupportErrorKind::Coverage),
                "end={end}"
            );
        }
        let early = batch(&plan, 0.5)?;
        assert_eq!(
            kind(plan.trial_rates_to(&rates, Some(&early), 2.0)),
            Some(SupportErrorKind::Coverage)
        );
        let late = batch(&plan, 3.0)?;
        assert_eq!(
            kind(plan.trial_rates_to(&rates, Some(&late), 2.0)),
            Some(SupportErrorKind::Coverage)
        );
        Ok(())
    }

    #[test]
    fn trial_rates_to_rejects_foreign_rates_and_a_missing_batch() -> Result {
        let plan = fixture()?.plan()?;
        let other = fixture()?.plan()?;
        let foreign = other.admit_rates(0.0, 4.0, &[record(0.0, 4.0, 0.0)])?;
        let accepted = batch(&plan, 0.0)?;
        assert_eq!(
            kind(plan.trial_rates_to(&foreign, Some(&accepted), 2.0)),
            Some(SupportErrorKind::Identity)
        );
        let rates = plan.admit_rates(0.0, 4.0, &[record(0.0, 4.0, 0.0)])?;
        assert_eq!(
            kind(plan.trial_rates_to::<CpuRuntime>(&rates, None, 2.0)),
            Some(SupportErrorKind::Coverage)
        );
        Ok(())
    }
}
