//! Cached native source preparation for coordinated exchange trials.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

use super::{LevelPoolBatch, LevelPoolError, LevelPoolSources, LevelPoolTrial};
use crate::storage::numerics::sources::PreparedSources;
use crate::storage::tensor::TensorPoolSourceInput;

#[doc(hidden)]
pub struct LevelPoolSourcePlan<'a, R: Runtime<DType = DType>> {
    accepted: &'a LevelPoolBatch<R>,
    input: TensorPoolSourceInput<R>,
}

#[doc(hidden)]
pub struct PreparedLevelPoolInterval<'a, R: Runtime<DType = DType>> {
    accepted: &'a LevelPoolBatch<R>,
    end: f64,
    steps: usize,
    sources: Option<PreparedSources<R>>,
}

impl<R: Runtime<DType = DType>> LevelPoolBatch<R>
where
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
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    #[doc(hidden)]
    pub fn bind_source_rates(
        &self,
        sources: &[LevelPoolSources],
    ) -> Result<LevelPoolSourcePlan<'_, R>, LevelPoolError> {
        Ok(LevelPoolSourcePlan {
            accepted: self,
            input: TensorPoolSourceInput::new(&self.context, sources)?,
        })
    }
}

impl<'a, R: Runtime<DType = DType>> LevelPoolSourcePlan<'a, R>
where
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
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    #[doc(hidden)]
    pub fn prepare(&self, end: f64) -> Result<PreparedLevelPoolInterval<'a, R>, LevelPoolError> {
        let dt = super::trial::duration(self.accepted.time, end)?;
        let steps = super::trial::next_step_count(self.accepted.steps)?;
        let sources = crate::storage::numerics::sources::prepare(
            &self.accepted.context,
            &self.accepted.state,
            dt,
            &self.input,
        )?;
        Ok(PreparedLevelPoolInterval {
            accepted: self.accepted,
            end,
            steps,
            sources,
        })
    }
}

impl<R: Runtime<DType = DType>> PreparedLevelPoolInterval<'_, R> {
    #[doc(hidden)]
    pub fn available_volume(&self) -> &Tensor<R> {
        self.sources
            .as_ref()
            .map_or(&self.accepted.state.volume_m3, |(step, _)| {
                &step.post_source_volume_m3
            })
    }
}

impl<R: Runtime<DType = DType>> PreparedLevelPoolInterval<'_, R>
where
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
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    #[doc(hidden)]
    pub fn trial_to_tensors(
        &self,
        incoming: &Tensor<R>,
        outgoing: &Tensor<R>,
    ) -> Result<LevelPoolTrial<R>, LevelPoolError> {
        self.accepted.check_exchange_tensors(incoming, outgoing)?;
        let (state, step) = crate::storage::numerics::advance_prepared(
            &self.accepted.context,
            &self.accepted.state,
            incoming,
            outgoing,
            self.steps,
            self.sources.as_ref(),
        )?;
        self.accepted
            .complete_trial(self.end, self.steps, state, step)
    }

    #[doc(hidden)]
    pub fn source_totals(
        &self,
    ) -> Result<super::source_summary::LevelPoolIntervalSources, LevelPoolError> {
        super::source_summary::interval(
            &self.accepted.context,
            self.sources.as_ref().map(|(s, _)| s),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::LevelPoolInput;
    use crate::storage::fixture::batch;
    use std::sync::Arc;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn prepared_sources_match_native_trials_and_keep_original_storage() -> TestResult {
        let batch = batch(true)?;
        let before = batch.snapshot_data()?;
        let context = Arc::clone(&batch.context);
        let sources = [
            LevelPoolSources {
                rain_rate_m_per_s: 0.5,
                potential_evaporation_rate_m_per_s: 0.125,
                bottom_loss_rate_m3_per_s: 0.25,
            },
            LevelPoolSources {
                rain_rate_m_per_s: 0.25,
                potential_evaporation_rate_m_per_s: 0.0,
                bottom_loss_rate_m3_per_s: 0.0,
            },
        ];
        let plan = batch.bind_source_rates(&sources)?;
        for end in [1.0, 0.5] {
            let prepared = plan.prepare(end)?;
            assert_eq!(
                prepared.available_volume().try_to_vec::<f64>()?,
                vec![2.0 + 0.125 * end, 0.25 * end]
            );
            let totals = prepared.source_totals()?;
            assert_eq!(totals.rain_volume_m3, 0.75 * end);
            assert_eq!(totals.bottom_loss_volume_m3, 0.25 * end);
            assert_eq!(totals.evaporation_volume_m3, 0.125 * end);
            assert!(totals.reduction_bound_m3 >= 0.0);
            assert_eq!(totals.roundoff_bound_m3, 0.0);
            assert_eq!(totals.counter_roundoff_bound_m3, 0.0);
            let incoming = batch.context.tensor(&[0.0, end])?;
            let outgoing = batch.context.tensor(&[end, 0.0])?;
            let candidate = prepared.trial_to_tensors(&incoming, &outgoing)?.into_next();
            let native = batch
                .trial_with_sources_to(
                    end,
                    &[
                        LevelPoolInput {
                            incoming_volume_m3: 0.0,
                            outgoing_volume_m3: end,
                        },
                        LevelPoolInput {
                            incoming_volume_m3: end,
                            outgoing_volume_m3: 0.0,
                        },
                    ],
                    &sources,
                )?
                .into_next();
            assert_eq!(candidate.snapshot_data()?, native.snapshot_data()?);
            assert_eq!(candidate.accepted_steps(), 1);
            assert!(Arc::ptr_eq(&candidate.context, &context));
            assert_eq!(candidate.context.client.parallelism().chunk_size, Some(64));
            let zero = batch.context.tensor(&[0.0, 0.0])?;
            let alternate = prepared.trial_to_tensors(&zero, &zero)?.into_next();
            assert_eq!(
                alternate.volumes_tensor().try_to_vec::<f64>()?,
                prepared.available_volume().try_to_vec::<f64>()?
            );
            assert_eq!(candidate.snapshot_data()?, native.snapshot_data()?);
        }
        assert_eq!(batch.snapshot_data()?, before);
        Ok(())
    }

    #[test]
    fn prepared_zero_sources_retain_the_native_zero_source_path() -> TestResult {
        let batch = batch(true)?;
        let zero = batch.context.tensor(&[0.0, 0.0])?;
        let sources = [LevelPoolSources {
            rain_rate_m_per_s: 0.0,
            potential_evaporation_rate_m_per_s: 0.0,
            bottom_loss_rate_m3_per_s: 0.0,
        }; 2];
        let plan = batch.bind_source_rates(&sources)?;
        let prepared = plan.prepare(1.0)?;
        let totals = prepared.source_totals()?;
        assert_eq!(
            [
                totals.rain_volume_m3,
                totals.bottom_loss_volume_m3,
                totals.evaporation_volume_m3,
                totals.reduction_bound_m3,
                totals.roundoff_bound_m3,
                totals.counter_roundoff_bound_m3
            ],
            [0.0; 6]
        );
        let candidate = prepared.trial_to_tensors(&zero, &zero)?.into_next();
        let native = batch
            .trial_with_sources_to(
                1.0,
                &[LevelPoolInput {
                    incoming_volume_m3: 0.0,
                    outgoing_volume_m3: 0.0,
                }; 2],
                &sources,
            )?
            .into_next();
        assert_eq!(candidate.snapshot_data()?, native.snapshot_data()?);
        assert!(
            candidate
                .last_step()?
                .ok_or("missing source trial")?
                .receipts
                .iter()
                .all(|receipt| receipt.source.is_none())
        );
        Ok(())
    }

    #[test]
    fn prepared_loss_caps_do_not_credit_incoming_water_to_withdrawals() -> TestResult {
        let batch = batch(true)?;
        let before = batch.snapshot_data()?;
        let plan = batch.bind_source_rates(
            &[LevelPoolSources {
                rain_rate_m_per_s: 0.0,
                potential_evaporation_rate_m_per_s: 1.0,
                bottom_loss_rate_m3_per_s: 2.0,
            }; 2],
        )?;
        let prepared = plan.prepare(1.0)?;
        assert_eq!(
            prepared.available_volume().try_to_vec::<f64>()?,
            vec![0.0, 0.0]
        );
        let totals = prepared.source_totals()?;
        assert_eq!(totals.rain_volume_m3, 0.0);
        assert_eq!(totals.bottom_loss_volume_m3, 2.0);
        assert_eq!(totals.evaporation_volume_m3, 0.0);
        let incoming = batch.context.tensor(&[3.0, 0.0])?;
        let outgoing = batch.context.tensor(&[1.0, 0.0])?;
        assert!(prepared.trial_to_tensors(&incoming, &outgoing).is_err());
        let invalid_shape = batch.context.tensor(&[0.0])?;
        assert!(
            prepared
                .trial_to_tensors(&invalid_shape, &outgoing)
                .is_err()
        );
        assert_eq!(batch.snapshot_data()?, before);
        Ok(())
    }

    #[test]
    fn source_plans_reject_missing_policies_rates_and_unresolved_products() -> TestResult {
        let zero_sources = [LevelPoolSources {
            rain_rate_m_per_s: 0.0,
            potential_evaporation_rate_m_per_s: 0.0,
            bottom_loss_rate_m3_per_s: 0.0,
        }; 2];
        assert!(batch(false)?.bind_source_rates(&zero_sources).is_err());
        let batch = batch(true)?;
        let before = batch.snapshot_data()?;
        assert!(batch.bind_source_rates(&zero_sources[..1]).is_err());
        for rate in [-1.0, f64::NAN, f64::INFINITY] {
            let mut sources = zero_sources;
            sources[0].rain_rate_m_per_s = rate;
            assert!(batch.bind_source_rates(&sources).is_err());
        }
        let plan = batch.bind_source_rates(&zero_sources)?;
        for end in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(plan.prepare(end).is_err());
        }
        let mut sources = zero_sources;
        sources[0].rain_rate_m_per_s = f64::EPSILON / 4.0;
        assert!(batch.bind_source_rates(&sources)?.prepare(1.0).is_err());
        assert_eq!(batch.snapshot_data()?, before);
        Ok(())
    }
}
