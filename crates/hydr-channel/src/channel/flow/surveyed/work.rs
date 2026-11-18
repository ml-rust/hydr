use super::config::PrismaticRiemannConfig;
use super::error::{HydraulicError, HydraulicResult, PrismaticRiemannRejection};
use std::cell::Cell;

/// Executed root and quadrature work across every refinement level.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrismaticRiemannWork {
    /// Bisection solver invocations across all refinement levels.
    pub root_solves: usize,
    /// Sum of completed root iterations reported by solver diagnostics.
    pub root_iterations: usize,
    /// Largest completed iteration count reported by one root solve batch.
    pub maximum_root_iterations: usize,
    /// Upper-bracket solver invocations, including endpoint-only searches.
    pub bracket_solves: usize,
    /// Sum of completed upper-bracket expansion rounds across all calls.
    pub bracket_expansion_rounds: usize,
    /// Largest completed expansion-round count reported by one upper-bracket call.
    pub maximum_bracket_expansion_rounds: usize,
    /// Actual residual callback points, excluding conservative admission reservations.
    pub residual_points: usize,
    /// Actual Simpson integrand callback points across all integration calls.
    pub simpson_value_points: usize,
    /// Actual Simpson derivative-bound callback points across all integration calls.
    pub simpson_derivative_points: usize,
    /// Executed refinement rounds after the initial solve.
    /// A round is one global panel doubling, or one point-flux controller round that advanced a star or sonic fan.
    pub refinements: usize,
    /// Largest panel count executed for one active band contribution.
    pub maximum_simpson_panels: usize,
}

pub(crate) enum QueryWork {
    Residual,
    SimpsonValue,
    SimpsonDerivative,
}

pub(crate) struct WorkLedger {
    counts: Cell<PrismaticRiemannWork>,
    limit: usize,
}

impl WorkLedger {
    pub(crate) fn new(config: &PrismaticRiemannConfig) -> Self {
        Self {
            counts: Cell::new(PrismaticRiemannWork::default()),
            limit: config.max_native_query_points,
        }
    }
    pub(crate) fn snapshot(&self) -> PrismaticRiemannWork {
        self.counts.get()
    }
    fn exceeded(&self) -> HydraulicError {
        HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::WorkLimit,
            query: None,
            section_owner: None,
            context: format!(
                "native work exceeds limit={}. Increase the work ceiling or split the batch",
                self.limit
            ),
        }
    }
    pub(crate) fn charge(&self, kind: QueryWork, points: usize) -> HydraulicResult<()> {
        let mut counts = self.snapshot();
        let total = counts
            .residual_points
            .checked_add(counts.simpson_value_points)
            .and_then(|n| n.checked_add(counts.simpson_derivative_points))
            .and_then(|n| n.checked_add(points))
            .ok_or_else(|| self.exceeded())?;
        if total > self.limit {
            return Err(self.exceeded());
        }
        let target = match kind {
            QueryWork::Residual => &mut counts.residual_points,
            QueryWork::SimpsonValue => &mut counts.simpson_value_points,
            QueryWork::SimpsonDerivative => &mut counts.simpson_derivative_points,
        };
        *target = target.checked_add(points).ok_or_else(|| self.exceeded())?;
        self.counts.set(counts);
        Ok(())
    }
    pub(crate) fn begin_root(&self) -> HydraulicResult<()> {
        let mut counts = self.snapshot();
        counts.root_solves = counts
            .root_solves
            .checked_add(1)
            .ok_or_else(|| self.exceeded())?;
        self.counts.set(counts);
        Ok(())
    }
    pub(crate) fn finish_root(&self, iterations: usize) -> HydraulicResult<()> {
        self.finish_root_progress(iterations, iterations)
    }
    pub(crate) fn finish_root_progress(
        &self,
        additional_iterations: usize,
        cumulative_session_iterations: usize,
    ) -> HydraulicResult<()> {
        if additional_iterations > cumulative_session_iterations {
            return Err(HydraulicError::Numerical {
                context: format!(
                    "Root additional_iterations={additional_iterations} exceeds cumulative_session_iterations={cumulative_session_iterations}. Discard inconsistent root progress"
                ),
            });
        }
        let mut counts = self.snapshot();
        counts.root_iterations = counts
            .root_iterations
            .checked_add(additional_iterations)
            .ok_or_else(|| self.exceeded())?;
        counts.maximum_root_iterations = counts
            .maximum_root_iterations
            .max(cumulative_session_iterations);
        self.counts.set(counts);
        Ok(())
    }
    pub(crate) fn begin_bracket(&self) -> HydraulicResult<()> {
        let mut counts = self.snapshot();
        counts.bracket_solves = counts
            .bracket_solves
            .checked_add(1)
            .ok_or_else(|| self.exceeded())?;
        self.counts.set(counts);
        Ok(())
    }
    pub(crate) fn finish_bracket(&self, rounds: usize) -> HydraulicResult<()> {
        let mut counts = self.snapshot();
        counts.bracket_expansion_rounds = counts
            .bracket_expansion_rounds
            .checked_add(rounds)
            .ok_or_else(|| self.exceeded())?;
        counts.maximum_bracket_expansion_rounds =
            counts.maximum_bracket_expansion_rounds.max(rounds);
        self.counts.set(counts);
        Ok(())
    }
    pub(crate) fn begin_refinement(&self, panels: usize, refinement: usize) -> HydraulicResult<()> {
        let mut counts = self.snapshot();
        if refinement != 0 {
            counts.refinements = counts
                .refinements
                .checked_add(1)
                .ok_or_else(|| self.exceeded())?;
        }
        counts.maximum_simpson_panels = counts.maximum_simpson_panels.max(panels);
        self.counts.set(counts);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracket_rounds_preserve_root_and_callback_counts() -> HydraulicResult<()> {
        let ledger = WorkLedger::new(&PrismaticRiemannConfig::default());
        ledger.begin_root()?;
        ledger.finish_root(7)?;
        ledger.charge(QueryWork::Residual, 11)?;
        let before = ledger.snapshot();
        ledger.finish_bracket(0)?;
        assert_eq!(ledger.snapshot(), before);
        ledger.finish_bracket(3)?;
        ledger.finish_bracket(5)?;
        assert_eq!(
            ledger.snapshot(),
            PrismaticRiemannWork {
                bracket_expansion_rounds: 8,
                maximum_bracket_expansion_rounds: 5,
                ..before
            }
        );
        Ok(())
    }

    #[test]
    fn bracket_round_overflow_preserves_executed_counts() {
        let ledger = WorkLedger::new(&PrismaticRiemannConfig::default());
        let before = PrismaticRiemannWork {
            bracket_expansion_rounds: usize::MAX,
            maximum_bracket_expansion_rounds: 3,
            ..PrismaticRiemannWork::default()
        };
        ledger.counts.set(before);
        assert!(matches!(
            ledger.finish_bracket(4),
            Err(HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::WorkLimit,
                ..
            })
        ));
        assert_eq!(ledger.snapshot(), before);
    }

    #[test]
    fn resumed_root_work_counts_deltas_and_keeps_the_session_maximum() -> HydraulicResult<()> {
        let ledger = WorkLedger::new(&PrismaticRiemannConfig::default());
        ledger.begin_root()?;
        ledger.finish_root_progress(3, 3)?;
        ledger.begin_root()?;
        ledger.finish_root_progress(2, 5)?;
        ledger.begin_root()?;
        ledger.finish_root_progress(1, 6)?;
        ledger.charge(QueryWork::Residual, 17)?;
        assert_eq!(
            ledger.snapshot(),
            PrismaticRiemannWork {
                root_solves: 3,
                root_iterations: 6,
                maximum_root_iterations: 6,
                residual_points: 17,
                ..PrismaticRiemannWork::default()
            }
        );
        let before = ledger.snapshot();
        ledger.finish_root_progress(0, 6)?;
        assert_eq!(ledger.snapshot(), before);
        Ok(())
    }

    #[test]
    fn independent_root_sessions_keep_separate_iteration_maxima() -> HydraulicResult<()> {
        let ledger = WorkLedger::new(&PrismaticRiemannConfig::default());
        ledger.finish_root_progress(5, 5)?;
        ledger.finish_root_progress(2, 2)?;
        assert_eq!(ledger.snapshot().root_iterations, 7);
        assert_eq!(ledger.snapshot().maximum_root_iterations, 5);
        Ok(())
    }

    #[test]
    fn invalid_or_overflowing_root_progress_preserves_all_counters() {
        let ledger = WorkLedger::new(&PrismaticRiemannConfig::default());
        let before = PrismaticRiemannWork {
            root_solves: 2,
            root_iterations: 7,
            maximum_root_iterations: 5,
            residual_points: 13,
            ..PrismaticRiemannWork::default()
        };
        ledger.counts.set(before);
        assert!(ledger.finish_root_progress(4, 3).is_err());
        assert_eq!(ledger.snapshot(), before);
        let overflow = PrismaticRiemannWork {
            root_iterations: usize::MAX,
            ..before
        };
        ledger.counts.set(overflow);
        assert!(matches!(
            ledger.finish_root_progress(1, 1),
            Err(HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::WorkLimit,
                ..
            })
        ));
        assert_eq!(ledger.snapshot(), overflow);
    }

    #[test]
    fn legacy_root_completion_matches_one_progress_advance() -> HydraulicResult<()> {
        let config = PrismaticRiemannConfig::default();
        let legacy = WorkLedger::new(&config);
        let progress = WorkLedger::new(&config);
        legacy.finish_root(7)?;
        progress.finish_root_progress(7, 7)?;
        assert_eq!(legacy.snapshot(), progress.snapshot());
        Ok(())
    }
}
