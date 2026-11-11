use super::config::PrismaticRiemannConfig;
use super::error::{HydraulicError, HydraulicResult, PrismaticRiemannRejection};
use std::cell::Cell;

/// Executed root and quadrature work across every refinement level.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrismaticRiemannWork {
    /// Root solver invocations across all refinement levels.
    pub root_solves: usize,
    /// Sum of completed root iterations reported by solver diagnostics.
    pub root_iterations: usize,
    /// Actual residual callback points, excluding conservative admission reservations.
    pub residual_points: usize,
    /// Actual Simpson integrand callback points across all integration calls.
    pub simpson_value_points: usize,
    /// Actual Simpson derivative-bound callback points across all integration calls.
    pub simpson_derivative_points: usize,
    /// Executed global panel-doubling rounds after the initial solve.
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
        let mut counts = self.snapshot();
        counts.root_iterations = counts
            .root_iterations
            .checked_add(iterations)
            .ok_or_else(|| self.exceeded())?;
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
