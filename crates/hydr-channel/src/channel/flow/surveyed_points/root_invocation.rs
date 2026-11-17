use super::source_binding::RootSourceWitness;
use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult};
use std::sync::Arc;

struct RootInvocationIdentity;

/// Own one root invocation's fixed source policy and cumulative callback allowance.
pub(super) struct RootInvocation {
    source: RootSourceWitness,
    point_limit: usize,
    callback_points: usize,
    identity: Arc<RootInvocationIdentity>,
}

/// Identity authorizes domains to use their owner. It carries no callback allowance.
#[derive(Clone)]
pub(super) struct RootInvocationWitness {
    identity: Arc<RootInvocationIdentity>,
}

impl RootInvocation {
    pub(super) fn new(source: RootSourceWitness, point_limit: usize) -> HydraulicResult<Self> {
        if point_limit > source.config().root.max_points {
            return Err(inconsistent(&format!(
                "point_limit={point_limit} exceeds captured root.max_points={}",
                source.config().root.max_points
            )));
        }
        Ok(Self {
            source,
            point_limit,
            callback_points: 0,
            identity: Arc::new(RootInvocationIdentity),
        })
    }

    pub(super) fn source_witness(&self) -> &RootSourceWitness {
        &self.source
    }
    pub(super) fn point_limit(&self) -> usize {
        self.point_limit
    }
    pub(super) fn callback_points(&self) -> usize {
        self.callback_points
    }
    pub(super) fn witness(&self) -> RootInvocationWitness {
        RootInvocationWitness {
            identity: Arc::clone(&self.identity),
        }
    }

    pub(super) fn check(&self, witness: &RootInvocationWitness) -> HydraulicResult<()> {
        if !Arc::ptr_eq(&self.identity, &witness.identity) {
            return Err(inconsistent("domain belongs to another invocation"));
        }
        Ok(())
    }

    pub(super) fn remaining(&self) -> HydraulicResult<usize> {
        self.point_limit
            .checked_sub(self.callback_points)
            .ok_or_else(|| {
                inconsistent(&format!(
                    "callback_points={} exceeds point_limit={}",
                    self.callback_points, self.point_limit
                ))
            })
    }

    pub(super) fn check_add_points(&self, points: usize) -> HydraulicResult<()> {
        self.checked_total(points).map(|_| ())
    }

    pub(super) fn add_points(&mut self, points: usize) -> HydraulicResult<()> {
        self.callback_points = self.checked_total(points)?;
        Ok(())
    }

    fn checked_total(&self, points: usize) -> HydraulicResult<usize> {
        let total = self.callback_points.checked_add(points).ok_or_else(|| {
            inconsistent(&format!(
                "callback_points={} plus points={points} overflows usize with point_limit={}",
                self.callback_points, self.point_limit
            ))
        })?;
        if total > self.point_limit {
            return Err(inconsistent(&format!(
                "callback_points={} plus points={points} gives total={total}, exceeding point_limit={}",
                self.callback_points, self.point_limit
            )));
        }
        Ok(total)
    }
}

fn inconsistent(message: &str) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!(
            "Root invocation {message}. Preserve the original invocation owner and allowance"
        ),
    }
}

#[cfg(test)]
#[path = "root_invocation_tests.rs"]
mod tests;
