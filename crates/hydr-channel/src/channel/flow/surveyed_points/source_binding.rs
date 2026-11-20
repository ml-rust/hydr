use std::sync::Arc;

use super::potential_range::PotentialEvaluation;
use crate::channel::flow::surveyed::{
    PrismaticRiemannConfig,
    error::{HydraulicError, HydraulicResult},
    work::WorkLedger,
};

pub(super) struct RootSourceIdentity;

/// Bind cached potentials and root evidence to one fixed source graph and numerical policy.
#[derive(Clone)]
pub(super) struct RootSourceWitness {
    identity: Arc<RootSourceIdentity>,
    config: PrismaticRiemannConfig,
    panels: usize,
    work: *const WorkLedger,
}

/// Authenticated panel policy for one immutable source graph. This token owns no cache or work allowance.
#[derive(Clone)]
pub(super) struct SourcePanelLevel {
    witness: RootSourceWitness,
    index: usize,
    panels: usize,
}

impl SourcePanelLevel {
    pub(super) fn index(&self) -> usize {
        self.index
    }
    pub(super) fn panels(&self) -> usize {
        self.panels
    }
    pub(super) fn witness(&self) -> &RootSourceWitness {
        &self.witness
    }
}

impl RootSourceWitness {
    pub(super) fn capture(
        identity: &Arc<RootSourceIdentity>,
        evaluation: &PotentialEvaluation<'_>,
    ) -> Self {
        Self {
            identity: Arc::clone(identity),
            config: evaluation.config.clone(),
            panels: evaluation.panels,
            work: std::ptr::from_ref(evaluation.work),
        }
    }

    pub(super) fn panel_level(&self, index: usize) -> HydraulicResult<SourcePanelLevel> {
        if !self.panels.is_power_of_two()
            || self.panels > self.config.max_simpson_panels
            || !self.config.max_simpson_panels.is_power_of_two()
        {
            return Err(self.panel_error(
                index,
                "captured panels are not an admitted positive power of two",
            ));
        }
        let mut panels = self.panels;
        if index > self.config.max_refinements || index >= usize::BITS as usize {
            return Err(self.panel_error(
                index,
                "requested level exceeds the refinement or integer-doubling ceiling",
            ));
        }
        for _ in 0..index {
            panels = panels
                .checked_mul(2)
                .ok_or_else(|| self.panel_error(index, "panel doubling overflows usize"))?;
            if panels > self.config.max_simpson_panels {
                return Err(self.panel_error(
                    index,
                    &format!("requested panels={panels} exceed the panel ceiling"),
                ));
            }
        }
        Ok(SourcePanelLevel {
            witness: self.clone(),
            index,
            panels,
        })
    }

    fn panel_error(&self, index: usize, detail: &str) -> HydraulicError {
        HydraulicError::Numerical {
            context: format!(
                "Source panel level={index}, captured_panels={}, max_refinements={}, max_panels={}, integer_doubling_limit={}: {detail}. Request a level within every captured ceiling",
                self.panels,
                self.config.max_refinements,
                self.config.max_simpson_panels,
                usize::BITS - 1,
            ),
        }
    }

    pub(super) fn config(&self) -> &PrismaticRiemannConfig {
        &self.config
    }

    pub(super) fn check(
        &self,
        identity: &Arc<RootSourceIdentity>,
        evaluation: &PotentialEvaluation<'_>,
    ) -> HydraulicResult<()> {
        let changed = if !Arc::ptr_eq(&self.identity, identity) {
            "identity".into()
        } else if !std::ptr::eq(self.work, evaluation.work) {
            "evaluation.work".into()
        } else if self.panels != evaluation.panels {
            format!(
                "evaluation.panels={}, captured={}",
                evaluation.panels, self.panels
            )
        } else if config_signature(&self.config) != config_signature(evaluation.config) {
            "evaluation.config".into()
        } else {
            return Ok(());
        };
        Err(HydraulicError::Numerical {
            context: format!(
                "Paired source {changed} differs from its fixed witness. Rebuild the source graph and locate roots again"
            ),
        })
    }
}

fn config_signature(
    value: &PrismaticRiemannConfig,
) -> ([u64; 8], [usize; 7], solvr::optimize::CertifiedRootProbe) {
    // Exhaustive field matching forces future numerical controls into the fixed-policy witness.
    let PrismaticRiemannConfig {
        gravity_m_per_s2,
        recovery,
        water_flux_atol_m3_per_s,
        momentum_flux_atol_m4_per_s2,
        flux_rtol,
        root,
        max_bracket_expansions,
        initial_simpson_panels,
        max_simpson_panels,
        max_refinements,
        max_native_query_points,
    } = value;
    let crate::channel::section::PrismaticRecoveryConfig {
        depth_atol_m,
        depth_rtol,
    } = recovery;
    let solvr::optimize::CertifiedBisectionOptions {
        atol,
        rtol,
        max_iterations,
        max_points,
        probe,
    } = root;
    (
        [
            gravity_m_per_s2.to_bits(),
            depth_atol_m.to_bits(),
            depth_rtol.to_bits(),
            water_flux_atol_m3_per_s.to_bits(),
            momentum_flux_atol_m4_per_s2.to_bits(),
            flux_rtol.to_bits(),
            atol.to_bits(),
            rtol.to_bits(),
        ],
        [
            *max_iterations,
            *max_points,
            *max_bracket_expansions,
            *initial_simpson_panels,
            *max_simpson_panels,
            *max_refinements,
            *max_native_query_points,
        ],
        *probe,
    )
}
