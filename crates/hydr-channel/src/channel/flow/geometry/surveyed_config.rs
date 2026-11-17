use super::super::{ChannelConfig, ChannelError};
use crate::channel::flow::surveyed::config::PrismaticRiemannConfig;
use crate::channel::section::PrismaticRecoveryConfig;
use solvr::optimize::CertifiedBisectionOptions;

/// Geometry allowances and work ceilings for a constant surveyed reach.
#[derive(Clone, Debug)]
pub struct ChannelSurveyedConfig {
    /// Recovered geometric-depth enclosure allowances, in metres.
    pub recovery: PrismaticRecoveryConfig,
    /// Normalized-depth root controls. `max_iterations` limits midpoint rounds.
    pub normalized_root: CertifiedBisectionOptions,
    /// Maximum upper-bracket expansion rounds, at most `i64::MAX`. Zero checks initial endpoints only. Default: `4096`.
    pub max_bracket_expansions: usize,
    /// Initial Simpson panels per active band contribution.
    pub initial_simpson_panels: usize,
    /// Maximum Simpson panels per active band contribution.
    pub max_simpson_panels: usize,
    /// Maximum panel doublings after the initial solve.
    pub max_refinements: usize,
    /// Cumulative native callback point ceiling.
    pub max_native_query_points: usize,
    /// Absolute momentum allowance for hydraulic-radius selection, in m4/s.
    pub friction_geometry_atol_m4_per_s: f64,
    /// Relative momentum allowance for hydraulic-radius selection.
    pub friction_geometry_rtol: f64,
}

impl Default for ChannelSurveyedConfig {
    fn default() -> Self {
        Self {
            recovery: PrismaticRecoveryConfig::default(),
            normalized_root: CertifiedBisectionOptions {
                atol: 1e-15,
                rtol: 1e-15,
                max_iterations: 256,
                max_points: 1 << 20,
            },
            max_bracket_expansions: 4096,
            initial_simpson_panels: 4,
            max_simpson_panels: 4096,
            max_refinements: 10,
            max_native_query_points: 1 << 24,
            friction_geometry_atol_m4_per_s: 1e-9,
            friction_geometry_rtol: 1e-12,
        }
    }
}

impl PartialEq for ChannelSurveyedConfig {
    fn eq(&self, other: &Self) -> bool {
        self.same_bits(other)
    }
}

impl ChannelSurveyedConfig {
    pub(in crate::channel::flow) fn same_bits(&self, other: &Self) -> bool {
        let values = |c: &Self| {
            [
                c.recovery.depth_atol_m,
                c.recovery.depth_rtol,
                c.normalized_root.atol,
                c.normalized_root.rtol,
                c.friction_geometry_atol_m4_per_s,
                c.friction_geometry_rtol,
            ]
        };
        values(self)
            .iter()
            .zip(values(other))
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.normalized_root.max_iterations == other.normalized_root.max_iterations
            && self.normalized_root.max_points == other.normalized_root.max_points
            && self.max_bracket_expansions == other.max_bracket_expansions
            && self.initial_simpson_panels == other.initial_simpson_panels
            && self.max_simpson_panels == other.max_simpson_panels
            && self.max_refinements == other.max_refinements
            && self.max_native_query_points == other.max_native_query_points
    }

    pub(in crate::channel::flow) fn riemann_config(
        &self,
        config: &ChannelConfig,
    ) -> PrismaticRiemannConfig {
        PrismaticRiemannConfig {
            gravity_m_per_s2: config.gravity,
            recovery: self.recovery,
            water_flux_atol_m3_per_s: config.riemann_water_flux_absolute_m3_per_s,
            momentum_flux_atol_m4_per_s2: config.riemann_momentum_flux_absolute_m4_per_s2,
            flux_rtol: config.riemann_flux_relative,
            root: self.normalized_root.clone(),
            max_bracket_expansions: self.max_bracket_expansions,
            initial_simpson_panels: self.initial_simpson_panels,
            max_simpson_panels: self.max_simpson_panels,
            max_refinements: self.max_refinements,
            max_native_query_points: self.max_native_query_points,
        }
    }

    pub(in crate::channel::flow) fn check(&self) -> Result<(), ChannelError> {
        if i64::try_from(self.max_bracket_expansions).is_err() {
            return Err(ChannelError::input(format!(
                "max_bracket_expansions={}: supply a ceiling no greater than i64::MAX",
                self.max_bracket_expansions
            )));
        }
        for (name, value) in [
            ("recovery.depth_atol_m", self.recovery.depth_atol_m),
            ("recovery.depth_rtol", self.recovery.depth_rtol),
            ("normalized_root.atol", self.normalized_root.atol),
            ("normalized_root.rtol", self.normalized_root.rtol),
            (
                "friction_geometry_atol_m4_per_s",
                self.friction_geometry_atol_m4_per_s,
            ),
            ("friction_geometry_rtol", self.friction_geometry_rtol),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ChannelError::input(format!(
                    "{name}={value}: supply a finite nonnegative allowance"
                )));
            }
        }
        if !self.initial_simpson_panels.is_power_of_two()
            || !self.max_simpson_panels.is_power_of_two()
            || self.initial_simpson_panels > self.max_simpson_panels
            || self.normalized_root.max_points == 0
            || self.max_native_query_points == 0
        {
            return Err(ChannelError::input(
                "surveyed work controls: supply ordered positive power-of-two panels and positive point ceilings",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracket_expansion_control_retains_identity_and_midpoint_budget() -> Result<(), ChannelError>
    {
        let original = ChannelSurveyedConfig::default();
        assert_eq!(original.max_bracket_expansions, 4096);
        let mut endpoint_only = original.clone();
        endpoint_only.max_bracket_expansions = 0;
        endpoint_only.check()?;
        assert!(!original.same_bits(&endpoint_only));
        let riemann = endpoint_only.riemann_config(&ChannelConfig::default());
        assert_eq!(riemann.max_bracket_expansions, 0);
        assert_eq!(
            riemann.root.max_iterations,
            original.normalized_root.max_iterations
        );
        Ok(())
    }
}
