use super::error::{HydraulicError, HydraulicResult};
use crate::channel::section::PrismaticRecoveryConfig;
use solvr::optimize::CertifiedBisectionOptions;

/// Accuracy allowances and native work ceilings for surveyed Godunov fluxes.
#[derive(Clone, Debug)]
pub struct PrismaticRiemannConfig {
    /// Finite positive gravity, in metres per second squared. Default: `9.81`.
    pub gravity_m_per_s2: f64,
    /// Recovered depth-width allowances. Both absolute and relative defaults are `1e-12`.
    pub recovery: PrismaticRecoveryConfig,
    /// Absolute water-flux width allowance, in cubic metres per second. Default: `1e-9`.
    pub water_flux_atol_m3_per_s: f64,
    /// Absolute momentum-flux width allowance, in metres to the fourth per second squared. Default: `1e-8`.
    pub momentum_flux_atol_m4_per_s2: f64,
    /// Relative flux-width allowance times the largest absolute endpoint. Default: `1e-10`.
    pub flux_rtol: f64,
    /// Normalized-depth root tolerances and work ceilings. Root convergence alone never accepts a flux.
    pub root: CertifiedBisectionOptions,
    /// Initial Simpson panels per active band contribution. Positive power of two. Default: `4`.
    pub initial_simpson_panels: usize,
    /// Maximum Simpson panels per active band contribution. Positive power of two. Default: `4096`.
    pub max_simpson_panels: usize,
    /// Maximum panel doublings after the initial solve. Default: `10`.
    pub max_refinements: usize,
    /// Cumulative residual, Simpson value, and derivative callback point ceiling. Default: `1 << 24`.
    pub max_native_query_points: usize,
}

impl Default for PrismaticRiemannConfig {
    fn default() -> Self {
        Self {
            gravity_m_per_s2: 9.81,
            recovery: PrismaticRecoveryConfig::default(),
            water_flux_atol_m3_per_s: 1e-9,
            momentum_flux_atol_m4_per_s2: 1e-8,
            flux_rtol: 1e-10,
            root: CertifiedBisectionOptions::default(),
            initial_simpson_panels: 4,
            max_simpson_panels: 4096,
            max_refinements: 10,
            max_native_query_points: 1 << 24,
        }
    }
}

impl PrismaticRiemannConfig {
    pub(crate) fn check(&self) -> HydraulicResult<()> {
        let invalid = |context: String| {
            HydraulicError::Geometry(crate::channel::section::CrossSectionError::InvalidInput {
                section: None,
                context,
            })
        };
        if !self.gravity_m_per_s2.is_finite() || self.gravity_m_per_s2 <= 0.0 {
            return Err(invalid(format!(
                "gravity_m_per_s2={}: supply finite positive gravity",
                self.gravity_m_per_s2
            )));
        }
        for (name, value) in [
            ("depth_atol_m", self.recovery.depth_atol_m),
            ("depth_rtol", self.recovery.depth_rtol),
            ("water_flux_atol_m3_per_s", self.water_flux_atol_m3_per_s),
            (
                "momentum_flux_atol_m4_per_s2",
                self.momentum_flux_atol_m4_per_s2,
            ),
            ("flux_rtol", self.flux_rtol),
            ("root.atol", self.root.atol),
            ("root.rtol", self.root.rtol),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid(format!(
                    "{name}={value}: supply a finite nonnegative allowance"
                )));
            }
        }
        if !self.initial_simpson_panels.is_power_of_two()
            || !self.max_simpson_panels.is_power_of_two()
            || self.initial_simpson_panels > self.max_simpson_panels
        {
            return Err(invalid(format!(
                "initial_simpson_panels={}, max_simpson_panels={}: supply ordered positive powers of two",
                self.initial_simpson_panels, self.max_simpson_panels
            )));
        }
        Ok(())
    }
}
