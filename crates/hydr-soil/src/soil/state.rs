//! Explicit full native soil memory and interval input/output units.

use super::{ModifiedGreenAmptParameters, SoilError};

/// Complete native antecedent state. Signed event memory and timers are retained.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativeSoilState {
    /// Initial moisture deficit, dimensionless.
    pub imd: f64,
    /// Native event infiltration memory in feet, including legal negative values.
    pub f_ft: f64,
    /// Upper-zone infiltration memory in feet.
    pub fu_ft: f64,
    /// Independent native saturation flag.
    pub saturated: bool,
    /// Native recovery timer in seconds, including expired negative values.
    pub t_s: f64,
}

impl NativeSoilState {
    #[doc(hidden)]
    pub fn check(
        self,
        parameters: ModifiedGreenAmptParameters,
        index: usize,
    ) -> Result<(), SoilError> {
        if ![self.imd, self.f_ft, self.fu_ft, self.t_s]
            .iter()
            .all(|x| x.is_finite())
            || self.imd < 0.0
            || self.imd > parameters.deficit_ceiling
            || self.fu_ft < 0.0
            || self.fu_ft > parameters.capacity
        {
            return Err(SoilError::input(format!(
                "cell={index}, state={self:?}: require finite state, IMD in [0,{}], and Fu in [0,{}]",
                parameters.deficit_ceiling, parameters.capacity
            )));
        }
        Ok(())
    }
}

/// Nonnegative incoming rain/runon excludes ponding and evaporation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilInput {
    /// Nonnegative incoming rate in metres/second.
    pub incoming_rate_m_per_s: f64,
    /// Nonnegative ponded depth in metres.
    pub ponded_depth_m: f64,
}

/// Exact native interoperability input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativeSoilInput {
    /// Nonnegative incoming rate in feet/second.
    pub incoming_rate_ft_per_s: f64,
    /// Nonnegative ponded depth in feet.
    pub ponded_depth_ft: f64,
}

/// Raw empirical infiltration for one interval. No lifetime or surface-water ledger is implied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilOutput {
    /// Native interval rate in feet/second.
    pub native_rate_ft_per_s: f64,
    /// Native interval infiltration depth in feet.
    pub native_depth_ft: f64,
    /// SI interval rate in metres/second.
    pub rate_m_per_s: f64,
    /// SI interval depth from SI rate times interval duration.
    pub depth_m: f64,
}
