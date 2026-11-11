use super::ChannelError;

/// Scalar endpoint compatibility controls, separate from conservation and flux-width tolerances.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelBoundaryConfig {
    /// Absolute full-section discharge compatibility allowance in m³/s.
    pub discharge_absolute_m3_per_s: f64,
    /// Relative full-section discharge compatibility allowance.
    pub discharge_relative: f64,
    /// Absolute momentum-flux compatibility allowance in m⁴/s².
    pub momentum_absolute_m4_per_s2: f64,
    /// Relative momentum-flux compatibility allowance.
    pub momentum_relative: f64,
    /// Relative wave-curve velocity allowance without a dimensional floor.
    pub velocity_relative: f64,
    /// Relative characteristic separation, strictly between zero and one.
    pub critical_relative: f64,
}

impl Default for ChannelBoundaryConfig {
    fn default() -> Self {
        Self {
            discharge_absolute_m3_per_s: 1e-12,
            discharge_relative: 1e-12,
            momentum_absolute_m4_per_s2: 1e-12,
            momentum_relative: 1e-12,
            velocity_relative: 1e-12,
            critical_relative: 64.0 * f64::EPSILON,
        }
    }
}

impl ChannelBoundaryConfig {
    pub(super) fn same_bits(&self, other: &Self) -> bool {
        self.discharge_absolute_m3_per_s.to_bits() == other.discharge_absolute_m3_per_s.to_bits()
            && self.discharge_relative.to_bits() == other.discharge_relative.to_bits()
            && self.momentum_absolute_m4_per_s2.to_bits()
                == other.momentum_absolute_m4_per_s2.to_bits()
            && self.momentum_relative.to_bits() == other.momentum_relative.to_bits()
            && self.velocity_relative.to_bits() == other.velocity_relative.to_bits()
            && self.critical_relative.to_bits() == other.critical_relative.to_bits()
    }

    pub(super) fn check(&self) -> Result<(), ChannelError> {
        for (name, value) in [
            (
                "discharge_absolute_m3_per_s",
                self.discharge_absolute_m3_per_s,
            ),
            ("discharge_relative", self.discharge_relative),
            (
                "momentum_absolute_m4_per_s2",
                self.momentum_absolute_m4_per_s2,
            ),
            ("momentum_relative", self.momentum_relative),
            ("velocity_relative", self.velocity_relative),
            ("critical_relative", self.critical_relative),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ChannelError::input(format!(
                    "boundary.{name}={value}: supply a finite nonnegative tolerance"
                )));
            }
        }
        for (name, absolute, relative) in [
            (
                "discharge",
                self.discharge_absolute_m3_per_s,
                self.discharge_relative,
            ),
            (
                "momentum",
                self.momentum_absolute_m4_per_s2,
                self.momentum_relative,
            ),
        ] {
            if absolute == 0.0 && relative == 0.0 {
                return Err(ChannelError::input(format!(
                    "boundary.{name} absolute={absolute}, relative={relative}: supply at least one positive tolerance"
                )));
            }
        }
        if self.velocity_relative <= 0.0
            || self.critical_relative <= 0.0
            || self.critical_relative >= 1.0
        {
            return Err(ChannelError::input(format!(
                "boundary.velocity_relative={}, critical_relative={}: supply positive velocity tolerance and critical separation strictly between zero and one",
                self.velocity_relative, self.critical_relative
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ChannelBoundaryConfig;

    #[test]
    fn compatibility_controls_reject_nonfinite_and_degenerate_allowances() {
        let defaults = ChannelBoundaryConfig::default();
        assert!(defaults.check().is_ok());
        for invalid in [-1.0, f64::INFINITY, f64::NAN] {
            for index in 0..6 {
                let mut config = defaults.clone();
                match index {
                    0 => config.discharge_absolute_m3_per_s = invalid,
                    1 => config.discharge_relative = invalid,
                    2 => config.momentum_absolute_m4_per_s2 = invalid,
                    3 => config.momentum_relative = invalid,
                    4 => config.velocity_relative = invalid,
                    _ => config.critical_relative = invalid,
                }
                assert!(config.check().is_err());
            }
        }
        for index in 0..4 {
            let mut config = defaults.clone();
            match index {
                0 => config.discharge_absolute_m3_per_s = 0.0,
                1 => config.discharge_relative = 0.0,
                2 => config.momentum_absolute_m4_per_s2 = 0.0,
                _ => config.momentum_relative = 0.0,
            }
            assert!(config.check().is_ok());
        }
        for index in 0..5 {
            let mut config = defaults.clone();
            match index {
                0 => {
                    config.discharge_absolute_m3_per_s = 0.0;
                    config.discharge_relative = 0.0;
                }
                1 => {
                    config.momentum_absolute_m4_per_s2 = 0.0;
                    config.momentum_relative = 0.0;
                }
                2 => config.velocity_relative = 0.0,
                3 => config.critical_relative = 0.0,
                _ => config.critical_relative = 1.0,
            }
            assert!(config.check().is_err());
        }
    }

    #[test]
    fn identity_retains_every_control_bit() {
        let original = ChannelBoundaryConfig::default();
        for index in 0..6 {
            let mut changed = original.clone();
            let value = match index {
                0 => &mut changed.discharge_absolute_m3_per_s,
                1 => &mut changed.discharge_relative,
                2 => &mut changed.momentum_absolute_m4_per_s2,
                3 => &mut changed.momentum_relative,
                4 => &mut changed.velocity_relative,
                _ => &mut changed.critical_relative,
            };
            *value = f64::from_bits(value.to_bits() + 1);
            assert!(!original.same_bits(&changed));
        }
        let mut positive = original;
        positive.discharge_relative = 0.0;
        let mut negative = positive.clone();
        negative.discharge_relative = -0.0;
        assert!(positive.check().is_ok());
        assert!(negative.check().is_ok());
        assert!(!positive.same_bits(&negative));
    }
}
