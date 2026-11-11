/// Explicit host projection of one accepted scalar endpoint trace.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelBoundaryTrace {
    /// Trace depth in metres.
    pub depth_m: f64,
    /// Full-section wetted area in m².
    pub area_m2: f64,
    /// Velocity toward the endpoint exterior in m/s.
    pub outward_velocity_m_per_s: f64,
    /// Full-section discharge toward the endpoint exterior in m³/s.
    pub outward_discharge_m3_per_s: f64,
    /// Physical longitudinal momentum flux in m⁴/s².
    pub momentum_flux_m4_per_s2: f64,
    /// Conservative trace signal-speed bound in m/s.
    pub signal_speed_bound_m_per_s: f64,
    /// Absolute wave-curve velocity defect bound in m/s.
    pub velocity_compatibility_abs_bound_m_per_s: f64,
    /// Absolute induced water-flux defect bound in m³/s.
    pub water_compatibility_abs_bound_m3_per_s: f64,
    /// Absolute induced momentum-flux defect bound in m⁴/s².
    pub momentum_compatibility_abs_bound_m4_per_s2: f64,
    /// Accepted water-flux enclosure width in m³/s.
    pub water_flux_width_m3_per_s: f64,
    /// Accepted momentum-flux enclosure width in m⁴/s².
    pub momentum_flux_width_m4_per_s2: f64,
    /// True for tolerance-based admission at the original geometric cap.
    pub cap_compatibility_approximation: bool,
    /// Maximum iterations among contributing batched root calculations.
    pub root_iterations: usize,
}

impl ChannelBoundaryTrace {
    pub(super) fn same_bits(&self, other: &Self) -> bool {
        self.cap_compatibility_approximation == other.cap_compatibility_approximation
            && self.root_iterations == other.root_iterations
            && self.depth_m.to_bits() == other.depth_m.to_bits()
            && self.area_m2.to_bits() == other.area_m2.to_bits()
            && self.outward_velocity_m_per_s.to_bits() == other.outward_velocity_m_per_s.to_bits()
            && self.outward_discharge_m3_per_s.to_bits()
                == other.outward_discharge_m3_per_s.to_bits()
            && self.momentum_flux_m4_per_s2.to_bits() == other.momentum_flux_m4_per_s2.to_bits()
            && self.signal_speed_bound_m_per_s.to_bits()
                == other.signal_speed_bound_m_per_s.to_bits()
            && self.velocity_compatibility_abs_bound_m_per_s.to_bits()
                == other.velocity_compatibility_abs_bound_m_per_s.to_bits()
            && self.water_compatibility_abs_bound_m3_per_s.to_bits()
                == other.water_compatibility_abs_bound_m3_per_s.to_bits()
            && self.momentum_compatibility_abs_bound_m4_per_s2.to_bits()
                == other.momentum_compatibility_abs_bound_m4_per_s2.to_bits()
            && self.water_flux_width_m3_per_s.to_bits() == other.water_flux_width_m3_per_s.to_bits()
            && self.momentum_flux_width_m4_per_s2.to_bits()
                == other.momentum_flux_width_m4_per_s2.to_bits()
    }
}
