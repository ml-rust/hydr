/// Certified reservoir depth and passive dissipation bounds for one accepted endpoint trace.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelReservoirTrace {
    /// Lower certified root depth above the endpoint bed, in metres.
    pub depth_lower_m: f64,
    /// Upper certified root depth above the endpoint bed, in metres.
    pub depth_upper_m: f64,
    /// Absolute delivered-state reservoir head defect bound in metres.
    pub head_compatibility_abs_bound_m: f64,
    /// Lower passive dissipation head bound in metres.
    pub dissipation_head_lower_m: f64,
    /// Upper passive dissipation head bound in metres.
    ///
    /// Multiply dissipation head by gravity and absolute discharge for dissipation power per density.
    pub dissipation_head_upper_m: f64,
}

impl ChannelReservoirTrace {
    fn same_bits(&self, other: &Self) -> bool {
        self.depth_lower_m.to_bits() == other.depth_lower_m.to_bits()
            && self.depth_upper_m.to_bits() == other.depth_upper_m.to_bits()
            && self.head_compatibility_abs_bound_m.to_bits()
                == other.head_compatibility_abs_bound_m.to_bits()
            && self.dissipation_head_lower_m.to_bits() == other.dissipation_head_lower_m.to_bits()
            && self.dissipation_head_upper_m.to_bits() == other.dissipation_head_upper_m.to_bits()
    }
}

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
    /// Reservoir certificates for an actual quiescent-reservoir endpoint, otherwise `None`.
    pub reservoir: Option<ChannelReservoirTrace>,
    /// Maximum iterations among contributing batched root calculations.
    pub root_iterations: usize,
}

impl ChannelBoundaryTrace {
    pub(super) fn same_bits(&self, other: &Self) -> bool {
        self.cap_compatibility_approximation == other.cap_compatibility_approximation
            && match (&self.reservoir, &other.reservoir) {
                (None, None) => true,
                (Some(left), Some(right)) => left.same_bits(right),
                _ => false,
            }
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
