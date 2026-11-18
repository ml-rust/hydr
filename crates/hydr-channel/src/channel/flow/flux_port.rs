//! Direct face rates that a flux provider supplies at one reach end.

use super::ChannelError;

/// Water and momentum rates through one reach end face, computed outside the reach.
///
/// The reach inserts both rates at the end face and zeros its own side pressure there.
/// `signal_speed_m_per_s` bounds every wave that enters the reach through that face.
/// Construction rejects nonfinite operands, a negative speed, and empty provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelFluxPort {
    outward_water_rate_m3_per_s: f64,
    outward_momentum_rate_m4_per_s2: f64,
    signal_speed_m_per_s: f64,
    exterior_invariant_m_per_s: Option<f64>,
    provenance: String,
}

impl ChannelFluxPort {
    /// Checks and keeps one held port state.
    ///
    /// - `outward_water_rate_m3_per_s`: water leaving the reach. A negative rate enters.
    /// - `outward_momentum_rate_m4_per_s2`: normal momentum flux along the outward normal.
    ///   It includes normal pressure and every bed correction once.
    /// - `signal_speed_m_per_s`: conservative absolute bound on the face signal speeds.
    /// - `exterior_invariant_m_per_s`: into-reach invariant `u + φ` of the exterior water.
    ///   `None` marks a dry exterior.
    /// - `provenance`: provider, compatible state, and basis of the speed bound.
    ///
    /// Water entering from a dry exterior rejects. A nonzero water rate needs a positive speed.
    pub fn new(
        outward_water_rate_m3_per_s: f64,
        outward_momentum_rate_m4_per_s2: f64,
        signal_speed_m_per_s: f64,
        exterior_invariant_m_per_s: Option<f64>,
        provenance: impl Into<String>,
    ) -> Result<Self, ChannelError> {
        let port = Self {
            outward_water_rate_m3_per_s,
            outward_momentum_rate_m4_per_s2,
            signal_speed_m_per_s,
            exterior_invariant_m_per_s,
            provenance: provenance.into(),
        };
        check_port_values(
            port.outward_water_rate_m3_per_s,
            port.outward_momentum_rate_m4_per_s2,
            port.signal_speed_m_per_s,
            port.exterior_invariant_m_per_s,
        )?;
        check_port_provenance(&port.provenance)?;
        Ok(port)
    }

    /// Water leaving the reach through the end face, in m³/s. A negative rate enters.
    pub fn outward_water_rate_m3_per_s(&self) -> f64 {
        self.outward_water_rate_m3_per_s
    }

    /// Normal momentum flux along the outward normal, including normal pressure, in m⁴/s².
    pub fn outward_momentum_rate_m4_per_s2(&self) -> f64 {
        self.outward_momentum_rate_m4_per_s2
    }

    /// Conservative absolute bound on the face signal speeds, in m/s.
    pub fn signal_speed_m_per_s(&self) -> f64 {
        self.signal_speed_m_per_s
    }

    /// Into-reach Riemann invariant `u + φ` of the exterior water, in m/s. `None` when dry.
    pub fn exterior_invariant_m_per_s(&self) -> Option<f64> {
        self.exterior_invariant_m_per_s
    }

    /// Provider, compatible state, and basis of the speed bound.
    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    #[doc(hidden)]
    pub fn same_bits(&self, other: &Self) -> bool {
        let invariant = match (
            self.exterior_invariant_m_per_s,
            other.exterior_invariant_m_per_s,
        ) {
            (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
            (None, None) => true,
            _ => false,
        };
        invariant
            && [
                self.outward_water_rate_m3_per_s,
                self.outward_momentum_rate_m4_per_s2,
                self.signal_speed_m_per_s,
            ]
            .iter()
            .zip([
                other.outward_water_rate_m3_per_s,
                other.outward_momentum_rate_m4_per_s2,
                other.signal_speed_m_per_s,
            ])
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.provenance == other.provenance
    }
}

/// Rejects nonfinite port operands, a negative speed, inflow from a dry exterior, and a
/// moving flux with zero speed.
pub(super) fn check_port_values(
    water: f64,
    momentum: f64,
    speed: f64,
    invariant: Option<f64>,
) -> Result<(), ChannelError> {
    if !water.is_finite()
        || !momentum.is_finite()
        || !speed.is_finite()
        || invariant.is_some_and(|value| !value.is_finite())
    {
        return Err(ChannelError::input(format!(
            "flux port water={water}, momentum={momentum}, speed={speed}, exterior_invariant={invariant:?}: supply finite port rates, speed and invariant"
        )));
    }
    if speed < 0.0 {
        return Err(ChannelError::input(format!(
            "flux port speed={speed}: supply a nonnegative absolute signal-speed bound"
        )));
    }
    if water < 0.0 && invariant.is_none() {
        return Err(ChannelError::input(format!(
            "flux port water={water} enters from a dry exterior: supply the exterior invariant of the entering water"
        )));
    }
    if water != 0.0 && speed == 0.0 {
        return Err(ChannelError::input(format!(
            "flux port water={water} with speed=0: supply a positive signal-speed bound for a moving flux"
        )));
    }
    Ok(())
}

/// Rejects empty provenance and provenance with control characters.
pub(super) fn check_port_provenance(provenance: &str) -> Result<(), ChannelError> {
    if provenance.trim().is_empty() || provenance.chars().any(char::is_control) {
        return Err(ChannelError::input(format!(
            "flux port provenance={provenance:?}: name the provider, its compatible state and the basis of the speed bound, without control characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn construction_keeps_every_operand_bit() -> TestResult {
        let port = ChannelFluxPort::new(-0.5, 2.25, 3.0, Some(4.5), "test provider")?;
        assert_eq!(
            port.outward_water_rate_m3_per_s().to_bits(),
            (-0.5f64).to_bits()
        );
        assert_eq!(
            port.outward_momentum_rate_m4_per_s2().to_bits(),
            2.25f64.to_bits()
        );
        assert_eq!(port.signal_speed_m_per_s().to_bits(), 3.0f64.to_bits());
        assert_eq!(port.exterior_invariant_m_per_s(), Some(4.5));
        assert_eq!(port.provenance(), "test provider");
        assert!(port.same_bits(&port.clone()));
        let closed = ChannelFluxPort::new(0.0, 0.0, 0.0, None, "closed")?;
        assert!(!port.same_bits(&closed));
        Ok(())
    }

    #[test]
    fn construction_rejects_each_invalid_operand() -> TestResult {
        for (result, fragment) in [
            (
                ChannelFluxPort::new(f64::NAN, 0.0, 1.0, None, "p"),
                "supply finite port rates",
            ),
            (
                ChannelFluxPort::new(0.0, f64::INFINITY, 1.0, None, "p"),
                "supply finite port rates",
            ),
            (
                ChannelFluxPort::new(0.0, 0.0, f64::NAN, None, "p"),
                "supply finite port rates",
            ),
            (
                ChannelFluxPort::new(0.0, 0.0, 1.0, Some(f64::NAN), "p"),
                "supply finite port rates",
            ),
            (
                ChannelFluxPort::new(0.0, 0.0, -1.0, None, "p"),
                "nonnegative absolute signal-speed bound",
            ),
            (
                ChannelFluxPort::new(-1.0, 0.0, 1.0, None, "p"),
                "enters from a dry exterior",
            ),
            (
                ChannelFluxPort::new(1.0, 0.0, 0.0, None, "p"),
                "positive signal-speed bound",
            ),
            (
                ChannelFluxPort::new(0.0, 0.0, 1.0, None, " "),
                "flux port provenance",
            ),
            (
                ChannelFluxPort::new(0.0, 0.0, 1.0, None, "a\nb"),
                "without control characters",
            ),
        ] {
            let Err(error) = result else {
                return Err(format!("admitted a port that must reject with `{fragment}`").into());
            };
            assert!(
                matches!(error, ChannelError::InvalidInput { .. }),
                "{error}"
            );
            let message = error.to_string();
            assert!(message.contains(fragment), "{message} lacks {fragment}");
        }
        Ok(())
    }
}
