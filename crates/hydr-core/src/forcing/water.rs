/// Original supplied water magnitude, without conversion between time-support variants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WaterAmount {
    /// Constant flow over an explicitly covered interval.
    Rate {
        /// Finite nonnegative prescribed flow in m³/s.
        rate_m3_per_s: f64,
    },
    /// Integrated volume for the exact forcing span.
    IntervalVolume {
        /// Finite nonnegative original volume in m³.
        volume_m3: f64,
    },
}

impl WaterAmount {
    #[doc(hidden)]
    #[inline]
    pub fn scalar(self) -> f64 {
        match self {
            Self::Rate { rate_m3_per_s } => rate_m3_per_s,
            Self::IntervalVolume { volume_m3 } => volume_m3,
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn is_rate(self) -> bool {
        matches!(self, Self::Rate { .. })
    }

    #[doc(hidden)]
    #[inline]
    pub fn same_bits(&self, other: &Self) -> bool {
        self.is_rate() == other.is_rate() && self.scalar().to_bits() == other.scalar().to_bits()
    }
}
