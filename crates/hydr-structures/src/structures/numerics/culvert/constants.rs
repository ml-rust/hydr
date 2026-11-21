//! Frozen barrel constants as outward binary64 bounds.
//!
//! Each `[lower, upper]` pair holds the bits of a rational or transcendental value rounded down
//! and up. A unit test checks every pair against the certificate rows bit for bit.

/// F64 below and above pi.
pub(crate) const PI: [u64; 2] = [0x400921fb54442d18, 0x400921fb54442d19];

/// Terms summed before the certified tail of each shallow series.
pub(crate) const SERIES_TERMS: usize = 14;

/// `a_k = (2k)! / (4^k (k!)^2 (2k + 1))`, the `asin` series coefficients, all positive.
pub(crate) const ASIN: [[u64; 2]; SERIES_TERMS + 1] = [
    [0x3ff0000000000000, 0x3ff0000000000000],
    [0x3fc5555555555555, 0x3fc5555555555556],
    [0x3fb3333333333333, 0x3fb3333333333334],
    [0x3fa6db6db6db6db6, 0x3fa6db6db6db6db7],
    [0x3f9f1c71c71c71c7, 0x3f9f1c71c71c71c8],
    [0x3f96e8ba2e8ba2e8, 0x3f96e8ba2e8ba2e9],
    [0x3f91c4ec4ec4ec4e, 0x3f91c4ec4ec4ec4f],
    [0x3f8c999999999999, 0x3f8c99999999999a],
    [0x3f87a87878787878, 0x3f87a87878787879],
    [0x3f83fde50d79435e, 0x3f83fde50d79435f],
    [0x3f812ef3cf3cf3cf, 0x3f812ef3cf3cf3d0],
    [0x3f7df3bd37a6f4de, 0x3f7df3bd37a6f4df],
    [0x3f7a6863d70a3d70, 0x3f7a6863d70a3d71],
    [0x3f7782dda12f684b, 0x3f7782dda12f684c],
    [0x3f751ba308d3dcb0, 0x3f751ba308d3dcb1],
];

/// `c_k = 2 (-1)^k C(1/2, k) / (k + 3/2)`, the area series coefficients.
/// `c_0 = 4/3`. Every later coefficient is negative, with decreasing magnitude.
pub(crate) const AREA: [[u64; 2]; SERIES_TERMS + 1] = [
    [0x3ff5555555555555, 0x3ff5555555555556],
    [0xbfd999999999999a, 0xbfd9999999999999],
    [0xbfb2492492492493, 0xbfb2492492492492],
    [0xbf9c71c71c71c71d, 0xbf9c71c71c71c71c],
    [0xbf8d1745d1745d18, 0xbf8d1745d1745d17],
    [0xbf813b13b13b13b2, 0xbf813b13b13b13b1],
    [0xbf76666666666667, 0xbf76666666666666],
    [0xbf6f0f0f0f0f0f10, 0xbf6f0f0f0f0f0f0f],
    [0xbf669435e50d7944, 0xbf669435e50d7943],
    [0xbf61061861861862, 0xbf61061861861861],
    [0xbf5a6c8590b21643, 0xbf5a6c8590b21642],
    [0xbf54feb851eb851f, 0xbf54feb851eb851e],
    [0xbf510284bda12f69, 0xbf510284bda12f68],
    [0xbf4c04d3dcb08d3e, 0xbf4c04d3dcb08d3d],
    [0xbf47672108421085, 0xbf47672108421084],
];

/// `c_k / (k + 5/2)`, the moment series coefficients, with the signs of `c_k`.
pub(crate) const MOMENT: [[u64; 2]; SERIES_TERMS + 1] = [
    [0x3fe1111111111111, 0x3fe1111111111112],
    [0xbfbd41d41d41d41e, 0xbfbd41d41d41d41d],
    [0xbf90410410410411, 0xbf90410410410410],
    [0xbf74afd6a052bf5b, 0xbf74afd6a052bf5a],
    [0xbf61e6efe35b4cfb, 0xbf61e6efe35b4cfa],
    [0xbf52612612612613, 0xbf52612612612612],
    [0xbf45151515151516, 0xbf45151515151515],
    [0xbf3a27a0e442936b, 0xbf3a27a0e442936a],
    [0xbf3133f84cfe1340, 0xbf3133f84cfe133f],
    [0xbf27af6fd5992d0e, 0xbf27af6fd5992d0d],
    [0xbf20e94b3de29d9c, 0xbf20e94b3de29d9b],
    [0xbf18e209dc598154, 0xbf18e209dc598153],
    [0xbf12c4fc6750c18e, 0xbf12c4fc6750c18d],
    [0xbf0cec3588f84fbc, 0xbf0cec3588f84fbb],
    [0xbf06b19465194652, 0xbf06b19465194651],
];

/// Lower end `2^-30` of the critical-depth and normal-depth domains in `s = y/D`.
pub(crate) const CRITICAL_FLOOR: f64 = 1.0 / 1_073_741_824.0;

/// Upper end `1 - 2^-53` of the circle critical-depth domain in `s = y/D`.
/// The box domain ends at the crown, `s = 1`.
pub(crate) const CIRCLE_CRITICAL_TOP: f64 = 1.0 - f64::EPSILON / 2.0;

/// Lower bound `5.4` of `d r_c / ds` for a circle on its critical domain.
/// The minimum over `(0, 1)` is near `5.41288` at `s = 0.778`.
pub(crate) const CIRCLE_CRITICAL_SLOPE: f64 = 5.4;

/// Lower bound `3` of `d r_c / ds = 3/s` for a box on `(0, 1]`.
pub(crate) const BOX_CRITICAL_SLOPE: f64 = 3.0;

/// Top `s_N = 0.93818` of the circle normal-depth domain in `s = y/D`, below the conveyance peak.
/// The box domain ends at the crown, `s = 1`.
pub(crate) const RISING_LIMB_TOP: f64 = 0.93818;

/// Lower bound `5.42e-5` of `d r_n / ds = 5T/A - 4/(TP)` for a unit circle on `[2^-30, s_N]`.
pub(crate) const CIRCLE_NORMAL_SLOPE: f64 = 5.42e-5;

/// Lower bound `3` of `d r_n / ds = 5/s - 4D/(B + 2Ds)` for a box on `(0, 1]`.
pub(crate) const BOX_NORMAL_SLOPE: f64 = 3.0;

/// Adjacent F64 below and above the unit-circle conveyance peak `s_K`.
/// Conveyance rises up to the lower bound and falls from the upper bound to the crown.
pub(crate) const CONVEYANCE_PEAK: [u64; 2] = [0x3fee05949d2433b9, 0x3fee05949d2433ba];

/// Bounds of `ln k_max`, the peak conveyance `A^(5/3) / P^(2/3)` of a unit circle.
pub(crate) const LOG_PEAK_CONVEYANCE: [u64; 2] = [0xbff17c0a73c51e10, 0xbff17c0a73c51e0d];

#[cfg(test)]
mod tests {
    use super::super::certificate_rows::{CERTIFICATE_SHA256, ROWS};
    use super::*;

    fn row(name: &str) -> Option<[u64; 2]> {
        ROWS.iter()
            .find(|(key, _, _)| *key == name)
            .map(|(_, lower, upper)| [*lower, *upper])
    }

    #[test]
    fn constants_equal_certificate_rows_bit_for_bit() {
        assert_eq!(
            CERTIFICATE_SHA256,
            "db5eff303d3afcf03abe0b81cfcc34ee20dd72e042b5282aadcfafac029a2b15"
        );
        assert_eq!(Some(PI), row("pi"));
        let single = |value: f64| Some([value.to_bits(); 2]);
        assert_eq!(single(CIRCLE_CRITICAL_SLOPE), row("critical_slope"));
        assert_eq!(single(BOX_CRITICAL_SLOPE), row("box_critical_slope"));
        assert_eq!(single(RISING_LIMB_TOP), row("rising_limb_top"));
        assert_eq!(single(CIRCLE_NORMAL_SLOPE), row("normal_slope"));
        assert_eq!(single(BOX_NORMAL_SLOPE), row("box_normal_slope"));
        assert_eq!(Some(CONVEYANCE_PEAK), row("conveyance_peak"));
        assert_eq!(Some(LOG_PEAK_CONVEYANCE), row("log_peak_conveyance"));
        assert_eq!(
            Some([CRITICAL_FLOOR.to_bits(), CIRCLE_CRITICAL_TOP.to_bits()]),
            row("critical_domain")
        );
        for (name, table) in [("asin", &ASIN), ("area", &AREA), ("moment", &MOMENT)] {
            for (k, pair) in table.iter().enumerate() {
                assert_eq!(Some(*pair), row(&format!("{name}[{k}]")), "{name}[{k}]");
            }
        }
    }

    #[test]
    fn bounds_are_ordered_and_pi_is_the_rounded_constant() {
        assert_eq!(f64::from_bits(PI[0]), std::f64::consts::PI);
        let pairs = [&PI, &CONVEYANCE_PEAK, &LOG_PEAK_CONVEYANCE];
        for pair in ASIN.iter().chain(&AREA).chain(&MOMENT).chain(pairs) {
            assert!(f64::from_bits(pair[0]) <= f64::from_bits(pair[1]));
        }
        let [peak_lower, peak_upper] = CONVEYANCE_PEAK.map(f64::from_bits);
        assert_eq!(peak_lower.next_up(), peak_upper);
        assert!(RISING_LIMB_TOP < peak_lower);
    }
}
