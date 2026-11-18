//! Checked SI geometry and separately cached native-derived closure.

use super::ReservoirError;
use crate::runoff::numerics::reservoir::EXPONENT;

const FOOT: f64 = 0.3048;
const SQUARE_FOOT: f64 = FOOT * FOOT;

/// Actual storage support and native-derived nonlinear reservoir law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReservoirParameters {
    si: [f64; 6],
    native: [f64; 6],
    alpha_native: Option<f64>,
    alpha_si: Option<f64>,
}
impl ReservoirParameters {
    /// Check SI area/class-area, width, slope, Manning coefficient, and depression depth.
    pub fn from_si(
        area_m2: f64,
        class_area_m2: f64,
        width_m: f64,
        slope: f64,
        manning: f64,
        depression_depth_m: f64,
    ) -> Result<Self, ReservoirError> {
        let si = [
            area_m2,
            class_area_m2,
            width_m,
            slope,
            manning,
            depression_depth_m,
        ];
        let native = [
            area_m2 / SQUARE_FOOT,
            class_area_m2 / SQUARE_FOOT,
            width_m / FOOT,
            slope,
            manning,
            depression_depth_m / FOOT,
        ];
        Self::checked(si, native)
    }
    /// Check native square-foot areas, foot lengths, slope, and Manning coefficient.
    pub fn from_native(
        area_ft2: f64,
        class_area_ft2: f64,
        width_ft: f64,
        slope: f64,
        manning: f64,
        depression_depth_ft: f64,
    ) -> Result<Self, ReservoirError> {
        let native = [
            area_ft2,
            class_area_ft2,
            width_ft,
            slope,
            manning,
            depression_depth_ft,
        ];
        let si = [
            area_ft2 * SQUARE_FOOT,
            class_area_ft2 * SQUARE_FOOT,
            width_ft * FOOT,
            slope,
            manning,
            depression_depth_ft * FOOT,
        ];
        Self::checked(si, native)
    }
    fn checked(si: [f64; 6], native: [f64; 6]) -> Result<Self, ReservoirError> {
        for geometry in [si, native] {
            if geometry.iter().any(|x| !x.is_finite() || *x < 0.0)
                || geometry[0] <= 0.0
                || geometry[1] <= 0.0
                || geometry[0] > geometry[1]
            {
                return Err(ReservoirError::input(format!(
                    "SI={si:?}, native={native:?}: require finite nonnegative geometry, positive areas, and area<=classArea"
                )));
            }
        }
        for index in [0, 1, 2, 5] {
            if (si[index] > 0.0) != (native[index] > 0.0) {
                return Err(ReservoirError::input(
                    "positive geometry loses its SI/native representation. Rescale support",
                ));
            }
        }
        let [_, class_area, width, slope, n, _] = native;
        if n == 0.0 {
            if width == 0.0 || slope == 0.0 {
                return Err(ReservoirError::input(
                    "zero roughness requires positive width and slope for explicit instantaneous overflow",
                ));
            }
            return Ok(Self {
                si,
                native,
                alpha_native: None,
                alpha_si: None,
            });
        }
        let alpha = if width == 0.0 || slope == 0.0 {
            0.0
        } else {
            let mut coefficient = 1.49 * width;
            for value in [coefficient, coefficient / class_area] {
                if !value.is_finite() || value <= 0.0 {
                    return Err(ReservoirError::input(
                        "native coefficient overflows or underflows. Rescale physical geometry",
                    ));
                }
            }
            coefficient /= class_area;
            coefficient *= slope.sqrt();
            if !coefficient.is_finite() || coefficient <= 0.0 {
                return Err(ReservoirError::input(
                    "native slope-weighted coefficient is unrepresentable. Rescale geometry",
                ));
            }
            coefficient /= n;
            if !coefficient.is_finite() || coefficient <= 0.0 {
                return Err(ReservoirError::input(
                    "positive native coefficient is unrepresentable. Rescale roughness or geometry",
                ));
            }
            coefficient
        };
        let alpha_si = alpha * numr::algorithm::correctly_rounded::pow(FOOT, 1.0 - EXPONENT);
        if !alpha_si.is_finite() || (alpha > 0.0 && alpha_si <= 0.0) {
            return Err(ReservoirError::input(
                "SI coefficient overflows or underflows. Rescale physical geometry",
            ));
        }
        Ok(Self {
            si,
            native,
            alpha_native: Some(alpha),
            alpha_si: Some(alpha_si),
        })
    }
    /// Actual storage area in m².
    pub fn area_m2(self) -> f64 {
        self.si[0]
    }
    /// Combined class area used in the native coefficient, in m².
    pub fn class_area_m2(self) -> f64 {
        self.si[1]
    }
    /// Closure width in metres.
    pub fn width_m(self) -> f64 {
        self.si[2]
    }
    /// Dimensionless slope.
    pub fn slope(self) -> f64 {
        self.si[3]
    }
    /// Manning coefficient supplied to the native-derived closure.
    pub fn manning(self) -> f64 {
        self.si[4]
    }
    /// Depression depth in metres.
    pub fn depression_depth_m(self) -> f64 {
        self.si[5]
    }
    /// SI depth-law coefficient. Undefined for instantaneous overflow.
    pub fn coefficient_si(self) -> Option<f64> {
        self.alpha_si
    }
    /// Original feet-based coefficient. Undefined for instantaneous overflow.
    pub fn coefficient_native(self) -> Option<f64> {
        self.alpha_native
    }
    /// Whether zero roughness explicitly selects instantaneous overflow.
    pub fn is_instantaneous(self) -> bool {
        self.alpha_si.is_none()
    }
    /// Whether positive roughness and zero width/slope explicitly select closed storage.
    pub fn is_closed(self) -> bool {
        self.alpha_si == Some(0.0)
    }
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.si
            .iter()
            .chain(&self.native)
            .zip(other.si.iter().chain(&other.native))
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.alpha_native.map(f64::to_bits) == other.alpha_native.map(f64::to_bits)
            && self.alpha_si.map(f64::to_bits) == other.alpha_si.map(f64::to_bits)
    }
}
