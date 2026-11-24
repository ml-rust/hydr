//! Circular and rectangular box conduit hydraulics with continuous Preissmann surcharge slot.

use super::super::DrainageError;
use std::f64::consts::PI;

/// Closed-conduit cross-section shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PipeCrossSection {
    /// Circular pipe of interior diameter `diameter_m` (`m`, `> 0`).
    Circular {
        /// Interior pipe diameter in meters.
        diameter_m: f64,
    },
    /// Rectangular box conduit of interior `width_m` by `height_m` (`m`, `> 0`).
    Box {
        /// Interior width in meters.
        width_m: f64,
        /// Interior height in meters.
        height_m: f64,
    },
}

/// Hydraulic section properties at a given depth `y` (free-surface or Preissmann-slot pressurized).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConduitHydraulics {
    /// Wetted flow area `A(y)` (`m²`).
    pub area_m2: f64,
    /// Free-surface or Preissmann-slot top width `T(y)` (`m`).
    pub top_width_m: f64,
    /// Wetted perimeter `P(y)` (`m`).
    pub wetted_perimeter_m: f64,
    /// Hydraulic radius `R_h = A / P` (`m`).
    pub hydraulic_radius_m: f64,
    /// True when `y > full_depth_m` (pressurized surcharge in Preissmann slot).
    pub pressurized: bool,
}

impl PipeCrossSection {
    /// Validates cross-section dimensions.
    pub fn validate(self) -> Result<(), DrainageError> {
        match self {
            Self::Circular { diameter_m } => {
                if !diameter_m.is_finite() || diameter_m <= 0.0 {
                    return Err(DrainageError::input(format!(
                        "circular conduit diameter_m must be finite and > 0, got {diameter_m}"
                    )));
                }
            }
            Self::Box { width_m, height_m } => {
                if !width_m.is_finite()
                    || width_m <= 0.0
                    || !height_m.is_finite()
                    || height_m <= 0.0
                {
                    return Err(DrainageError::input(format!(
                        "box conduit width_m ({width_m}) and height_m ({height_m}) must be > 0"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Full interior crown height (`m`).
    pub fn full_depth_m(self) -> f64 {
        match self {
            Self::Circular { diameter_m } => diameter_m,
            Self::Box { height_m, .. } => height_m,
        }
    }

    /// Full interior cross-sectional area (`m²`).
    pub fn full_area_m2(self) -> f64 {
        match self {
            Self::Circular { diameter_m } => 0.25 * PI * diameter_m * diameter_m,
            Self::Box { width_m, height_m } => width_m * height_m,
        }
    }

    /// Full wetted perimeter (`m`).
    pub fn full_perimeter_m(self) -> f64 {
        match self {
            Self::Circular { diameter_m } => PI * diameter_m,
            Self::Box { width_m, height_m } => 2.0 * (width_m + height_m),
        }
    }

    /// Preissmann slot width `T_s = g * A_full / c_slot²` (`m`).
    pub fn slot_width_m(self, gravity: f64, slot_celerity_m_per_s: f64) -> f64 {
        let a_full = self.full_area_m2();
        (gravity * a_full / (slot_celerity_m_per_s * slot_celerity_m_per_s)).max(1e-4)
    }

    /// Evaluates continuous hydraulics at depth `depth_m` (`>= 0`) with Preissmann slot above crown.
    pub fn hydraulics(
        self,
        depth_m: f64,
        gravity: f64,
        slot_celerity_m_per_s: f64,
    ) -> ConduitHydraulics {
        let y = depth_m.max(0.0);
        let y_full = self.full_depth_m();
        let t_slot = self.slot_width_m(gravity, slot_celerity_m_per_s);

        if y <= 1e-9 {
            return ConduitHydraulics {
                area_m2: 0.0,
                top_width_m: t_slot,
                wetted_perimeter_m: 0.0,
                hydraulic_radius_m: 0.0,
                pressurized: false,
            };
        }

        if y >= y_full {
            let a_full = self.full_area_m2();
            let p_full = self.full_perimeter_m();
            let extra = y - y_full;
            let area = a_full + t_slot * extra;
            // Perimeter stays at full pipe perimeter (slot walls are frictionless).
            let rh = a_full / p_full;
            return ConduitHydraulics {
                area_m2: area,
                top_width_m: t_slot,
                wetted_perimeter_m: p_full,
                hydraulic_radius_m: rh,
                pressurized: true,
            };
        }

        match self {
            Self::Circular { diameter_m } => {
                let r = 0.5 * diameter_m;
                let cos_arg = ((r - y) / r).clamp(-1.0, 1.0);
                let theta = 2.0 * cos_arg.acos();
                let area = 0.5 * r * r * (theta - theta.sin());
                let top_width = (diameter_m * (0.5 * theta).sin()).max(t_slot);
                let perimeter = r * theta;
                let rh = if perimeter > 1e-12 {
                    area / perimeter
                } else {
                    0.0
                };
                ConduitHydraulics {
                    area_m2: area,
                    top_width_m: top_width,
                    wetted_perimeter_m: perimeter,
                    hydraulic_radius_m: rh,
                    pressurized: false,
                }
            }
            Self::Box { width_m, .. } => {
                let area = width_m * y;
                let top_width = width_m.max(t_slot);
                let perimeter = width_m + 2.0 * y;
                let rh = area / perimeter;
                ConduitHydraulics {
                    area_m2: area,
                    top_width_m: top_width,
                    wetted_perimeter_m: perimeter,
                    hydraulic_radius_m: rh,
                    pressurized: false,
                }
            }
        }
    }
}
