use super::Case;

pub const ASYMMETRIC_POINTED_SAME_AREA_SAME_AREA_ONE: Case = Case {
    id: "asymmetric_pointed_same_area_same_area_one",
    owner: 0,
    input_high: 0x3ff0000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3ff0000000000000, 0x3ff0000000000000]),
        Some([0x3ff0000000000000, 0x3ff0000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x3fd5555555555555, 0x3fd5555555555556]),
        Some([0x3fe0000000000000, 0x3fe0000000000000]),
    ],
    true_band_indices: &[0],
    perimeter: [0x4009e3779b97f4a7, 0x4009e3779b97f4a8],
    radius: Some([0x3fd3c6ef372fe94f, 0x3fd3c6ef372fe950]),
    perimeter_allowance: 0x3e16bd48c53143b0,
    radius_allowance: 0x3e11b5f4209ab3ed,
    conditioned_radius: false,
    radius_positive_lower: true,
};
