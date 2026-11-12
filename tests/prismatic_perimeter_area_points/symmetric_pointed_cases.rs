use super::Case;

pub const SYMMETRIC_POINTED_SAME_AREA_ONE: Case = Case {
    id: "symmetric_pointed_same_area_one",
    owner: 7,
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
    perimeter: [0x4006a09e667f3bcc, 0x4006a09e667f3bcd],
    radius: Some([0x3fd6a09e667f3bcc, 0x3fd6a09e667f3bcd]),
    perimeter_allowance: 0x3e160a007d8f5ee3,
    radius_allowance: 0x3e11c98a7ad3e79e,
    conditioned_radius: false,
    radius_positive_lower: true,
};
