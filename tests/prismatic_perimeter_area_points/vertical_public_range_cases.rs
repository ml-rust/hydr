use super::Case;

pub const VERTICAL_PUBLIC_RANGE_TINY_AREA: Case = Case {
    id: "vertical_public_range_tiny_area",
    owner: 12,
    input_high: 0x07c0000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x07b0000000000000, 0x07b0000000000000]),
        Some([0x07c0000000000000, 0x07c0000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x07b0000000000000, 0x07b0000000000000]),
    ],
    true_band_indices: &[0],
    perimeter: [0x4000000000000000, 0x4000000000000001],
    radius: Some([0x07afffffffffffff, 0x07b0000000000000]),
    perimeter_allowance: 0x3e149da7e361ce4c,
    radius_allowance: 0x059b7cdfd9d7bdb9,
    conditioned_radius: true,
    radius_positive_lower: true,
};
