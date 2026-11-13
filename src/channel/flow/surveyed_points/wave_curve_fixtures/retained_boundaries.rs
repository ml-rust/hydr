use super::model::{Case, Expected, baseline, local};

pub(super) const CASES: &[Case] = &[
    baseline(
        "source_first_knot_forward__physical_anchor__physical_lower",
        0,
        0,
        Expected::Equality {
            limit: local(0x3febb67ae8584caa, 0x3febb67ae8584cab, 0x3dd7ce1cac9fa1d1),
        },
    ),
    baseline(
        "source_first_knot_forward__physical_anchor__physical_upper",
        0,
        1,
        Expected::Shock {
            curve: local(0x3ae279a74590331c, 0x3ae279a74590331d, 0x38cfbd7b90d4d7c1),
            offset: local(0x3febb67ae8584caa, 0x3febb67ae8584cab, 0x3dd7ce1cac9fa1d1),
        },
    ),
    baseline(
        "source_second_knot_reverse__physical_anchor__physical_lower",
        1,
        1,
        Expected::Rare {
            curve: local(0xbad4e6fdf33cf032, 0xbad4e6fdf33cf031, 0x38c1f477954f03df),
        },
    ),
    baseline(
        "source_second_knot_reverse__physical_anchor__physical_upper",
        1,
        0,
        Expected::Equality {
            limit: local(0x3ff3988e1409212e, 0x3ff3988e1409212f, 0x3de0d5301bfa13a1),
        },
    ),
    baseline(
        "three_band_cap_deficit__physical_anchor__physical_lower",
        2,
        1,
        Expected::Rare {
            curve: local(0xbac83091e6a7f7e7, 0xbac83091e6a7f7e6, 0x38b4c7608720b692),
        },
    ),
    baseline(
        "three_band_cap_deficit__physical_anchor__physical_upper",
        2,
        0,
        Expected::Rare {
            curve: local(0xbab83091e6a7f7e7, 0xbab83091e6a7f7e6, 0x38a4c7608720b692),
        },
    ),
    baseline(
        "rectangle_cap_deficit__physical_anchor__physical_lower",
        3,
        1,
        Expected::Rare {
            curve: local(0xbae279a74590331d, 0xbae279a74590331c, 0x38cfbd7b90d4d7c1),
        },
    ),
    baseline(
        "rectangle_cap_deficit__physical_anchor__physical_upper",
        3,
        0,
        Expected::Rare {
            curve: local(0xbad279a74590331d, 0xbad279a74590331c, 0x38bfbd7b90d4d7c1),
        },
    ),
    baseline(
        "first_band_start_signed_offsets__physical_anchor__physical_lower",
        4,
        0,
        Expected::Rare {
            curve: local(0xbad279a74590331d, 0xbad279a74590331c, 0x38bfbd7b90d4d7c1),
        },
    ),
    baseline(
        "first_band_start_signed_offsets__physical_anchor__physical_upper",
        4,
        1,
        Expected::Shock {
            curve: local(0x3ad279a74590331c, 0x3ad279a74590331d, 0x38bfbd7b90d4d7c1),
            offset: local(0x3febb67ae8584caa, 0x3febb67ae8584cab, 0x3dd7ce1cac9fa1d1),
        },
    ),
    baseline(
        "middle_band_end_signed_offsets__physical_anchor__physical_lower",
        5,
        0,
        Expected::Rare {
            curve: local(0xbac4e6fdf33cf032, 0xbac4e6fdf33cf031, 0x38b1f477954f03df),
        },
    ),
    baseline(
        "middle_band_end_signed_offsets__physical_anchor__physical_upper",
        5,
        1,
        Expected::Shock {
            curve: local(0x3ac4e6fdf33cf031, 0x3ac4e6fdf33cf032, 0x38b1f477954f03df),
            offset: local(0x3ff3988e1409212e, 0x3ff3988e1409212f, 0x3de0d5301bfa13a1),
        },
    ),
    baseline(
        "decimal_knot_signed_offsets__physical_anchor__physical_lower",
        6,
        0,
        Expected::Rare {
            curve: local(0xbae7d9f4cf754636, 0xbae7d9f4cf754635, 0x38d47cf9fcefb154),
        },
    ),
    baseline(
        "decimal_knot_signed_offsets__physical_anchor__physical_upper",
        6,
        1,
        Expected::Shock {
            curve: local(0x3ae7d9f4cf754635, 0x3ae7d9f4cf754636, 0x38d47cf9fcefb154),
            offset: local(0x3fe5775c544ff262, 0x3fe5775c544ff263, 0x3dd2707a96d7b932),
        },
    ),
    baseline(
        "trapezoid_source_retained_offsets__physical_anchor__physical_lower",
        16,
        0,
        Expected::Equality {
            limit: local(0x3fe1feb33c1c381e, 0x3fe1feb33c1c381f, 0x3dceea40256491f9),
        },
    ),
    baseline(
        "trapezoid_source_retained_offsets__physical_anchor__physical_upper",
        16,
        1,
        Expected::Shock {
            curve: local(0x3af1feb33c1c381e, 0x3af1feb33c1c381f, 0x38deea40256491f9),
            offset: local(0x3fe1feb33c1c381e, 0x3fe1feb33c1c381f, 0x3dceea40256491f9),
        },
    ),
];
