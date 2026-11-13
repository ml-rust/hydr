use super::model::{Case, Reference, SideReference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_triangle_right_dry_sonic",
        source_index: 16,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x4010000000000000, 0x4010000000000000],
                    native_tolerance: 0x3dfb7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff0000000000000, 0x3ff0000000000000],
                    native_tolerance: 0x3ddb7cdfd9d7bdba,
                    exact_zero: false,
                },
                dry: false,
                physical_eligible: true,
                physical_wet_eligible: true,
                rectangular: false,
                pointed: true,
                within_first_band: true,
                film_exact: false,
            },
            SideReference {
                phi: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_tolerance: 0x0000000000000000,
                    exact_zero: true,
                },
                sound: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_tolerance: 0x0000000000000000,
                    exact_zero: true,
                },
                dry: true,
                physical_eligible: true,
                physical_wet_eligible: false,
                rectangular: false,
                pointed: true,
                within_first_band: true,
                film_exact: false,
            },
        ],
    },
    Case {
        id: "scalar_triangle_left_dry_sonic",
        source_index: 17,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_tolerance: 0x0000000000000000,
                    exact_zero: true,
                },
                sound: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_tolerance: 0x0000000000000000,
                    exact_zero: true,
                },
                dry: true,
                physical_eligible: true,
                physical_wet_eligible: false,
                rectangular: false,
                pointed: true,
                within_first_band: true,
                film_exact: false,
            },
            SideReference {
                phi: Reference {
                    bounds: [0x4010000000000000, 0x4010000000000000],
                    native_tolerance: 0x3dfb7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff0000000000000, 0x3ff0000000000000],
                    native_tolerance: 0x3ddb7cdfd9d7bdba,
                    exact_zero: false,
                },
                dry: false,
                physical_eligible: true,
                physical_wet_eligible: true,
                rectangular: false,
                pointed: true,
                within_first_band: true,
                film_exact: false,
            },
        ],
    },
    Case {
        id: "scalar_three_bands_both_knots_dry_fan",
        source_index: 18,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x400f5eaa2c86b6d8, 0x400f5eaa2c86b6d9],
                    native_tolerance: 0x3dfaf249da3d8a99,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff3ff9e3d8e5992, 0x3ff3ff9e3d8e5993],
                    native_tolerance: 0x3de12db7ee999d3a,
                    exact_zero: false,
                },
                dry: false,
                physical_eligible: false,
                physical_wet_eligible: false,
                rectangular: false,
                pointed: false,
                within_first_band: false,
                film_exact: false,
            },
            SideReference {
                phi: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_tolerance: 0x0000000000000000,
                    exact_zero: true,
                },
                sound: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_tolerance: 0x0000000000000000,
                    exact_zero: true,
                },
                dry: true,
                physical_eligible: false,
                physical_wet_eligible: false,
                rectangular: false,
                pointed: false,
                within_first_band: true,
                film_exact: false,
            },
        ],
    },
    Case {
        id: "scalar_three_bands_both_knots_dry_fan_reverse",
        source_index: 19,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_tolerance: 0x0000000000000000,
                    exact_zero: true,
                },
                sound: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_tolerance: 0x0000000000000000,
                    exact_zero: true,
                },
                dry: true,
                physical_eligible: false,
                physical_wet_eligible: false,
                rectangular: false,
                pointed: false,
                within_first_band: true,
                film_exact: false,
            },
            SideReference {
                phi: Reference {
                    bounds: [0x400f5eaa2c86b6d8, 0x400f5eaa2c86b6d9],
                    native_tolerance: 0x3dfaf249da3d8a99,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff3ff9e3d8e5992, 0x3ff3ff9e3d8e5993],
                    native_tolerance: 0x3de12db7ee999d3a,
                    exact_zero: false,
                },
                dry: false,
                physical_eligible: false,
                physical_wet_eligible: false,
                rectangular: false,
                pointed: false,
                within_first_band: false,
                film_exact: false,
            },
        ],
    },
];
