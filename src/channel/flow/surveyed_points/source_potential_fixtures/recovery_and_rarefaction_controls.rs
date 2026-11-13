use super::model::{Case, Reference, SideReference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_three_bands_loose_recovery_allowance",
        source_index: 32,
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
                    bounds: [0x3fedffea81796292, 0x3fedffea81796293],
                    native_tolerance: 0x3dd9c4ff6598075b,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3fdae89f995ad3ad, 0x3fdae89f995ad3ae],
                    native_tolerance: 0x3dc71d48501dfc02,
                    exact_zero: false,
                },
                dry: false,
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
        id: "scalar_three_bands_zero_work_budget",
        source_index: 33,
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
                    bounds: [0x3fedffea81796292, 0x3fedffea81796293],
                    native_tolerance: 0x3dd9c4ff6598075b,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3fdae89f995ad3ad, 0x3fdae89f995ad3ae],
                    native_tolerance: 0x3dc71d48501dfc02,
                    exact_zero: false,
                },
                dry: false,
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
        id: "scalar_planned_three_bands_both_knots_dry_fan",
        source_index: 34,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x400eed016f605088, 0x400eed016f605089],
                    native_tolerance: 0x3dfa90a7f1b28acb,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff52a7fa9d2f8e9, 0x3ff52a7fa9d2f8ea],
                    native_tolerance: 0x3de22e74763c9fbf,
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
        id: "scalar_rectangle_exact_wet_pure_fan",
        source_index: 35,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x4010000000000000, 0x4010000000000000],
                    native_tolerance: 0x3dfb7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x4000000000000000, 0x4000000000000000],
                    native_tolerance: 0x3deb7cdfd9d7bdba,
                    exact_zero: false,
                },
                dry: false,
                physical_eligible: true,
                physical_wet_eligible: true,
                rectangular: true,
                pointed: false,
                within_first_band: true,
                film_exact: false,
            },
            SideReference {
                phi: Reference {
                    bounds: [0x4000000000000000, 0x4000000000000000],
                    native_tolerance: 0x3deb7cdfd9d7bdba,
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
                rectangular: true,
                pointed: false,
                within_first_band: true,
                film_exact: false,
            },
        ],
    },
];
