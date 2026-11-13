use super::model::{Case, Reference, SideReference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "tiny_rectangle_compression",
        source_index: 64,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x2d40000000000000, 0x2d40000000000000],
                    native_tolerance: 0x2b2b7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x2d30000000000000, 0x2d30000000000000],
                    native_tolerance: 0x2b1b7cdfd9d7bdba,
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
                    bounds: [0x2d40000000000000, 0x2d40000000000000],
                    native_tolerance: 0x2b2b7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x2d30000000000000, 0x2d30000000000000],
                    native_tolerance: 0x2b1b7cdfd9d7bdba,
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
    Case {
        id: "pointed_minimum_uniform_moving",
        source_index: 65,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x2f40000000000000, 0x2f40000000000000],
                    native_tolerance: 0x2d2b7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x2f20000000000000, 0x2f20000000000000],
                    native_tolerance: 0x2d0b7cdfd9d7bdba,
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
                    bounds: [0x2f40000000000000, 0x2f40000000000000],
                    native_tolerance: 0x2d2b7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x2f20000000000000, 0x2f20000000000000],
                    native_tolerance: 0x2d0b7cdfd9d7bdba,
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
        id: "low_only_rectangle_uniform_moving",
        source_index: 66,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x20c0000000000000, 0x20c0000000000000],
                    native_tolerance: 0x1eab7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x20b0000000000000, 0x20b0000000000000],
                    native_tolerance: 0x1e9b7cdfd9d7bdba,
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
                    bounds: [0x20c0000000000000, 0x20c0000000000000],
                    native_tolerance: 0x1eab7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x20b0000000000000, 0x20b0000000000000],
                    native_tolerance: 0x1e9b7cdfd9d7bdba,
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
    Case {
        id: "tiny_rectangle_right_dry_fan",
        source_index: 67,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x2d40000000000000, 0x2d40000000000000],
                    native_tolerance: 0x2b2b7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x2d30000000000000, 0x2d30000000000000],
                    native_tolerance: 0x2b1b7cdfd9d7bdba,
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
                rectangular: true,
                pointed: false,
                within_first_band: true,
                film_exact: false,
            },
        ],
    },
];
