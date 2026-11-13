use super::model::{Case, Reference, SideReference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_rectangle_weak_depth_jump",
        source_index: 12,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x4000000000000000, 0x4000000000000001],
                    native_tolerance: 0x3deb7cdfd9d7bdbb,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff0000000000000, 0x3ff0000000000001],
                    native_tolerance: 0x3ddb7cdfd9d7bdbb,
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
    Case {
        id: "scalar_trapezoid_weak_depth_jump",
        source_index: 13,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x40019851d45d52c1, 0x40019851d45d52c2],
                    native_tolerance: 0x3dee3a5cbb0ee3fc,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3fea20bd700c2c3e, 0x3fea20bd700c2c3f],
                    native_tolerance: 0x3dd671957aa2c4d7,
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
            SideReference {
                phi: Reference {
                    bounds: [0x40019851d45d52c0, 0x40019851d45d52c1],
                    native_tolerance: 0x3dee3a5cbb0ee3fb,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e],
                    native_tolerance: 0x3dd671957aa2c4d6,
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
        id: "scalar_rectangle_right_dry_sonic",
        source_index: 14,
        gravity_bits: 0x3ff0000000000000,
        sides: [
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
    Case {
        id: "scalar_rectangle_left_dry_sonic",
        source_index: 15,
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
