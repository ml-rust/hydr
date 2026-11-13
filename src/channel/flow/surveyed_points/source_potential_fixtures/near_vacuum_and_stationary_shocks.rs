use super::model::{Case, Reference, SideReference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_rectangle_vacuum_threshold_inside",
        source_index: 24,
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
        id: "scalar_rectangle_stationary_shock",
        source_index: 25,
        gravity_bits: 0x4008000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x400bb67ae8584caa, 0x400bb67ae8584cab],
                    native_tolerance: 0x3df7ce1cac9fa1d1,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ffbb67ae8584caa, 0x3ffbb67ae8584cab],
                    native_tolerance: 0x3de7ce1cac9fa1d1,
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
                    bounds: [0x4013988e1409212e, 0x4013988e1409212f],
                    native_tolerance: 0x3e00d5301bfa13a1,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x4003988e1409212e, 0x4003988e1409212f],
                    native_tolerance: 0x3df0d5301bfa13a1,
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
        id: "scalar_rectangle_stationary_shock_reverse",
        source_index: 26,
        gravity_bits: 0x4008000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x4013988e1409212e, 0x4013988e1409212f],
                    native_tolerance: 0x3e00d5301bfa13a1,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x4003988e1409212e, 0x4003988e1409212f],
                    native_tolerance: 0x3df0d5301bfa13a1,
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
                    bounds: [0x400bb67ae8584caa, 0x400bb67ae8584cab],
                    native_tolerance: 0x3df7ce1cac9fa1d1,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ffbb67ae8584caa, 0x3ffbb67ae8584cab],
                    native_tolerance: 0x3de7ce1cac9fa1d1,
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
        id: "scalar_trapezoid_stationary_shock",
        source_index: 27,
        gravity_bits: 0x4047000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x402dd5720f7430d0, 0x402dd5720f7430d1],
                    native_tolerance: 0x3e19a08409915790,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x401626a7bcbe6ab6, 0x401626a7bcbe6ab7],
                    native_tolerance: 0x3e03070e45686fbb,
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
                    bounds: [0x4036001dae08f55e, 0x4036001dae08f55f],
                    native_tolerance: 0x3e22e5f3646de309,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x401db7fb7a43d09d, 0x401db7fb7a43d09e],
                    native_tolerance: 0x3e09873502137880,
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
];
