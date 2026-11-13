use super::model::{Case, Reference, SideReference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "retained_vacuum_inside_film",
        source_index: 68,
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
                film_exact: true,
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
                film_exact: true,
            },
        ],
    },
    Case {
        id: "captured_tiny_upwind_face_64",
        source_index: 69,
        gravity_bits: 0x40239eb851eb851f,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x3cee58af3b073e4a, 0x3cee58af3b073e4b],
                    native_tolerance: 0x3ada113fc96df809,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3cde58af3b073e4a, 0x3cde58af3b073e4b],
                    native_tolerance: 0x3aca113fc96df809,
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
                    bounds: [0x3c91cabc26352ae5, 0x3c91cabc26352ae6],
                    native_tolerance: 0x3a7e90f99e7da7f6,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3c81cabc26352ae5, 0x3c81cabc26352ae6],
                    native_tolerance: 0x3a6e90f99e7da7f6,
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
