use super::model::{Case, Reference, SideReference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_three_bands_first_knot_neighbors",
        source_index: 28,
        gravity_bits: 0x3ff0000000000000,
        sides: [
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
                within_first_band: false,
                film_exact: false,
            },
        ],
    },
    Case {
        id: "scalar_three_bands_second_knot_neighbors",
        source_index: 29,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x400a884e949f44ca, 0x400a884e949f44cb],
                    native_tolerance: 0x3df6ca8c18691ceb,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3fefffffffffffff, 0x3ff0000000000000],
                    native_tolerance: 0x3ddb7cdfd9d7bdba,
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
                    bounds: [0x400a884e949f44cb, 0x400a884e949f44cc],
                    native_tolerance: 0x3df6ca8c18691ceb,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff0000000000000, 0x3ff0000000000001],
                    native_tolerance: 0x3ddb7cdfd9d7bdbb,
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
    Case {
        id: "scalar_three_bands_cap_rest",
        source_index: 30,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x4013617c9e1421af, 0x4013617c9e1421b0],
                    native_tolerance: 0x3e00a5e27b91b701,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff993e89c9eb01d, 0x3ff993e89c9eb01e],
                    native_tolerance: 0x3de5f89c566141c5,
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
                    bounds: [0x4013617c9e1421af, 0x4013617c9e1421b0],
                    native_tolerance: 0x3e00a5e27b91b701,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff993e89c9eb01d, 0x3ff993e89c9eb01e],
                    native_tolerance: 0x3de5f89c566141c5,
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
    Case {
        id: "scalar_three_bands_cap_compression",
        source_index: 31,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x4013617c9e1421af, 0x4013617c9e1421b0],
                    native_tolerance: 0x3e00a5e27b91b701,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff993e89c9eb01d, 0x3ff993e89c9eb01e],
                    native_tolerance: 0x3de5f89c566141c5,
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
                    bounds: [0x4013617c9e1421af, 0x4013617c9e1421b0],
                    native_tolerance: 0x3e00a5e27b91b701,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff993e89c9eb01d, 0x3ff993e89c9eb01e],
                    native_tolerance: 0x3de5f89c566141c5,
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
