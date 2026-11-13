use super::model::{Case, Reference, SideReference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_trapezoid_uniform_flow",
        source_index: 4,
        gravity_bits: 0x40239eb851eb851f,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x402451d7975d5986, 0x402451d7975d5987],
                    native_tolerance: 0x3e1174592fcbb06c,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x400b72c2637ae565, 0x400b72c2637ae566],
                    native_tolerance: 0x3df793f0b7c71701,
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
                    bounds: [0x402451d7975d5986, 0x402451d7975d5987],
                    native_tolerance: 0x3e1174592fcbb06c,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x400b72c2637ae565, 0x400b72c2637ae566],
                    native_tolerance: 0x3df793f0b7c71701,
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
        id: "scalar_three_bands_uniform_upper",
        source_index: 5,
        gravity_bits: 0x40239eb851eb851f,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x40289031f248697f, 0x40289031f2486980],
                    native_tolerance: 0x3e151984ba0b6f2a,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x400f518eafa820dd, 0x400f518eafa820de],
                    native_tolerance: 0x3dfae70782ad1494,
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
                    bounds: [0x40289031f248697f, 0x40289031f2486980],
                    native_tolerance: 0x3e151984ba0b6f2a,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x400f518eafa820dd, 0x400f518eafa820de],
                    native_tolerance: 0x3dfae70782ad1494,
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
        id: "scalar_rectangle_depth_jump",
        source_index: 6,
        gravity_bits: 0x3ff0000000000000,
        sides: [
            SideReference {
                phi: Reference {
                    bounds: [0x4006a09e667f3bcc, 0x4006a09e667f3bcd],
                    native_tolerance: 0x3df36fd255a2213a,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3ff6a09e667f3bcc, 0x3ff6a09e667f3bcd],
                    native_tolerance: 0x3de36fd255a2213a,
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
                    bounds: [0x3ff6a09e667f3bcc, 0x3ff6a09e667f3bcd],
                    native_tolerance: 0x3de36fd255a2213a,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3fe6a09e667f3bcc, 0x3fe6a09e667f3bcd],
                    native_tolerance: 0x3dd36fd255a2213a,
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
        id: "scalar_triangle_depth_jump",
        source_index: 7,
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
                    bounds: [0x4000000000000000, 0x4000000000000000],
                    native_tolerance: 0x3deb7cdfd9d7bdba,
                    exact_zero: false,
                },
                sound: Reference {
                    bounds: [0x3fe0000000000000, 0x3fe0000000000000],
                    native_tolerance: 0x3dcb7cdfd9d7bdba,
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
];
