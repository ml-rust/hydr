use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_three_bands_uniform_upper",
        owner: 2,
        group: 54,
        inputs: [
            [0x4028000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4028000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0x3fd0000000000000, 0x3fd0000000000000],
                center: [0x3fd0000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x3fd0000000000000, 0x3fd0000000000000],
                center: [0x3fd0000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x0000000000000000, 0x0000000000000000],
            center: [0x0000000000000000, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: true,
        difference_allowance: None,
    },
    Case {
        id: "scalar_three_bands_cap_rest",
        owner: 2,
        group: 53,
        inputs: [
            [0x4037000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4037000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0x0000000000000000, 0x0000000000000000],
                center: [0x0000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x0000000000000000, 0x0000000000000000],
                center: [0x0000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x0000000000000000, 0x0000000000000000],
            center: [0x0000000000000000, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: true,
        difference_allowance: None,
    },
];
