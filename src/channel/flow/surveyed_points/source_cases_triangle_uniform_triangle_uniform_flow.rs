use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_triangle_uniform_flow",
        owner: 4,
        group: 52,
        inputs: [
            [0x4010000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4010000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0x3fe0000000000000, 0x3fe0000000000000],
                center: [0x3fe0000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x3fe0000000000000, 0x3fe0000000000000],
                center: [0x3fe0000000000000, 0x0000000000000000],
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
        id: "pointed_minimum_uniform_moving",
        owner: 4,
        group: 31,
        inputs: [
            [0x0000000000000001, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000005, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000001, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000005, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0x4014000000000000, 0x4014000000000000],
                center: [0x4014000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x4014000000000000, 0x4014000000000000],
                center: [0x4014000000000000, 0x0000000000000000],
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
