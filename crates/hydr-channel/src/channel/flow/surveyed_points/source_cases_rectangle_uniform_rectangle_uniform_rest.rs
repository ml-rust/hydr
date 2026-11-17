use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_rectangle_uniform_rest",
        owner: 1,
        group: 8,
        inputs: [
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
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
    Case {
        id: "scalar_rectangle_uniform_forward",
        owner: 1,
        group: 27,
        inputs: [
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0x4008000000000000, 0x4008000000000000],
                center: [0x4008000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x4008000000000000, 0x4008000000000000],
                center: [0x4008000000000000, 0x0000000000000000],
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
        id: "scalar_rectangle_uniform_reverse",
        owner: 1,
        group: 9,
        inputs: [
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xc008000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xc008000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0xc008000000000000, 0xc008000000000000],
                center: [0xc008000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0xc008000000000000, 0xc008000000000000],
                center: [0xc008000000000000, 0x0000000000000000],
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
        id: "low_only_rectangle_uniform_moving",
        owner: 1,
        group: 38,
        inputs: [
            [0x0000000000000000, 0x0170000000000000, 0x0000000000000000],
            [0x0190000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0170000000000000, 0x0000000000000000],
            [0x0190000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0x4010000000000000, 0x4010000000000000],
                center: [0x4010000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x4010000000000000, 0x4010000000000000],
                center: [0x4010000000000000, 0x0000000000000000],
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
