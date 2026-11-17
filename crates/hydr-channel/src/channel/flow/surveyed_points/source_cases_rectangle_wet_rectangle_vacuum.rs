use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_rectangle_vacuum",
        owner: 1,
        group: 12,
        inputs: [
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xc008000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0xc008000000000000, 0xc008000000000000],
                center: [0xc008000000000000, 0x0000000000000000],
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
            bounds: [0x4018000000000000, 0x4018000000000000],
            center: [0x4018000000000000, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
    Case {
        id: "scalar_rectangle_vacuum_threshold",
        owner: 1,
        group: 23,
        inputs: [
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xc000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0xc000000000000000, 0xc000000000000000],
                center: [0xc000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x4000000000000000, 0x4000000000000000],
                center: [0x4000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x4010000000000000, 0x4010000000000000],
            center: [0x4010000000000000, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
    Case {
        id: "scalar_rectangle_vacuum_threshold_outside",
        owner: 1,
        group: 50,
        inputs: [
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xc000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4000000000000001, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0xc000000000000000, 0xc000000000000000],
                center: [0xc000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x4000000000000001, 0x4000000000000001],
                center: [0x4000000000000001, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x4010000000000000, 0x4010000000000001],
            center: [0x4010000000000000, 0x3cc0000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
    Case {
        id: "scalar_rectangle_vacuum_threshold_inside",
        owner: 1,
        group: 24,
        inputs: [
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xc000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3fffffffffffffff, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0xc000000000000000, 0xc000000000000000],
                center: [0xc000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x3fffffffffffffff, 0x3fffffffffffffff],
                center: [0x3fffffffffffffff, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x400fffffffffffff, 0x4010000000000000],
            center: [0x4010000000000000, 0xbcb0000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
];
