use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_three_bands_loose_recovery_allowance",
        owner: 2,
        group: 39,
        inputs: [
            [0x4028000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3fd0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
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
        id: "scalar_three_bands_zero_work_budget",
        owner: 2,
        group: 39,
        inputs: [
            [0x4028000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3fd0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
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
        id: "retained_knot_2_crossing",
        owner: 2,
        group: 16,
        inputs: [
            [0x4000000000000000, 0xbaf0000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x3af0000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
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
        id: "retained_knot_7_crossing",
        owner: 2,
        group: 49,
        inputs: [
            [0x401c000000000000, 0xbaf0000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x401c000000000000, 0x3af0000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
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
