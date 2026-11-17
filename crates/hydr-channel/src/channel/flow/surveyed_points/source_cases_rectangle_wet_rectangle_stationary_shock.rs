use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_rectangle_stationary_shock",
        owner: 1,
        group: 28,
        inputs: [
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x4008000000000000, 0x4008000000000000],
                center: [0x4008000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x3ff8000000000000, 0x3ff8000000000000],
                center: [0x3ff8000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: Some(Reference {
            bounds: [0xbff8000000000000, 0xbff8000000000000],
            center: [0xbff8000000000000, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
    Case {
        id: "scalar_rectangle_stationary_shock_reverse",
        owner: 1,
        group: 4,
        inputs: [
            [0x4000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xc008000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xc008000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0xbff8000000000000, 0xbff8000000000000],
                center: [0xbff8000000000000, 0x0000000000000000],
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
            bounds: [0xbff8000000000000, 0xbff8000000000000],
            center: [0xbff8000000000000, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
    Case {
        id: "scalar_rectangle_exact_wet_pure_fan",
        owner: 1,
        group: 48,
        inputs: [
            [0x4010000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x0000000000000000, 0x0000000000000000],
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
                bounds: [0x4000000000000000, 0x4000000000000000],
                center: [0x4000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x4000000000000000, 0x4000000000000000],
            center: [0x4000000000000000, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
    Case {
        id: "tiny_rectangle_compression",
        owner: 1,
        group: 0,
        inputs: [
            [0x1a70000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x1a94000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x1a70000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x1a90000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x4014000000000000, 0x4014000000000000],
                center: [0x4014000000000000, 0x0000000000000000],
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
            bounds: [0xbff0000000000000, 0xbff0000000000000],
            center: [0xbff0000000000000, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
];
