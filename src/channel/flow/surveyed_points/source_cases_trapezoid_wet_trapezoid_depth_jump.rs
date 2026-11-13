use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_trapezoid_depth_jump",
        owner: 3,
        group: 51,
        inputs: [
            [0x4018000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3fe0000000000000, 0x0000000000000000, 0x0000000000000000],
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
        id: "scalar_trapezoid_weak_depth_jump",
        owner: 3,
        group: 25,
        inputs: [
            [0x4000000000000001, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4000000000000000, 0x0000000000000000, 0x0000000000000000],
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
        id: "scalar_trapezoid_stationary_shock",
        owner: 3,
        group: 42,
        inputs: [
            [0x4000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4037000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4018000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4037000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x4027000000000000, 0x4027000000000000],
                center: [0x4027000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x400eaaaaaaaaaaaa, 0x400eaaaaaaaaaaab],
                center: [0x400eaaaaaaaaaaab, 0xbca5555555555555],
                residual: 0x3945555555555556,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0xc01eaaaaaaaaaaab, 0xc01eaaaaaaaaaaaa],
            center: [0xc01eaaaaaaaaaaab, 0x3cb5555555555555],
            residual: 0x3955555555555556,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
];
