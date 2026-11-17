use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[Case {
    id: "scalar_trapezoid_uniform_flow",
    owner: 3,
    group: 59,
    inputs: [
        [0x4018000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x4008000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x4018000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x4008000000000000, 0x0000000000000000, 0x0000000000000000],
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
}];
