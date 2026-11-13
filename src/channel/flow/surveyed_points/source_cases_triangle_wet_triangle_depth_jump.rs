use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[Case {
    id: "scalar_triangle_depth_jump",
    owner: 4,
    group: 33,
    inputs: [
        [0x4010000000000000, 0x0000000000000000, 0x0000000000000000],
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
}];
