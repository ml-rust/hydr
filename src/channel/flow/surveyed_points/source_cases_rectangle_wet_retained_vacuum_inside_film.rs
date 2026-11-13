use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[Case {
    id: "retained_vacuum_inside_film",
    owner: 1,
    group: 18,
    inputs: [
        [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
        [0xc000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x4000000000000000, 0xbaf0000000000000, 0x0000000000000000],
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
            bounds: [0x3fffffffffffffff, 0x4000000000000000],
            center: [0x4000000000000000, 0xbaf0000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
    ],
    difference: Some(Reference {
        bounds: [0x400fffffffffffff, 0x4010000000000000],
        center: [0x4010000000000000, 0xbaf0000000000000],
        residual: 0x0000000000000000,
        non_dyadic: false,
    }),
    difference_zero: false,
    difference_allowance: Some(0x38db7cdfd9d7bdba),
}];
