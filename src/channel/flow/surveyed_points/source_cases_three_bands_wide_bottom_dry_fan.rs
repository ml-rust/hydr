use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[Case {
    id: "scalar_planned_three_bands_both_knots_dry_fan",
    owner: 0,
    group: 46,
    inputs: [
        [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
    ],
    dry: [false, true],
    uniform: false,
    velocity: [
        Some(Reference {
            bounds: [0x0000000000000000, 0x0000000000000000],
            center: [0x0000000000000000, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        None,
    ],
    difference: None,
    difference_zero: false,
    difference_allowance: None,
}];
