use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_three_bands_both_knots_dry_fan",
        owner: 2,
        group: 34,
        inputs: [
            [0x4028000000000000, 0x0000000000000000, 0x0000000000000000],
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
    },
    Case {
        id: "scalar_three_bands_both_knots_dry_fan_reverse",
        owner: 2,
        group: 44,
        inputs: [
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x4028000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [true, false],
        uniform: false,
        velocity: [
            None,
            Some(Reference {
                bounds: [0x0000000000000000, 0x0000000000000000],
                center: [0x0000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: None,
        difference_zero: false,
        difference_allowance: None,
    },
    Case {
        id: "scalar_three_bands_both_dry",
        owner: 2,
        group: 56,
        inputs: [
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [true, true],
        uniform: false,
        velocity: [None, None],
        difference: None,
        difference_zero: false,
        difference_allowance: None,
    },
];
