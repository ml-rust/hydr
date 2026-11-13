use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_rectangle_right_dry_sonic",
        owner: 1,
        group: 7,
        inputs: [
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
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
        id: "scalar_rectangle_left_dry_sonic",
        owner: 1,
        group: 30,
        inputs: [
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
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
        id: "tiny_rectangle_right_dry_fan",
        owner: 1,
        group: 20,
        inputs: [
            [0x1a70000000000000, 0x0000000000000000, 0x0000000000000000],
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
        difference_allowance: Some(0x2b1b7cdfd9d7bdba),
    },
];
