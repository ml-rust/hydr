use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "uniform_cap_represented_tenth_forward",
        owner: 0,
        group: 3,
        inputs: [
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff6666666666667, 0xbc90000000000000, 0x0000000000000000],
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff6666666666667, 0xbc90000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0x3fb999999999999a, 0x3fb999999999999a],
                center: [0x3fb999999999999a, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x3fb999999999999a, 0x3fb999999999999a],
                center: [0x3fb999999999999a, 0x0000000000000000],
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
        id: "uniform_cap_represented_tenth_reverse",
        owner: 0,
        group: 1,
        inputs: [
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xbff6666666666667, 0x3c90000000000000, 0x0000000000000000],
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0xbff6666666666667, 0x3c90000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0xbfb999999999999a, 0xbfb999999999999a],
                center: [0xbfb999999999999a, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0xbfb999999999999a, 0xbfb999999999999a],
                center: [0xbfb999999999999a, 0x0000000000000000],
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
        id: "uniform_cap_retained_deficit_dyadic_velocity",
        owner: 0,
        group: 15,
        inputs: [
            [0x402c000000000000, 0xbaf0000000000000, 0x0000000000000000],
            [0x3ffc000000000000, 0xbac0000000000000, 0x0000000000000000],
            [0x402c000000000000, 0xbaf0000000000000, 0x0000000000000000],
            [0x3ffc000000000000, 0xbac0000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: true,
        velocity: [
            Some(Reference {
                bounds: [0x3fc0000000000000, 0x3fc0000000000000],
                center: [0x3fc0000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x3fc0000000000000, 0x3fc0000000000000],
                center: [0x3fc0000000000000, 0x0000000000000000],
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
