//! Original binary64 inputs for the frozen pump reference observer.

use super::{Case, Expected};

pub(crate) const CASES: [Case; 11] = [
    Case {
        id: "reference_knot",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x4010000000000000),
        speed: f64::from_bits(0x3ff0000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: true,
        expected: Expected::Running,
    },
    Case {
        id: "reference_zero_head",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x0000000000000000),
        speed: f64::from_bits(0x3ff0000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: true,
        expected: Expected::Running,
    },
    Case {
        id: "reference_shutoff",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x4020000000000000),
        speed: f64::from_bits(0x3ff0000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: true,
        expected: Expected::Shutoff,
    },
    Case {
        id: "half_speed",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x3ff0000000000000),
        speed: f64::from_bits(0x3fe0000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
    Case {
        id: "double_speed",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x4030000000000000),
        speed: f64::from_bits(0x4000000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
    Case {
        id: "half_speed_shutoff",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x4000000000000000),
        speed: f64::from_bits(0x3fe0000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Shutoff,
    },
    Case {
        id: "double_speed_shutoff",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x4040000000000000),
        speed: f64::from_bits(0x4000000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Shutoff,
    },
    Case {
        id: "negative_covered_head",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0xbff0000000000000),
        speed: f64::from_bits(0x3ff0000000000000),
        knots: &[
            (
                f64::from_bits(0xc000000000000000),
                f64::from_bits(0x4014000000000000),
            ),
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
    Case {
        id: "positive_plateau",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x3ff0000000000000),
        speed: f64::from_bits(0x3ff0000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4008000000000000),
            ),
            (
                f64::from_bits(0x4000000000000000),
                f64::from_bits(0x4008000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
    Case {
        id: "nonbinary_speed_07",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x3ff0000000000000),
        speed: f64::from_bits(0x3fe6666666666666),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
    Case {
        id: "nonbinary_speed_13",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x4010000000000000),
        speed: f64::from_bits(0x3ff4cccccccccccd),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
];
