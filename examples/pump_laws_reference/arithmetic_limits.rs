//! Original binary64 inputs for the frozen pump reference observer.

use super::{Case, Expected};

pub(crate) const CASES: [Case; 15] = [
    Case {
        id: "tiny_speed_exact_zero_head",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x0000000000000000),
        speed: f64::from_bits(0x0000000000000001),
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
        id: "huge_speed_exact_zero_head",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x0000000000000000),
        speed: f64::from_bits(0x7fefffffffffffff),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x0000000000000001),
            ),
            (
                f64::from_bits(0x3ff0000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
    Case {
        id: "unused_speed_square_underflow",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x0000000000000001),
        speed: f64::from_bits(0x1a70000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x47e0000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
    Case {
        id: "unused_speed_square_overflow",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x7fb0000000000000),
        speed: f64::from_bits(0x6570000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x1a70000000000000),
            ),
            (
                f64::from_bits(0x34c0000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
    Case {
        id: "positive_discharge_underflow",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x0000000000000000),
        speed: f64::from_bits(0x0000000000000001),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x0000000000000001),
            ),
            (
                f64::from_bits(0x3ff0000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Numerical,
    },
    Case {
        id: "positive_discharge_overflow",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x0000000000000000),
        speed: f64::from_bits(0x7fefffffffffffff),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x3ff0000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Numerical,
    },
    Case {
        id: "above_curve_domain",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x4022000000000000),
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
        reference_only: false,
        expected: Expected::UnsupportedRegime,
    },
    Case {
        id: "below_curve_domain",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0xbff0000000000000),
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
        reference_only: false,
        expected: Expected::UnsupportedRegime,
    },
    Case {
        id: "nonbinary_upper_domain_below",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x400f5c28f5c28f5a),
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
        expected: Expected::Numerical,
    },
    Case {
        id: "nonbinary_upper_domain_rounded",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x400f5c28f5c28f5b),
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
        expected: Expected::Numerical,
    },
    Case {
        id: "nonbinary_upper_domain_above",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x400f5c28f5c28f5c),
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
        expected: Expected::Numerical,
    },
    Case {
        id: "nonbinary_shutoff_ambiguity",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x3fff5c28f5c28f5b),
        speed: f64::from_bits(0x3fe6666666666666),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x0000000000000000),
            ),
            (
                f64::from_bits(0x4020000000000000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Numerical,
    },
    Case {
        id: "nonbinary_certified_zero_plateau",
        donor_stage: f64::from_bits(0x0000000000000000),
        receiving_stage: f64::from_bits(0x4008000000000000),
        speed: f64::from_bits(0x3fe6666666666666),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4010000000000000),
                f64::from_bits(0x0000000000000000),
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
        id: "original_head_low_inside_domain",
        donor_stage: f64::from_bits(0xbff0000000000000),
        receiving_stage: f64::from_bits(0x4341c37937e08000),
        speed: f64::from_bits(0x3ff0000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4341c37937e08000),
                f64::from_bits(0x4000000000000000),
            ),
            (
                f64::from_bits(0x4351c37937e08000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Running,
    },
    Case {
        id: "original_head_low_domain_ambiguity",
        donor_stage: f64::from_bits(0xbff0000000000000),
        receiving_stage: f64::from_bits(0x4341c37937e08000),
        speed: f64::from_bits(0x3ff0000000000000),
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x4010000000000000),
            ),
            (
                f64::from_bits(0x4341c37937e08000),
                f64::from_bits(0x0000000000000000),
            ),
        ],
        reference_only: false,
        expected: Expected::Numerical,
    },
];
