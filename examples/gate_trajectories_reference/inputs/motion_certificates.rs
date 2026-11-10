//! Original-bit inputs and separately frozen runtime admission.
use super::types::Case;

pub const CASES: &[Case] = &[
    Case {
        family: "motion",
        id: "subnormal_back_product_exact",
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x0000000000000000),
            ),
            (
                f64::from_bits(0x4000000000000000),
                f64::from_bits(0x0000000000000002),
            ),
        ],
        opening_speed: f64::from_bits(0x0000000000000001),
        closing_speed: f64::from_bits(0x0000000000000000),
        maximum_opening: f64::from_bits(0x0000000000000002),
        absolute: f64::from_bits(0x7fefffffffffffff),
        relative: f64::from_bits(0x0000000000000000),
        binding: "success",
        static_invalid: false,
        queries: &[],
    },
    Case {
        family: "motion",
        id: "subnormal_back_product_ambiguous",
        knots: &[
            (
                f64::from_bits(0x0000000000000000),
                f64::from_bits(0x0000000000000000),
            ),
            (
                f64::from_bits(0x4000000000000000),
                f64::from_bits(0x0000000000000003),
            ),
        ],
        opening_speed: f64::from_bits(0x0000000000000002),
        closing_speed: f64::from_bits(0x0000000000000000),
        maximum_opening: f64::from_bits(0x0000000000000003),
        absolute: f64::from_bits(0x7fefffffffffffff),
        relative: f64::from_bits(0x0000000000000000),
        binding: "Numerical",
        static_invalid: false,
        queries: &[],
    },
];
