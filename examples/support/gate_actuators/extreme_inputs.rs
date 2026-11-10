use super::{Case, Query};
use hydr::structures::GateActuatorAccuracy;
const CASES: &[Case] = &[
    Case {
        family: "physical",
        id: "large_clock_subulp_travel",
        policy: "L",
        binding_expected: "Success",
        start: f64::from_bits(0x4340000000000000),
        end: f64::from_bits(0x4340000000000002),
        initial: f64::from_bits(0x0000000000000000),
        target: f64::from_bits(0x3ff0000000000000),
        opening_rate: f64::from_bits(0x4010000000000000),
        closing_rate: f64::from_bits(0x3ff0000000000000),
        maximum: f64::from_bits(0x3ff0000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x4010000000000000),
        },
        updates: &[],
        queries: &[
            Query {
                time: f64::from_bits(0x4340000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4340000000000001),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4340000000000002),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "physical",
        id: "saturated_huge_product",
        policy: "U",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x7fefffffffffffff),
        initial: f64::from_bits(0x0000000000000000),
        target: f64::from_bits(0x3ff0000000000000),
        opening_rate: f64::from_bits(0x7fefffffffffffff),
        closing_rate: f64::from_bits(0x7fefffffffffffff),
        maximum: f64::from_bits(0x3ff0000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x0000000000000010),
        },
        updates: &[],
        queries: &[
            Query {
                time: f64::from_bits(0x0000000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x0000000000000001),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3ff0000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x7fefffffffffffff),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "physical",
        id: "slow_arrival_beyond_horizon",
        policy: "B",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x3ff0000000000000),
        initial: f64::from_bits(0x3fd0000000000000),
        target: f64::from_bits(0x7fefffffffffffff),
        opening_rate: f64::from_bits(0x0000000000000001),
        closing_rate: f64::from_bits(0x0000000000000000),
        maximum: f64::from_bits(0x7fefffffffffffff),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x0000000000000000),
        },
        updates: &[],
        queries: &[
            Query {
                time: f64::from_bits(0x0000000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3fd0000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3ff0000000000000),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "physical",
        id: "negative_elapsed_low_overflow_counterexample",
        policy: "X",
        binding_expected: "Success",
        start: f64::from_bits(0x7c90000000000000),
        end: f64::from_bits(0x7feffffffc000000),
        initial: f64::from_bits(0x7fefffffffffffff),
        target: f64::from_bits(0x0000000000000000),
        opening_rate: f64::from_bits(0x0000000000000000),
        closing_rate: f64::from_bits(0x3ff0000002000000),
        maximum: f64::from_bits(0x7fefffffffffffff),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x0000000000000000),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x7d30000000000000),
        },
        updates: &[],
        queries: &[
            Query {
                time: f64::from_bits(0x7c90000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x7feffffffc000000),
                expected: "Numerical",
            },
        ],
    },
];

pub(super) fn append(output: &mut Vec<Case>) {
    output.extend(CASES.iter().copied());
}
