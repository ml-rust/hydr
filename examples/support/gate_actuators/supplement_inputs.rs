use super::{Case, Query};
use hydr::structures::GateActuatorAccuracy;
const CASES: &[Case] = &[
    Case {
        family: "supplement",
        id: "ordinary_closing_one_rate_point_zero_two",
        policy: "O",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4049000000000001),
        initial: f64::from_bits(0x3ff0000000000000),
        target: f64::from_bits(0x0000000000000000),
        opening_rate: f64::from_bits(0x0000000000000000),
        closing_rate: f64::from_bits(0x3f947ae147ae147b),
        maximum: f64::from_bits(0x3ff0000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        updates: &[],
        queries: &[
            Query {
                time: f64::from_bits(0x4048ffffffffffff),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4049000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4049000000000001),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "supplement",
        id: "ordinary_closing_point_one_five_rate_point_three",
        policy: "O",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x3fe0000000000001),
        initial: f64::from_bits(0x3fc3333333333333),
        target: f64::from_bits(0x0000000000000000),
        opening_rate: f64::from_bits(0x0000000000000000),
        closing_rate: f64::from_bits(0x3fd3333333333333),
        maximum: f64::from_bits(0x3fc3333333333333),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        updates: &[],
        queries: &[
            Query {
                time: f64::from_bits(0x3fdfffffffffffff),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3fe0000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3fe0000000000001),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "companion",
        id: "negative_elapsed_low_overflow_counterexample",
        policy: "X-wide",
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
            maximum_relative_opening_error: f64::from_bits(0x3eb0000000000000),
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
                expected: "Success",
            },
        ],
    },
    Case {
        family: "companion",
        id: "large_clock_subulp_travel",
        policy: "L",
        binding_expected: "Numerical",
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
            maximum_arrival_window_s: f64::from_bits(0x0000000000000000),
        },
        updates: &[],
        queries: &[
            Query {
                time: f64::from_bits(0x4340000000000000),
                expected: "Numerical",
            },
            Query {
                time: f64::from_bits(0x4340000000000001),
                expected: "Numerical",
            },
            Query {
                time: f64::from_bits(0x4340000000000002),
                expected: "Numerical",
            },
        ],
    },
];

pub(super) fn append(output: &mut Vec<Case>) {
    output.extend(CASES.iter().copied());
}
