use super::{Case, Query, Update};
use hydr::structures::GateActuatorAccuracy;
const CASES: &[Case] = &[
    Case {
        family: "physical",
        id: "zero_closing_rate_stall",
        policy: "O",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4010000000000000),
        initial: f64::from_bits(0x3fe8000000000000),
        target: f64::from_bits(0x0000000000000000),
        opening_rate: f64::from_bits(0x3fe0000000000000),
        closing_rate: f64::from_bits(0x0000000000000000),
        maximum: f64::from_bits(0x3ff0000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        updates: &[
            Update {
                time: f64::from_bits(0x4000000000000000),
                target: f64::from_bits(0x3ff0000000000000),
                available: f64::from_bits(0x0000000000000000),
            },
            Update {
                time: f64::from_bits(0x4008000000000000),
                target: f64::from_bits(0x0000000000000000),
                available: f64::from_bits(0x0000000000000000),
            },
        ],
        queries: &[
            Query {
                time: f64::from_bits(0x0000000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3ff0000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4000000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4002000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4004000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4008000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4010000000000000),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "physical",
        id: "huge_origin_absorbed_increment",
        policy: "H",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4020000000000000),
        initial: f64::from_bits(0x4341c37937e08000),
        target: f64::from_bits(0x4341c37937e08001),
        opening_rate: f64::from_bits(0x3fd0000000000000),
        closing_rate: f64::from_bits(0x3fe0000000000000),
        maximum: f64::from_bits(0x4341c37937e08001),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x0000000000000000),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        updates: &[],
        queries: &[
            Query {
                time: f64::from_bits(0x0000000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3ff0000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4000000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4010000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x401c000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4020000000000000),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "physical",
        id: "subnormal_positive_opening",
        policy: "S",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4008000000000000),
        initial: f64::from_bits(0x0000000000000000),
        target: f64::from_bits(0x0000000000000002),
        opening_rate: f64::from_bits(0x0000000000000001),
        closing_rate: f64::from_bits(0x0000000000000001),
        maximum: f64::from_bits(0x0000000000000002),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x0000000000000002),
            maximum_relative_opening_error: f64::from_bits(0x0000000000000000),
            maximum_arrival_window_s: f64::from_bits(0x3d719799812dea11),
        },
        updates: &[],
        queries: &[
            Query {
                time: f64::from_bits(0x0000000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3fd0000000000000),
                expected: "Numerical",
            },
            Query {
                time: f64::from_bits(0x3fe0000000000000),
                expected: "Numerical",
            },
            Query {
                time: f64::from_bits(0x3fe8000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3ff0000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3ff8000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4000000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4008000000000000),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "physical",
        id: "subnormal_positive_closing",
        policy: "S",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4000000000000000),
        initial: f64::from_bits(0x0000000000000001),
        target: f64::from_bits(0x0000000000000000),
        opening_rate: f64::from_bits(0x0000000000000001),
        closing_rate: f64::from_bits(0x0000000000000001),
        maximum: f64::from_bits(0x0000000000000001),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x0000000000000002),
            maximum_relative_opening_error: f64::from_bits(0x0000000000000000),
            maximum_arrival_window_s: f64::from_bits(0x3d719799812dea11),
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
                time: f64::from_bits(0x3fe0000000000000),
                expected: "Numerical",
            },
            Query {
                time: f64::from_bits(0x3fe8000000000000),
                expected: "Numerical",
            },
            Query {
                time: f64::from_bits(0x3ff0000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4000000000000000),
                expected: "Success",
            },
        ],
    },
];

pub(super) fn append(output: &mut Vec<Case>) {
    output.extend(CASES.iter().copied());
}
