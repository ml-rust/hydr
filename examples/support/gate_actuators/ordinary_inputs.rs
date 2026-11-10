use super::{Case, Query, Update};
use hydr::structures::GateActuatorAccuracy;
const CASES: &[Case] = &[
    Case {
        family: "physical",
        id: "dyadic_open_close_hold",
        policy: "O",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4020000000000000),
        initial: f64::from_bits(0x0000000000000000),
        target: f64::from_bits(0x3ff0000000000000),
        opening_rate: f64::from_bits(0x3fd0000000000000),
        closing_rate: f64::from_bits(0x3fe0000000000000),
        maximum: f64::from_bits(0x3ff0000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        updates: &[Update {
            time: f64::from_bits(0x4014000000000000),
            target: f64::from_bits(0x0000000000000000),
            available: f64::from_bits(0x0000000000000000),
        }],
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
                time: f64::from_bits(0x4010000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4014000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4018000000000000),
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
        id: "nonbinary_rates_and_times",
        policy: "O",
        binding_expected: "Success",
        start: f64::from_bits(0x3fb999999999999a),
        end: f64::from_bits(0x400599999999999a),
        initial: f64::from_bits(0x3fc999999999999a),
        target: f64::from_bits(0x3feccccccccccccd),
        opening_rate: f64::from_bits(0x3fd3333333333333),
        closing_rate: f64::from_bits(0x3fe6666666666666),
        maximum: f64::from_bits(0x3ff0000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        updates: &[
            Update {
                time: f64::from_bits(0x3fe999999999999a),
                target: f64::from_bits(0x3fb999999999999a),
                available: f64::from_bits(0xbfe0000000000000),
            },
            Update {
                time: f64::from_bits(0x3ffb333333333333),
                target: f64::from_bits(0x3fe3333333333333),
                available: f64::from_bits(0x3ff0000000000000),
            },
        ],
        queries: &[
            Query {
                time: f64::from_bits(0x3fb999999999999a),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3fd999999999999a),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3fe999999999999a),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3ff199999999999a),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3ffb333333333333),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4000cccccccccccd),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x400599999999999a),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "physical",
        id: "asymmetric_reversal",
        policy: "O",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4014000000000000),
        initial: f64::from_bits(0x3fd0000000000000),
        target: f64::from_bits(0x3ff0000000000000),
        opening_rate: f64::from_bits(0x3fe0000000000000),
        closing_rate: f64::from_bits(0x3fc0000000000000),
        maximum: f64::from_bits(0x3ff0000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        updates: &[
            Update {
                time: f64::from_bits(0x3ff0000000000000),
                target: f64::from_bits(0x0000000000000000),
                available: f64::from_bits(0x0000000000000000),
            },
            Update {
                time: f64::from_bits(0x4008000000000000),
                target: f64::from_bits(0x3fe0000000000000),
                available: f64::from_bits(0x0000000000000000),
            },
        ],
        queries: &[
            Query {
                time: f64::from_bits(0x0000000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3fe0000000000000),
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
                time: f64::from_bits(0x4008000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4010000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4014000000000000),
                expected: "Success",
            },
        ],
    },
    Case {
        family: "physical",
        id: "repeated_target",
        policy: "O",
        binding_expected: "Success",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4014000000000000),
        initial: f64::from_bits(0x0000000000000000),
        target: f64::from_bits(0x3ff0000000000000),
        opening_rate: f64::from_bits(0x3fd0000000000000),
        closing_rate: f64::from_bits(0x3fe0000000000000),
        maximum: f64::from_bits(0x3ff0000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        updates: &[
            Update {
                time: f64::from_bits(0x3ff0000000000000),
                target: f64::from_bits(0x3ff0000000000000),
                available: f64::from_bits(0x0000000000000000),
            },
            Update {
                time: f64::from_bits(0x4000000000000000),
                target: f64::from_bits(0x3ff0000000000000),
                available: f64::from_bits(0x0000000000000000),
            },
            Update {
                time: f64::from_bits(0x4010000000000000),
                target: f64::from_bits(0x3ff0000000000000),
                available: f64::from_bits(0x0000000000000000),
            },
        ],
        queries: &[
            Query {
                time: f64::from_bits(0x4014000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x3ff0000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x0000000000000000),
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
            Query {
                time: f64::from_bits(0x4010000000000000),
                expected: "Success",
            },
        ],
    },
];

pub(super) fn append(output: &mut Vec<Case>) {
    output.extend(CASES.iter().copied());
}
