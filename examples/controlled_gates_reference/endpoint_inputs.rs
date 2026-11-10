use super::{Case, Decision, Query};
use hydr::structures::GateActuatorAccuracy;
const CASES: &[Case] = &[
    Case {
        id: "old_closure_and_new_opening_command",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4018000000000000),
        initial: f64::from_bits(0x3ff0000000000000),
        target: f64::from_bits(0x0000000000000000),
        opening_rate: f64::from_bits(0x3fd0000000000000),
        closing_rate: f64::from_bits(0x3fe0000000000000),
        low: f64::from_bits(0x0000000000000000),
        high: f64::from_bits(0x3ff0000000000000),
        lower_stage: f64::from_bits(0x3ff0000000000000),
        upper_stage: f64::from_bits(0x4000000000000000),
        high_delay: f64::from_bits(0x0000000000000000),
        low_delay: f64::from_bits(0x0000000000000000),
        gap: f64::from_bits(0x4000000000000000),
        lateness: f64::from_bits(0x0000000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        decisions: &[
            Decision {
                time: f64::from_bits(0x4000000000000000),
                available: f64::from_bits(0x4000000000000000),
                stage: f64::from_bits(0x4000000000000000),
                admitted: true,
            },
            Decision {
                time: f64::from_bits(0x4010000000000000),
                available: f64::from_bits(0x4010000000000000),
                stage: f64::from_bits(0x3ff8000000000000),
                admitted: true,
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
                time: f64::from_bits(0x4008000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4010000000000000),
                expected: "Success",
            },
            Query {
                time: f64::from_bits(0x4018000000000000),
                expected: "Success",
            },
        ],
    },
    Case {
        id: "signed_zero_equal_targets",
        start: f64::from_bits(0x0000000000000000),
        end: f64::from_bits(0x4008000000000000),
        initial: f64::from_bits(0x0000000000000000),
        target: f64::from_bits(0x8000000000000000),
        opening_rate: f64::from_bits(0x0000000000000000),
        closing_rate: f64::from_bits(0x0000000000000000),
        low: f64::from_bits(0x8000000000000000),
        high: f64::from_bits(0x0000000000000000),
        lower_stage: f64::from_bits(0x3ff0000000000000),
        upper_stage: f64::from_bits(0x4000000000000000),
        high_delay: f64::from_bits(0x0000000000000000),
        low_delay: f64::from_bits(0x0000000000000000),
        gap: f64::from_bits(0x3ff0000000000000),
        lateness: f64::from_bits(0x0000000000000000),
        accuracy: GateActuatorAccuracy {
            maximum_absolute_opening_error_m: f64::from_bits(0x3d719799812dea11),
            maximum_relative_opening_error: f64::from_bits(0x3d719799812dea11),
            maximum_arrival_window_s: f64::from_bits(0x3ddb7cdfd9d7bdbb),
        },
        decisions: &[
            Decision {
                time: f64::from_bits(0x3ff0000000000000),
                available: f64::from_bits(0x3ff0000000000000),
                stage: f64::from_bits(0x4000000000000000),
                admitted: true,
            },
            Decision {
                time: f64::from_bits(0x4000000000000000),
                available: f64::from_bits(0x4000000000000000),
                stage: f64::from_bits(0x3ff0000000000000),
                admitted: true,
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
];
pub(super) fn append(out: &mut Vec<Case>) {
    out.extend(CASES.iter().copied());
}
