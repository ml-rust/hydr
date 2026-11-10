use super::{Case, Decision, Query};
use hydr::structures::GateActuatorAccuracy;
const CASES: &[Case] = &[Case {
    id: "old_target_left_endpoint_rejected",
    start: f64::from_bits(0x0000000000000000),
    end: f64::from_bits(0x4000000000000000),
    initial: f64::from_bits(0x0000000000000001),
    target: f64::from_bits(0x0000000000000000),
    opening_rate: f64::from_bits(0x0000000000000001),
    closing_rate: f64::from_bits(0x0000000000000001),
    low: f64::from_bits(0x0000000000000000),
    high: f64::from_bits(0x3ff0000000000000),
    lower_stage: f64::from_bits(0x3ff0000000000000),
    upper_stage: f64::from_bits(0x4000000000000000),
    high_delay: f64::from_bits(0x0000000000000000),
    low_delay: f64::from_bits(0x0000000000000000),
    gap: f64::from_bits(0x3ff8000000000000),
    lateness: f64::from_bits(0x0000000000000000),
    accuracy: GateActuatorAccuracy {
        maximum_absolute_opening_error_m: f64::from_bits(0x0000000000000002),
        maximum_relative_opening_error: f64::from_bits(0x0000000000000000),
        maximum_arrival_window_s: f64::from_bits(0x3d719799812dea11),
    },
    decisions: &[Decision {
        time: f64::from_bits(0x3fe0000000000000),
        available: f64::from_bits(0x3fe0000000000000),
        stage: f64::from_bits(0x4000000000000000),
        admitted: false,
    }],
    queries: &[
        Query {
            time: f64::from_bits(0x0000000000000000),
            expected: "Success",
        },
        Query {
            time: f64::from_bits(0x3fe0000000000000),
            expected: "Numerical",
        },
    ],
}];
pub(super) fn append(out: &mut Vec<Case>) {
    out.extend(CASES.iter().copied());
}
