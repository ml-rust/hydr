use super::{Case, Decision};
use hydr::structures::{ControlCommand, ControlLatch};
const CASES: &[Case] = &[
    Case {
        id: "stopped_running_reference",
        end: f64::from_bits(0x4010000000000000),
        initial_latch: ControlLatch::Low,
        low: ControlCommand::PumpStoppedIsolated,
        high: ControlCommand::PumpRunning {
            speed_ratio: f64::from_bits(0x3ff0000000000000),
        },
        high_delay: f64::from_bits(0x0000000000000000),
        low_delay: f64::from_bits(0x0000000000000000),
        gap: f64::from_bits(0x3ff0000000000000),
        affinity: false,
        adoption: f64::from_bits(0x0000000000000000),
        decisions: &[
            Decision {
                time: f64::from_bits(0x3ff0000000000000),
                available: f64::from_bits(0x3ff0000000000000),
                stage: f64::from_bits(0x4000000000000000),
            },
            Decision {
                time: f64::from_bits(0x4000000000000000),
                available: f64::from_bits(0x4000000000000000),
                stage: f64::from_bits(0x3ff8000000000000),
            },
            Decision {
                time: f64::from_bits(0x4008000000000000),
                available: f64::from_bits(0x4008000000000000),
                stage: f64::from_bits(0x3ff0000000000000),
            },
        ],
        queries: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3fe0000000000000),
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x3ff8000000000000),
            f64::from_bits(0x4000000000000000),
            f64::from_bits(0x4008000000000000),
            f64::from_bits(0x4010000000000000),
        ],
    },
    Case {
        id: "running_power_of_two_speed_changes",
        end: f64::from_bits(0x4010000000000000),
        initial_latch: ControlLatch::Low,
        low: ControlCommand::PumpRunning {
            speed_ratio: f64::from_bits(0x3fe0000000000000),
        },
        high: ControlCommand::PumpRunning {
            speed_ratio: f64::from_bits(0x4000000000000000),
        },
        high_delay: f64::from_bits(0x0000000000000000),
        low_delay: f64::from_bits(0x0000000000000000),
        gap: f64::from_bits(0x3ff0000000000000),
        affinity: true,
        adoption: f64::from_bits(0x0000000000000000),
        decisions: &[
            Decision {
                time: f64::from_bits(0x3ff0000000000000),
                available: f64::from_bits(0x3ff0000000000000),
                stage: f64::from_bits(0x4000000000000000),
            },
            Decision {
                time: f64::from_bits(0x4000000000000000),
                available: f64::from_bits(0x4000000000000000),
                stage: f64::from_bits(0x3ff0000000000000),
            },
            Decision {
                time: f64::from_bits(0x4008000000000000),
                available: f64::from_bits(0x4008000000000000),
                stage: f64::from_bits(0x4000000000000000),
            },
        ],
        queries: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3fe0000000000000),
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x3ff8000000000000),
            f64::from_bits(0x4000000000000000),
            f64::from_bits(0x4004000000000000),
            f64::from_bits(0x4008000000000000),
            f64::from_bits(0x4010000000000000),
        ],
    },
];
pub(super) fn append(output: &mut Vec<Case>) {
    output.extend(CASES.iter().copied());
}
