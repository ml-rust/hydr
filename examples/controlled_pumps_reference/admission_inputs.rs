use super::{Admission, Support};
use hydr::structures::ControlCommand;
const CASES: &[Admission] = &[
    Admission {
        id: "stopped_unknown_isolated",
        operation: ControlCommand::PumpStoppedIsolated,
        donor: Support::Unknown,
        receiver: Support::Unknown,
        expected: "Success",
    },
    Admission {
        id: "running_shutoff_unknown_not_isolated",
        operation: ControlCommand::PumpRunning {
            speed_ratio: f64::from_bits(0x3fe0000000000000),
        },
        donor: Support::Unknown,
        receiver: Support::Unknown,
        expected: "UnknownSupport",
    },
    Admission {
        id: "running_reference_unknown",
        operation: ControlCommand::PumpRunning {
            speed_ratio: f64::from_bits(0x3ff0000000000000),
        },
        donor: Support::Unknown,
        receiver: Support::Unknown,
        expected: "UnknownSupport",
    },
    Admission {
        id: "running_shutoff_wet",
        operation: ControlCommand::PumpRunning {
            speed_ratio: f64::from_bits(0x3fe0000000000000),
        },
        donor: Support::Wet(f64::from_bits(0x4000000000000000)),
        receiver: Support::Wet(f64::from_bits(0x4008000000000000)),
        expected: "Success",
    },
    Admission {
        id: "running_reference_outside_closed_curve",
        operation: ControlCommand::PumpRunning {
            speed_ratio: f64::from_bits(0x3ff0000000000000),
        },
        donor: Support::Wet(f64::from_bits(0x4000000000000000)),
        receiver: Support::Wet(f64::from_bits(0x401c000000000000)),
        expected: "UnsupportedRegime",
    },
    Admission {
        id: "half_speed_positive_shutoff_neighbor",
        operation: ControlCommand::PumpRunning {
            speed_ratio: f64::from_bits(0x3fe0000000000000),
        },
        donor: Support::Wet(f64::from_bits(0x4000000000000000)),
        receiver: Support::Wet(f64::from_bits(0x4007ffffffffffff)),
        expected: "Success",
    },
    Admission {
        id: "half_speed_outside_shutoff_neighbor",
        operation: ControlCommand::PumpRunning {
            speed_ratio: f64::from_bits(0x3fe0000000000000),
        },
        donor: Support::Wet(f64::from_bits(0x4000000000000000)),
        receiver: Support::Wet(f64::from_bits(0x4008000000000001)),
        expected: "UnsupportedRegime",
    },
    Admission {
        id: "stopped_malformed_known_head",
        operation: ControlCommand::PumpStoppedIsolated,
        donor: Support::MalformedKnownNaN,
        receiver: Support::Unknown,
        expected: "NonfiniteInput",
    },
];
pub fn admissions() -> &'static [Admission] {
    CASES
}
