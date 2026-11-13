use super::model::{AnchorKind, Chart, InvalidCase};

pub(crate) const INVALID: &[InvalidCase] = &[
    InvalidCase {
        id: "area_below_dry",
        cause: "invalid_physical_support",
        chart: Chart {
            owner: 1,
            kind: AnchorKind::Zero,
            band: 0,
            source: [0; 3],
            add: false,
            scale: [0x1a70000000000000, 0x0000000000000000, 0x0000000000000000],
            parameter: [0x0000000000000000, 0x3ff0000000000000],
        },
    },
    InvalidCase {
        id: "area_above_cap",
        cause: "invalid_physical_support",
        chart: Chart {
            owner: 1,
            kind: AnchorKind::Cap,
            band: 0,
            source: [0; 3],
            add: true,
            scale: [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            parameter: [0x3af0000000000000, 0x3b00000000000000],
        },
    },
    InvalidCase {
        id: "negative_cap_deficit",
        cause: "negative_cap_deficit",
        chart: Chart {
            owner: 2,
            kind: AnchorKind::Cap,
            band: 0,
            source: [0; 3],
            add: false,
            scale: [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            parameter: [0xbaf0000000000000, 0x0000000000000000],
        },
    },
    InvalidCase {
        id: "unknown_band_metadata",
        cause: "invalid_band_metadata",
        chart: Chart {
            owner: 2,
            kind: AnchorKind::BandStart,
            band: 99,
            source: [0; 3],
            add: true,
            scale: [0x3af0000000000000, 0x0000000000000000, 0x0000000000000000],
            parameter: [0xbff0000000000000, 0x3ff0000000000000],
        },
    },
    InvalidCase {
        id: "negative_chart_scale",
        cause: "invalid_nonnegative_scale",
        chart: Chart {
            owner: 1,
            kind: AnchorKind::Zero,
            band: 0,
            source: [0; 3],
            add: true,
            scale: [0xbff0000000000000, 0x0000000000000000, 0x0000000000000000],
            parameter: [0x0000000000000000, 0x3ff0000000000000],
        },
    },
    InvalidCase {
        id: "unordered_parameter_metadata",
        cause: "invalid_parameter_metadata",
        chart: Chart {
            owner: 1,
            kind: AnchorKind::Zero,
            band: 0,
            source: [0; 3],
            add: true,
            scale: [0x1a70000000000000, 0x0000000000000000, 0x0000000000000000],
            parameter: [0x4000000000000000, 0x3ff0000000000000],
        },
    },
];
