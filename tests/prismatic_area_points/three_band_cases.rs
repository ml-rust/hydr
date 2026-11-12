use super::{BandCase, Case};

pub const THREE_BAND_DRY_ZERO: Case = Case {
    id: "three_band_dry_zero",
    owner: 3,
    input_high: 0x0000000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: true,
    geometry: [
        Some([0x0000000000000000, 0x0000000000000000]),
        Some([0x0000000000000000, 0x0000000000000000]),
        None,
        Some([0x0000000000000000, 0x0000000000000000]),
        None,
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x4008000000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x3ff0000000000000],
        ],
        allowances: [
            0x0000000000000000,
            0x3df49da7e361ce4c,
            0x0000000000000000,
            0x3ddb7cdfd9d7bdba,
        ],
        exact_zero: [true, false, true, false],
    }],
};
pub const THREE_BAND_AREA_ONE: Case = Case {
    id: "three_band_area_one",
    owner: 3,
    input_high: 0x3ff0000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fda827999fcef32, 0x3fda827999fcef33]),
        Some([0x3fefffffffffffff, 0x3ff0000000000001]),
        Some([0x4006a09e667f3bcc, 0x4006a09e667f3bcd]),
        Some([0x3fc8fe5999576089, 0x3fc8fe599957608a]),
        Some([0x3fd6a09e667f3bcc, 0x3fd6a09e667f3bcd]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3ff0000000000000, 0x3ff0000000000000],
            [0x4000000000000000, 0x4000000000000000],
            [0x3fda827999fcef32, 0x3fda827999fcef33],
            [0x3fe2bec333018866, 0x3fe2bec333018867],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3deb7cdfd9d7bdba,
            0x3dc6c589a2d90976,
            0x3dd01a1b086b38ff,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const THREE_BAND_KNOT_3_BELOW: Case = Case {
    id: "three_band_knot_3_below",
    owner: 3,
    input_high: 0x4008000000000000,
    input_low: 0xbaf0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fefffffffffffff, 0x3ff0000000000000]),
        Some([0x4007ffffffffffff, 0x4008000000000000]),
        Some([0x400fffffffffffff, 0x4010000000000000]),
        Some([0x3ff5555555555555, 0x3ff5555555555556]),
        Some([0x3fe7ffffffffffff, 0x3fe8000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x4007ffffffffffff, 0x4008000000000000],
            [0x3af0000000000000, 0x3af0000000000000],
            [0x3fefffffffffffff, 0x3ff0000000000000],
            [0x3ad0000000000000, 0x3ad0000000000001],
        ],
        allowances: [
            0x3df49da7e361ce4c,
            0x38db7cdfd9d7bdba,
            0x3ddb7cdfd9d7bdba,
            0x38bb7cdfd9d7bdba,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const THREE_BAND_KNOT_3_EXACT: Case = Case {
    id: "three_band_knot_3_exact",
    owner: 3,
    input_high: 0x4008000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3ff0000000000000, 0x3ff0000000000000]),
        Some([0x4008000000000000, 0x4008000000000000]),
        Some([0x4010000000000000, 0x4010000000000000]),
        Some([0x3ff5555555555555, 0x3ff5555555555556]),
        Some([0x3fe8000000000000, 0x3fe8000000000000]),
    ],
    bands: &[
        BandCase {
            band_index: 0,
            bounds: [
                [0x4008000000000000, 0x4008000000000000],
                [0x0000000000000000, 0x0000000000000000],
                [0x3ff0000000000000, 0x3ff0000000000000],
                [0x0000000000000000, 0x0000000000000000],
            ],
            allowances: [
                0x3df49da7e361ce4c,
                0x0000000000000000,
                0x3ddb7cdfd9d7bdba,
                0x0000000000000000,
            ],
            exact_zero: [false, true, false, true],
        },
        BandCase {
            band_index: 1,
            bounds: [
                [0x0000000000000000, 0x0000000000000000],
                [0x4012000000000000, 0x4012000000000000],
                [0x0000000000000000, 0x0000000000000000],
                [0x3ff0000000000000, 0x3ff0000000000000],
            ],
            allowances: [
                0x0000000000000000,
                0x3dfeec7bd512b572,
                0x0000000000000000,
                0x3ddb7cdfd9d7bdba,
            ],
            exact_zero: [true, false, true, false],
        },
    ],
};
pub const THREE_BAND_KNOT_3_ABOVE: Case = Case {
    id: "three_band_knot_3_above",
    owner: 3,
    input_high: 0x4008000000000000,
    input_low: 0x3af0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3ff0000000000000, 0x3ff0000000000001]),
        Some([0x4008000000000000, 0x4008000000000001]),
        Some([0x4010000000000000, 0x4010000000000001]),
        Some([0x3ff5555555555555, 0x3ff5555555555556]),
        Some([0x3fe8000000000000, 0x3fe8000000000001]),
    ],
    bands: &[BandCase {
        band_index: 1,
        bounds: [
            [0x3af0000000000000, 0x3af0000000000000],
            [0x4011ffffffffffff, 0x4012000000000000],
            [0x3acfffffffffffff, 0x3ad0000000000000],
            [0x3fefffffffffffff, 0x3ff0000000000000],
        ],
        allowances: [
            0x38db7cdfd9d7bdba,
            0x3dfeec7bd512b572,
            0x38bb7cdfd9d7bdba,
            0x3ddb7cdfd9d7bdba,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const THREE_BAND_KNOT_15_2_BELOW: Case = Case {
    id: "three_band_knot_15_2_below",
    owner: 3,
    input_high: 0x401e000000000000,
    input_low: 0xbaf0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fffffffffffffff, 0x4000000000000000]),
        Some([0x401dffffffffffff, 0x401e000000000000]),
        Some([0x4013ffffffffffff, 0x4014000000000000]),
        Some([0x4019ffffffffffff, 0x401a000000000000]),
        Some([0x3ff7ffffffffffff, 0x3ff8000000000000]),
    ],
    bands: &[BandCase {
        band_index: 1,
        bounds: [
            [0x4011ffffffffffff, 0x4012000000000000],
            [0x3af0000000000000, 0x3af0000000000000],
            [0x3fefffffffffffff, 0x3ff0000000000000],
            [0x3ac9999999999999, 0x3ac999999999999a],
        ],
        allowances: [
            0x3dfeec7bd512b572,
            0x38db7cdfd9d7bdba,
            0x3ddb7cdfd9d7bdba,
            0x38b5fd7fe1796495,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const THREE_BAND_KNOT_15_2_EXACT: Case = Case {
    id: "three_band_knot_15_2_exact",
    owner: 3,
    input_high: 0x401e000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x401e000000000000, 0x401e000000000000]),
        Some([0x4014000000000000, 0x4014000000000000]),
        Some([0x401a000000000000, 0x401a000000000000]),
        Some([0x3ff8000000000000, 0x3ff8000000000000]),
    ],
    bands: &[
        BandCase {
            band_index: 1,
            bounds: [
                [0x4012000000000000, 0x4012000000000000],
                [0x0000000000000000, 0x0000000000000000],
                [0x3ff0000000000000, 0x3ff0000000000000],
                [0x0000000000000000, 0x0000000000000000],
            ],
            allowances: [
                0x3dfeec7bd512b572,
                0x0000000000000000,
                0x3ddb7cdfd9d7bdba,
                0x0000000000000000,
            ],
            exact_zero: [false, true, false, true],
        },
        BandCase {
            band_index: 2,
            bounds: [
                [0x0000000000000000, 0x0000000000000000],
                [0x401a000000000000, 0x401a000000000000],
                [0x0000000000000000, 0x0000000000000000],
                [0x3ff0000000000000, 0x3ff0000000000000],
            ],
            allowances: [
                0x0000000000000000,
                0x3e065575e0ff4a27,
                0x0000000000000000,
                0x3ddb7cdfd9d7bdba,
            ],
            exact_zero: [true, false, true, false],
        },
    ],
};
pub const THREE_BAND_KNOT_15_2_ABOVE: Case = Case {
    id: "three_band_knot_15_2_above",
    owner: 3,
    input_high: 0x401e000000000000,
    input_low: 0x3af0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4000000000000000, 0x4000000000000001]),
        Some([0x401e000000000000, 0x401e000000000001]),
        Some([0x4014000000000000, 0x4014000000000001]),
        Some([0x401a000000000000, 0x401a000000000001]),
        Some([0x3ff8000000000000, 0x3ff8000000000001]),
    ],
    bands: &[BandCase {
        band_index: 2,
        bounds: [
            [0x3af0000000000000, 0x3af0000000000000],
            [0x4019ffffffffffff, 0x401a000000000000],
            [0x3ac9999999999999, 0x3ac999999999999a],
            [0x3fefffffffffffff, 0x3ff0000000000000],
        ],
        allowances: [
            0x38db7cdfd9d7bdba,
            0x3e065575e0ff4a27,
            0x38b5fd7fe1796495,
            0x3ddb7cdfd9d7bdba,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const THREE_BAND_EXACT_CAP: Case = Case {
    id: "three_band_exact_cap",
    owner: 3,
    input_high: 0x402c000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4008000000000000, 0x4008000000000000]),
        Some([0x402c000000000000, 0x402c000000000000]),
        Some([0x4020000000000000, 0x4020000000000000]),
        Some([0x4031000000000000, 0x4031000000000000]),
        Some([0x3ffc000000000000, 0x3ffc000000000000]),
    ],
    bands: &[BandCase {
        band_index: 2,
        bounds: [
            [0x401a000000000000, 0x401a000000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x3ff0000000000000],
            [0x0000000000000000, 0x0000000000000000],
        ],
        allowances: [
            0x3e065575e0ff4a27,
            0x0000000000000000,
            0x3ddb7cdfd9d7bdba,
            0x0000000000000000,
        ],
        exact_zero: [false, true, false, true],
    }],
};
pub const THREE_BAND_CAP_RETAINED_DEFICIT: Case = Case {
    id: "three_band_cap_retained_deficit",
    owner: 3,
    input_high: 0x402c000000000000,
    input_low: 0xbaf0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4007ffffffffffff, 0x4008000000000000]),
        Some([0x402bffffffffffff, 0x402c000000000000]),
        Some([0x401fffffffffffff, 0x4020000000000000]),
        Some([0x4030ffffffffffff, 0x4031000000000000]),
        Some([0x3ffbffffffffffff, 0x3ffc000000000000]),
    ],
    bands: &[BandCase {
        band_index: 2,
        bounds: [
            [0x4019ffffffffffff, 0x401a000000000000],
            [0x3af0000000000000, 0x3af0000000000000],
            [0x3fefffffffffffff, 0x3ff0000000000000],
            [0x3ac0000000000000, 0x3ac0000000000001],
        ],
        allowances: [
            0x3e065575e0ff4a27,
            0x38db7cdfd9d7bdba,
            0x3ddb7cdfd9d7bdba,
            0x38ab7cdfd9d7bdba,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const THREE_BAND_TINY_600_LOW_ONLY: Case = Case {
    id: "three_band_tiny_600_low_only",
    owner: 3,
    input_high: 0x0000000000000000,
    input_low: 0x1a70000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x1a5fffffffffffff, 0x1a60000000000000]),
        Some([0x1a6fffffffffffff, 0x1a70000000000001]),
        Some([0x4000000000000000, 0x4000000000000001]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x1a5fffffffffffff, 0x1a60000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x1a70000000000000, 0x1a70000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
            [0x1a5fffffffffffff, 0x1a60000000000000],
            [0x3fefffffffffffff, 0x3ff0000000000000],
        ],
        allowances: [
            0x185b7cdfd9d7bdba,
            0x3df49da7e361ce4c,
            0x184b7cdfd9d7bdba,
            0x3ddb7cdfd9d7bdba,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const THREE_BAND_TINY_1000_LOW_ONLY: Case = Case {
    id: "three_band_tiny_1000_low_only",
    owner: 3,
    input_high: 0x0000000000000000,
    input_low: 0x0170000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x015fffffffffffff, 0x0160000000000000]),
        Some([0x016fffffffffffff, 0x0170000000000001]),
        Some([0x4000000000000000, 0x4000000000000001]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x015fffffffffffff, 0x0160000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0170000000000000, 0x0170000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
            [0x015fffffffffffff, 0x0160000000000000],
            [0x3fefffffffffffff, 0x3ff0000000000000],
        ],
        allowances: [
            0x000001b7cdfd9d7b,
            0x3df49da7e361ce4c,
            0x000000dbe6fecebd,
            0x3ddb7cdfd9d7bdba,
        ],
        exact_zero: [false, false, false, false],
    }],
};
