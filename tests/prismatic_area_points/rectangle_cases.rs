use super::{BandCase, Case};

pub const RECTANGLE_DRY_ZERO: Case = Case {
    id: "rectangle_dry_zero",
    owner: 0,
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
            [0x4018000000000000, 0x4018000000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x4008000000000000],
        ],
        allowances: [
            0x0000000000000000,
            0x3e049da7e361ce4c,
            0x0000000000000000,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [true, false, true, false],
    }],
};
pub const RECTANGLE_AREA_ONE_BELOW: Case = Case {
    id: "rectangle_area_one_below",
    owner: 0,
    input_high: 0x3ff0000000000000,
    input_low: 0xbaf0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fdfffffffffffff, 0x3fe0000000000000]),
        Some([0x3fefffffffffffff, 0x3ff0000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x3fcfffffffffffff, 0x3fd0000000000000]),
        Some([0x3fdfffffffffffff, 0x3fe0000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3fefffffffffffff, 0x3ff0000000000000],
            [0x4014000000000000, 0x4014000000000001],
            [0x3fdfffffffffffff, 0x3fe0000000000000],
            [0x4004000000000000, 0x4004000000000001],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3e012e0be826d694,
            0x3dcb7cdfd9d7bdba,
            0x3df12e0be826d694,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const RECTANGLE_AREA_ONE_EXACT: Case = Case {
    id: "rectangle_area_one_exact",
    owner: 0,
    input_high: 0x3ff0000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fe0000000000000, 0x3fe0000000000000]),
        Some([0x3ff0000000000000, 0x3ff0000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x3fd0000000000000, 0x3fd0000000000000]),
        Some([0x3fe0000000000000, 0x3fe0000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3ff0000000000000, 0x3ff0000000000000],
            [0x4014000000000000, 0x4014000000000000],
            [0x3fe0000000000000, 0x3fe0000000000000],
            [0x4004000000000000, 0x4004000000000000],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3e012e0be826d694,
            0x3dcb7cdfd9d7bdba,
            0x3df12e0be826d694,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const RECTANGLE_AREA_ONE_ABOVE: Case = Case {
    id: "rectangle_area_one_above",
    owner: 0,
    input_high: 0x3ff0000000000000,
    input_low: 0x3af0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fe0000000000000, 0x3fe0000000000001]),
        Some([0x3ff0000000000000, 0x3ff0000000000001]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x3fd0000000000000, 0x3fd0000000000001]),
        Some([0x3fe0000000000000, 0x3fe0000000000001]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3ff0000000000000, 0x3ff0000000000001],
            [0x4013ffffffffffff, 0x4014000000000000],
            [0x3fe0000000000000, 0x3fe0000000000001],
            [0x4003ffffffffffff, 0x4004000000000000],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3e012e0be826d694,
            0x3dcb7cdfd9d7bdba,
            0x3df12e0be826d694,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const RECTANGLE_EXACT_CAP: Case = Case {
    id: "rectangle_exact_cap",
    owner: 0,
    input_high: 0x4018000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4008000000000000, 0x4008000000000000]),
        Some([0x4018000000000000, 0x4018000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x4022000000000000, 0x4022000000000000]),
        Some([0x4008000000000000, 0x4008000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x4018000000000000, 0x4018000000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x4008000000000000],
            [0x0000000000000000, 0x0000000000000000],
        ],
        allowances: [
            0x3e049da7e361ce4c,
            0x0000000000000000,
            0x3df49da7e361ce4c,
            0x0000000000000000,
        ],
        exact_zero: [false, true, false, true],
    }],
};
pub const RECTANGLE_CAP_RETAINED_DEFICIT: Case = Case {
    id: "rectangle_cap_retained_deficit",
    owner: 0,
    input_high: 0x4018000000000000,
    input_low: 0xbaf0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4007ffffffffffff, 0x4008000000000000]),
        Some([0x4017ffffffffffff, 0x4018000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x4021ffffffffffff, 0x4022000000000000]),
        Some([0x4007ffffffffffff, 0x4008000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x4017ffffffffffff, 0x4018000000000000],
            [0x3af0000000000000, 0x3af0000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
            [0x3ae0000000000000, 0x3ae0000000000000],
        ],
        allowances: [
            0x3e049da7e361ce4c,
            0x38db7cdfd9d7bdba,
            0x3df49da7e361ce4c,
            0x38cb7cdfd9d7bdba,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const RECTANGLE_TINY_600_HIGH: Case = Case {
    id: "rectangle_tiny_600_high",
    owner: 0,
    input_high: 0x1a70000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x1a60000000000000, 0x1a60000000000000]),
        Some([0x1a70000000000000, 0x1a70000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x1a60000000000000, 0x1a60000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x1a70000000000000, 0x1a70000000000000],
            [0x4017ffffffffffff, 0x4018000000000000],
            [0x1a60000000000000, 0x1a60000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x185b7cdfd9d7bdba,
            0x3e049da7e361ce4c,
            0x184b7cdfd9d7bdba,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const RECTANGLE_TINY_600_LOW_ONLY: Case = Case {
    id: "rectangle_tiny_600_low_only",
    owner: 0,
    input_high: 0x0000000000000000,
    input_low: 0x1a70000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x1a60000000000000, 0x1a60000000000000]),
        Some([0x1a70000000000000, 0x1a70000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x1a60000000000000, 0x1a60000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x1a70000000000000, 0x1a70000000000000],
            [0x4017ffffffffffff, 0x4018000000000000],
            [0x1a60000000000000, 0x1a60000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x185b7cdfd9d7bdba,
            0x3e049da7e361ce4c,
            0x184b7cdfd9d7bdba,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const RECTANGLE_TINY_1000_HIGH: Case = Case {
    id: "rectangle_tiny_1000_high",
    owner: 0,
    input_high: 0x0170000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x0160000000000000, 0x0160000000000000]),
        Some([0x0170000000000000, 0x0170000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x0160000000000000, 0x0160000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0170000000000000, 0x0170000000000000],
            [0x4017ffffffffffff, 0x4018000000000000],
            [0x0160000000000000, 0x0160000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x000001b7cdfd9d7b,
            0x3e049da7e361ce4c,
            0x000000dbe6fecebd,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const RECTANGLE_TINY_1000_LOW_ONLY: Case = Case {
    id: "rectangle_tiny_1000_low_only",
    owner: 0,
    input_high: 0x0000000000000000,
    input_low: 0x0170000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x0160000000000000, 0x0160000000000000]),
        Some([0x0170000000000000, 0x0170000000000000]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x0160000000000000, 0x0160000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0170000000000000, 0x0170000000000000],
            [0x4017ffffffffffff, 0x4018000000000000],
            [0x0160000000000000, 0x0160000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x000001b7cdfd9d7b,
            0x3e049da7e361ce4c,
            0x000000dbe6fecebd,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const RECTANGLE_MINIMUM_HIGH: Case = Case {
    id: "rectangle_minimum_high",
    owner: 0,
    input_high: 0x0000000000000001,
    input_low: 0x0000000000000000,
    resolved: false,
    dry: false,
    geometry: [
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x0000000000000001, 0x0000000000000001]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x0000000000000000, 0x0000000000000001]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0000000000000001, 0x0000000000000001],
            [0x4017ffffffffffff, 0x4018000000000000],
            [0x0000000000000000, 0x0000000000000001],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x0000000000000000,
            0x3e049da7e361ce4c,
            0x0000000000000000,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const RECTANGLE_MINIMUM_LOW_ONLY: Case = Case {
    id: "rectangle_minimum_low_only",
    owner: 0,
    input_high: 0x0000000000000000,
    input_low: 0x0000000000000001,
    resolved: false,
    dry: false,
    geometry: [
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x0000000000000001, 0x0000000000000001]),
        Some([0x4000000000000000, 0x4000000000000000]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x0000000000000000, 0x0000000000000001]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0000000000000001, 0x0000000000000001],
            [0x4017ffffffffffff, 0x4018000000000000],
            [0x0000000000000000, 0x0000000000000001],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x0000000000000000,
            0x3e049da7e361ce4c,
            0x0000000000000000,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
