use super::{BandCase, Case};

pub const TRAPEZOID_DRY_ZERO: Case = Case {
    id: "trapezoid_dry_zero",
    owner: 2,
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
            [0x4033800000000000, 0x4033800000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x4008000000000000],
        ],
        allowances: [
            0x0000000000000000,
            0x3e20c01868bf779d,
            0x0000000000000000,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [true, false, true, false],
    }],
};
pub const TRAPEZOID_AREA_ONE_BELOW: Case = Case {
    id: "trapezoid_area_one_below",
    owner: 2,
    input_high: 0x3ff0000000000000,
    input_low: 0xbaf0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fd8cb95f24648dc, 0x3fd8cb95f24648dd]),
        Some([0x3fefffffffffffff, 0x3ff0000000000000]),
        Some([0x40094c583ada5b52, 0x40094c583ada5b53]),
        Some([0x3fc6ef34030cd35d, 0x3fc6ef34030cd35e]),
        Some([0x3fd43d136248490e, 0x3fd43d136248490f]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3fefffffffffffff, 0x3ff0000000000000],
            [0x4032800000000000, 0x4032800000000001],
            [0x3fd8cb95f24648dc, 0x3fd8cb95f24648dd],
            [0x4004e68d41b736e4, 0x4004e68d41b736e5],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3e1fc862d3e1735f,
            0x3dc54c88dcfec9db,
            0x3df1f416c7c1f510,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRAPEZOID_AREA_ONE_EXACT: Case = Case {
    id: "trapezoid_area_one_exact",
    owner: 2,
    input_high: 0x3ff0000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fd8cb95f24648dc, 0x3fd8cb95f24648dd]),
        Some([0x3fefffffffffffff, 0x3ff0000000000001]),
        Some([0x40094c583ada5b52, 0x40094c583ada5b53]),
        Some([0x3fc6ef34030cd35d, 0x3fc6ef34030cd35e]),
        Some([0x3fd43d136248490e, 0x3fd43d136248490f]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3ff0000000000000, 0x3ff0000000000000],
            [0x4032800000000000, 0x4032800000000000],
            [0x3fd8cb95f24648dc, 0x3fd8cb95f24648dd],
            [0x4004e68d41b736e4, 0x4004e68d41b736e5],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3e1fc862d3e1735f,
            0x3dc54c88dcfec9db,
            0x3df1f416c7c1f510,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRAPEZOID_AREA_ONE_ABOVE: Case = Case {
    id: "trapezoid_area_one_above",
    owner: 2,
    input_high: 0x3ff0000000000000,
    input_low: 0x3af0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fd8cb95f24648dc, 0x3fd8cb95f24648dd]),
        Some([0x3ff0000000000000, 0x3ff0000000000001]),
        Some([0x40094c583ada5b52, 0x40094c583ada5b53]),
        Some([0x3fc6ef34030cd35d, 0x3fc6ef34030cd35e]),
        Some([0x3fd43d136248490e, 0x3fd43d136248490f]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3ff0000000000000, 0x3ff0000000000001],
            [0x40327fffffffffff, 0x4032800000000000],
            [0x3fd8cb95f24648dc, 0x3fd8cb95f24648dd],
            [0x4004e68d41b736e4, 0x4004e68d41b736e5],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3e1fc862d3e1735f,
            0x3dc54c88dcfec9db,
            0x3df1f416c7c1f510,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRAPEZOID_EXACT_CAP: Case = Case {
    id: "trapezoid_exact_cap",
    owner: 2,
    input_high: 0x4033800000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4008000000000000, 0x4008000000000000]),
        Some([0x4033800000000000, 0x4033800000000000]),
        Some([0x4026000000000000, 0x4026000000000000]),
        Some([0x4036800000000000, 0x4036800000000000]),
        Some([0x3ffc5d1745d1745d, 0x3ffc5d1745d1745e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x4033800000000000, 0x4033800000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x4008000000000000],
            [0x0000000000000000, 0x0000000000000000],
        ],
        allowances: [
            0x3e20c01868bf779d,
            0x0000000000000000,
            0x3df49da7e361ce4c,
            0x0000000000000000,
        ],
        exact_zero: [false, true, false, true],
    }],
};
pub const TRAPEZOID_CAP_RETAINED_DEFICIT: Case = Case {
    id: "trapezoid_cap_retained_deficit",
    owner: 2,
    input_high: 0x4033800000000000,
    input_low: 0xbaf0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4007ffffffffffff, 0x4008000000000000]),
        Some([0x40337fffffffffff, 0x4033800000000000]),
        Some([0x4025ffffffffffff, 0x4026000000000000]),
        Some([0x40367fffffffffff, 0x4036800000000000]),
        Some([0x3ffc5d1745d1745d, 0x3ffc5d1745d1745e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x40337fffffffffff, 0x4033800000000000],
            [0x3af0000000000000, 0x3af0000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
            [0x3ab745d1745d1745, 0x3ab745d1745d1746],
        ],
        allowances: [
            0x3e20c01868bf779d,
            0x38db7cdfd9d7bdba,
            0x3df49da7e361ce4c,
            0x38a3fdba12cb72b6,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRAPEZOID_TINY_600_HIGH: Case = Case {
    id: "trapezoid_tiny_600_high",
    owner: 2,
    input_high: 0x1a70000000000000,
    input_low: 0x0000000000000000,
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
            [0x40337fffffffffff, 0x4033800000000000],
            [0x1a5fffffffffffff, 0x1a60000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x185b7cdfd9d7bdba,
            0x3e20c01868bf779d,
            0x184b7cdfd9d7bdba,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRAPEZOID_TINY_600_LOW_ONLY: Case = Case {
    id: "trapezoid_tiny_600_low_only",
    owner: 2,
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
            [0x40337fffffffffff, 0x4033800000000000],
            [0x1a5fffffffffffff, 0x1a60000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x185b7cdfd9d7bdba,
            0x3e20c01868bf779d,
            0x184b7cdfd9d7bdba,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRAPEZOID_TINY_1000_HIGH: Case = Case {
    id: "trapezoid_tiny_1000_high",
    owner: 2,
    input_high: 0x0170000000000000,
    input_low: 0x0000000000000000,
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
            [0x40337fffffffffff, 0x4033800000000000],
            [0x015fffffffffffff, 0x0160000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x000001b7cdfd9d7b,
            0x3e20c01868bf779d,
            0x000000dbe6fecebd,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRAPEZOID_TINY_1000_LOW_ONLY: Case = Case {
    id: "trapezoid_tiny_1000_low_only",
    owner: 2,
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
            [0x40337fffffffffff, 0x4033800000000000],
            [0x015fffffffffffff, 0x0160000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x000001b7cdfd9d7b,
            0x3e20c01868bf779d,
            0x000000dbe6fecebd,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
