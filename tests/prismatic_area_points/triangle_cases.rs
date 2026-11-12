use super::{BandCase, Case};

pub const TRIANGLE_DRY_ZERO: Case = Case {
    id: "triangle_dry_zero",
    owner: 1,
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
            [0x402b000000000000, 0x402b000000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x4008000000000000],
        ],
        allowances: [
            0x0000000000000000,
            0x3e17315cdfce0815,
            0x0000000000000000,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [true, false, true, false],
    }],
};
pub const TRIANGLE_AREA_ONE_BELOW: Case = Case {
    id: "triangle_area_one_below",
    owner: 1,
    input_high: 0x3ff0000000000000,
    input_low: 0xbaf0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e]),
        Some([0x3fefffffffffffff, 0x3ff0000000000000]),
        Some([0x4003988e1409212e, 0x4003988e1409212f]),
        Some([0x3fd16b28f55d72d3, 0x3fd16b28f55d72d4]),
        Some([0x3fda20bd700c2c3d, 0x3fda20bd700c2c3e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3fefffffffffffff, 0x3ff0000000000000],
            [0x4029000000000000, 0x4029000000000001],
            [0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e],
            [0x400177d0a3fcf4f0, 0x400177d0a3fcf4f1],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3e15798ee2308c39,
            0x3dd671957aa2c4d6,
            0x3dee028509723a2c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRIANGLE_AREA_ONE_EXACT: Case = Case {
    id: "triangle_area_one_exact",
    owner: 1,
    input_high: 0x3ff0000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e]),
        Some([0x3fefffffffffffff, 0x3ff0000000000001]),
        Some([0x4003988e1409212e, 0x4003988e1409212f]),
        Some([0x3fd16b28f55d72d3, 0x3fd16b28f55d72d4]),
        Some([0x3fda20bd700c2c3d, 0x3fda20bd700c2c3e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3ff0000000000000, 0x3ff0000000000000],
            [0x4029000000000000, 0x4029000000000000],
            [0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e],
            [0x400177d0a3fcf4f0, 0x400177d0a3fcf4f1],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3e15798ee2308c39,
            0x3dd671957aa2c4d6,
            0x3dee028509723a2c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRIANGLE_AREA_ONE_ABOVE: Case = Case {
    id: "triangle_area_one_above",
    owner: 1,
    input_high: 0x3ff0000000000000,
    input_low: 0x3af0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e]),
        Some([0x3ff0000000000000, 0x3ff0000000000001]),
        Some([0x4003988e1409212e, 0x4003988e1409212f]),
        Some([0x3fd16b28f55d72d3, 0x3fd16b28f55d72d4]),
        Some([0x3fda20bd700c2c3d, 0x3fda20bd700c2c3e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x3ff0000000000000, 0x3ff0000000000001],
            [0x4028ffffffffffff, 0x4029000000000000],
            [0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e],
            [0x400177d0a3fcf4f0, 0x400177d0a3fcf4f1],
        ],
        allowances: [
            0x3ddb7cdfd9d7bdba,
            0x3e15798ee2308c39,
            0x3dd671957aa2c4d6,
            0x3dee028509723a2c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRIANGLE_EXACT_CAP: Case = Case {
    id: "triangle_exact_cap",
    owner: 1,
    input_high: 0x402b000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4008000000000000, 0x4008000000000000]),
        Some([0x402b000000000000, 0x402b000000000000]),
        Some([0x4022000000000000, 0x4022000000000000]),
        Some([0x402b000000000000, 0x402b000000000000]),
        Some([0x3ff8000000000000, 0x3ff8000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x402b000000000000, 0x402b000000000000],
            [0x0000000000000000, 0x0000000000000000],
            [0x4008000000000000, 0x4008000000000000],
            [0x0000000000000000, 0x0000000000000000],
        ],
        allowances: [
            0x3e17315cdfce0815,
            0x0000000000000000,
            0x3df49da7e361ce4c,
            0x0000000000000000,
        ],
        exact_zero: [false, true, false, true],
    }],
};
pub const TRIANGLE_CAP_RETAINED_DEFICIT: Case = Case {
    id: "triangle_cap_retained_deficit",
    owner: 1,
    input_high: 0x402b000000000000,
    input_low: 0xbaf0000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x4007ffffffffffff, 0x4008000000000000]),
        Some([0x402affffffffffff, 0x402b000000000000]),
        Some([0x4021ffffffffffff, 0x4022000000000000]),
        Some([0x402affffffffffff, 0x402b000000000000]),
        Some([0x3ff7ffffffffffff, 0x3ff8000000000000]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x402affffffffffff, 0x402b000000000000],
            [0x3af0000000000000, 0x3af0000000000000],
            [0x4007ffffffffffff, 0x4008000000000000],
            [0x3abc71c71c71c71c, 0x3abc71c71c71c71d],
        ],
        allowances: [
            0x3e17315cdfce0815,
            0x38db7cdfd9d7bdba,
            0x3df49da7e361ce4c,
            0x38a86effde151a6d,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRIANGLE_TINY_600_HIGH: Case = Case {
    id: "triangle_tiny_600_high",
    owner: 1,
    input_high: 0x1a70000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x2d2a20bd700c2c3d, 0x2d2a20bd700c2c3e]),
        Some([0x1a6fffffffffffff, 0x1a70000000000001]),
        Some([0x2d43988e1409212e, 0x2d43988e1409212f]),
        Some([0x07916b28f55d72d3, 0x07916b28f55d72d4]),
        Some([0x2d1a20bd700c2c3d, 0x2d1a20bd700c2c3e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x1a70000000000000, 0x1a70000000000000],
            [0x402affffffffffff, 0x402b000000000000],
            [0x2d2a20bd700c2c3d, 0x2d2a20bd700c2c3e],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x185b7cdfd9d7bdba,
            0x3e17315cdfce0815,
            0x2b1671957aa2c4d6,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRIANGLE_TINY_600_LOW_ONLY: Case = Case {
    id: "triangle_tiny_600_low_only",
    owner: 1,
    input_high: 0x0000000000000000,
    input_low: 0x1a70000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x2d2a20bd700c2c3d, 0x2d2a20bd700c2c3e]),
        Some([0x1a6fffffffffffff, 0x1a70000000000001]),
        Some([0x2d43988e1409212e, 0x2d43988e1409212f]),
        Some([0x07916b28f55d72d3, 0x07916b28f55d72d4]),
        Some([0x2d1a20bd700c2c3d, 0x2d1a20bd700c2c3e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x1a70000000000000, 0x1a70000000000000],
            [0x402affffffffffff, 0x402b000000000000],
            [0x2d2a20bd700c2c3d, 0x2d2a20bd700c2c3e],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x185b7cdfd9d7bdba,
            0x3e17315cdfce0815,
            0x2b1671957aa2c4d6,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRIANGLE_TINY_1000_HIGH: Case = Case {
    id: "triangle_tiny_1000_high",
    owner: 1,
    input_high: 0x0170000000000000,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x20aa20bd700c2c3d, 0x20aa20bd700c2c3e]),
        Some([0x016fffffffffffff, 0x0170000000000001]),
        Some([0x20c3988e1409212e, 0x20c3988e1409212f]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x209a20bd700c2c3d, 0x209a20bd700c2c3e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0170000000000000, 0x0170000000000000],
            [0x402affffffffffff, 0x402b000000000000],
            [0x20aa20bd700c2c3d, 0x20aa20bd700c2c3e],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x000001b7cdfd9d7b,
            0x3e17315cdfce0815,
            0x1e9671957aa2c4d6,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRIANGLE_TINY_1000_LOW_ONLY: Case = Case {
    id: "triangle_tiny_1000_low_only",
    owner: 1,
    input_high: 0x0000000000000000,
    input_low: 0x0170000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x20aa20bd700c2c3d, 0x20aa20bd700c2c3e]),
        Some([0x016fffffffffffff, 0x0170000000000001]),
        Some([0x20c3988e1409212e, 0x20c3988e1409212f]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x209a20bd700c2c3d, 0x209a20bd700c2c3e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0170000000000000, 0x0170000000000000],
            [0x402affffffffffff, 0x402b000000000000],
            [0x20aa20bd700c2c3d, 0x20aa20bd700c2c3e],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x000001b7cdfd9d7b,
            0x3e17315cdfce0815,
            0x1e9671957aa2c4d6,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRIANGLE_MINIMUM_HIGH: Case = Case {
    id: "triangle_minimum_high",
    owner: 1,
    input_high: 0x0000000000000001,
    input_low: 0x0000000000000000,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x1e5a20bd700c2c3d, 0x1e5a20bd700c2c3e]),
        Some([0x0000000000000000, 0x0000000000000002]),
        Some([0x1e73988e1409212e, 0x1e73988e1409212f]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x1e4a20bd700c2c3d, 0x1e4a20bd700c2c3e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0000000000000001, 0x0000000000000001],
            [0x402affffffffffff, 0x402b000000000000],
            [0x1e5a20bd700c2c3d, 0x1e5a20bd700c2c3e],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x0000000000000000,
            0x3e17315cdfce0815,
            0x1c4671957aa2c4d6,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
pub const TRIANGLE_MINIMUM_LOW_ONLY: Case = Case {
    id: "triangle_minimum_low_only",
    owner: 1,
    input_high: 0x0000000000000000,
    input_low: 0x0000000000000001,
    resolved: true,
    dry: false,
    geometry: [
        Some([0x1e5a20bd700c2c3d, 0x1e5a20bd700c2c3e]),
        Some([0x0000000000000000, 0x0000000000000002]),
        Some([0x1e73988e1409212e, 0x1e73988e1409212f]),
        Some([0x0000000000000000, 0x0000000000000001]),
        Some([0x1e4a20bd700c2c3d, 0x1e4a20bd700c2c3e]),
    ],
    bands: &[BandCase {
        band_index: 0,
        bounds: [
            [0x0000000000000001, 0x0000000000000001],
            [0x402affffffffffff, 0x402b000000000000],
            [0x1e5a20bd700c2c3d, 0x1e5a20bd700c2c3e],
            [0x4007ffffffffffff, 0x4008000000000000],
        ],
        allowances: [
            0x0000000000000000,
            0x3e17315cdfce0815,
            0x1c4671957aa2c4d6,
            0x3df49da7e361ce4c,
        ],
        exact_zero: [false, false, false, false],
    }],
};
