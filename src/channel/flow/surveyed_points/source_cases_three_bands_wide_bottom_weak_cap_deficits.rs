use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "trace_origin_after_retained_history_face_193",
        owner: 0,
        group: 57,
        inputs: [
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402c000000000000, 0xbcc105bc4806908f, 0x0000000000000000],
            [0x3cc9800000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x0000000000000000, 0x0000000000000000],
                center: [0x0000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
            Some(Reference {
                bounds: [0x3c8d249249249249, 0x3c8d24924924924a],
                center: [0x3c8d249249249249, 0x392adc2c74c2ea0d],
                residual: 0x35b58ac67afa1da6,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3c8d249249249249, 0x3c8d24924924924a],
            center: [0x3c8d249249249249, 0x392adc2c74c2ea0d],
            residual: 0x35b58ac67afa1da6,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3a79089501d2315c),
    },
    Case {
        id: "trace_origin_after_retained_history_face_194",
        owner: 0,
        group: 11,
        inputs: [
            [0x402c000000000000, 0xbcc105bc4806908f, 0x0000000000000000],
            [0x3cc9800000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402bfffffffffea0, 0x3cc0e106d850320f, 0x0000000000000000],
            [0x3d6d15e000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3c8d249249249249, 0x3c8d24924924924a],
                center: [0x3c8d249249249249, 0x392adc2c74c2ea0d],
                residual: 0x35b58ac67afa1da6,
                non_dyadic: true,
            }),
            Some(Reference {
                bounds: [0x3d309ec9249249f5, 0x3d309ec9249249f6],
                center: [0x3d309ec9249249f5, 0x39d6bc6db1d47d47],
                residual: 0x367152461b6264c1,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3d309b2492492563, 0x3d309b2492492564],
            center: [0x3d309b2492492563, 0x39b1c448b11793a8],
            residual: 0x362d1daddf56a3d9,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3b1c8ca9d282ebd0),
    },
    Case {
        id: "trace_history_final_retained_face_190",
        owner: 0,
        group: 14,
        inputs: [
            [0x402c000000000000, 0xbc189374bc6a7f02, 0x0000000000000000],
            [0x3ca8000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402c000000000000, 0xbc289374bc6a7f00, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3c6b6db6db6db6db, 0x3c6b6db6db6db6dc],
                center: [0x3c6b6db6db6db6db, 0x390b70b93d508a6e],
                residual: 0x35aa0fac13f7c4ad,
                non_dyadic: true,
            }),
            Some(Reference {
                bounds: [0x0000000000000000, 0x0000000000000000],
                center: [0x0000000000000000, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: Some(Reference {
            bounds: [0xbc6b6db6db6db6dc, 0xbc6b6db6db6db6db],
            center: [0xbc6b6db6db6db6db, 0xb90b70b93d508a6e],
            residual: 0x35aa0fac13f7c4ad,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3a578f9b4d021056),
    },
];
