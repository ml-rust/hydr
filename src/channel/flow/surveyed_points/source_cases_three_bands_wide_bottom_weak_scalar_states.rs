use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "trace_origin_before_native_face_197",
        owner: 0,
        group: 32,
        inputs: [
            [0x402bfffffe670550, 0x0000000000000000, 0x0000000000000000],
            [0x3e70e83a8fe00000, 0x0000000000000000, 0x0000000000000000],
            [0x402bfffe6f05ec68, 0x0000000000000000, 0x0000000000000000],
            [0x3ef09364b4a8c000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3e33528c133ecd8a, 0x3e33528c133ecd8b],
                center: [0x3e33528c133ecd8a, 0x3adbebdbc137ffa2],
                residual: 0x3778749d946e3cc8,
                non_dyadic: true,
            }),
            Some(Reference {
                bounds: [0x3eb2f198b92e926a, 0x3eb2f198b92e926b],
                center: [0x3eb2f198b92e926b, 0xbb53ac28e9b243e1],
                residual: 0x37fb6511e6eaced3,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3eb2de462d1b539d, 0x3eb2de462d1b539e],
            center: [0x3eb2de462d1b539d, 0x3b436fd675190840],
            residual: 0x37ea66c56d533ed4,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3ca045e98f44cf47),
    },
    Case {
        id: "trace_origin_after_native_face_193",
        owner: 0,
        group: 43,
        inputs: [
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
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
                center: [0x3c8d249249249249, 0x3912492492492492],
                residual: 0x35b2492492492493,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3c8d249249249249, 0x3c8d24924924924a],
            center: [0x3c8d249249249249, 0x3912492492492492],
            residual: 0x35b2492492492493,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3a79089501d2315c),
    },
    Case {
        id: "trace_origin_after_native_face_194",
        owner: 0,
        group: 17,
        inputs: [
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3cc9800000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402bfffffffffea1, 0x0000000000000000, 0x0000000000000000],
            [0x3d6d15e000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3c8d249249249249, 0x3c8d24924924924a],
                center: [0x3c8d249249249249, 0x3912492492492492],
                residual: 0x35b2492492492493,
                non_dyadic: true,
            }),
            Some(Reference {
                bounds: [0x3d309ec9249249f4, 0x3d309ec9249249f5],
                center: [0x3d309ec9249249f5, 0xb9b4efac68732f7d],
                residual: 0x3653b0e2c469666c,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3d309b2492492562, 0x3d309b2492492563],
            center: [0x3d309b2492492563, 0xb9d7850fac65f072],
            residual: 0x3678caa2bc9c81d3,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3b1c876866aee679),
    },
    Case {
        id: "trace_history_final_native_face_190",
        owner: 0,
        group: 19,
        inputs: [
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ca8000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3c6b6db6db6db6db, 0x3c6b6db6db6db6dc],
                center: [0x3c6b6db6db6db6db, 0x390b6db6db6db6db],
                residual: 0x35ab6db6db6db6dc,
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
            center: [0xbc6b6db6db6db6db, 0xb90b6db6db6db6db],
            residual: 0x35ab6db6db6db6dc,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3a578f9b4d021056),
    },
];
