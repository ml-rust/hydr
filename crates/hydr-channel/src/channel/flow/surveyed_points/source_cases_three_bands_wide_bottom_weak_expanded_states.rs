use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "trace_origin_before_retained_history_face_194",
        owner: 0,
        group: 21,
        inputs: [
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402bffffffffffce, 0xbccf22e12fa09a80, 0x0000000000000000],
            [0x3d40b60000000000, 0x0000000000000000, 0x0000000000000000],
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
                bounds: [0x3d031924924924b4, 0x3d031924924924b5],
                center: [0x3d031924924924b5, 0xb9a1d2e738bd16c5],
                residual: 0x35e38c4c1c40955d,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3d031924924924b4, 0x3d031924924924b5],
            center: [0x3d031924924924b5, 0xb9a1d2e738bd16c5],
            residual: 0x35e38c4c1c40955d,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3af067bde25eaffe),
    },
    Case {
        id: "trace_origin_before_retained_history_face_195",
        owner: 0,
        group: 5,
        inputs: [
            [0x402bffffffffffce, 0xbccf22e12fa09a80, 0x0000000000000000],
            [0x3d40b60000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402bffffffff216b, 0x3cc613ee6d159a80, 0x0000000000000000],
            [0x3de2672640000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3d031924924924b4, 0x3d031924924924b5],
                center: [0x3d031924924924b5, 0xb9a1d2e738bd16c5],
                residual: 0x35e38c4c1c40955d,
                non_dyadic: true,
            }),
            Some(Reference {
                bounds: [0x3da5082bb6dc14e7, 0x3da5082bb6dc14e8],
                center: [0x3da5082bb6dc14e7, 0x3a43385c84ebf43f],
                residual: 0x36c90adaf7710185,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3da503656db7829e, 0x3da503656db7829f],
            center: [0x3da503656db7829e, 0x3a2fb344fae88e13],
            residual: 0x36b496525142e50f,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3b9210f6955b3c68),
    },
    Case {
        id: "trace_origin_before_retained_history_face_196",
        owner: 0,
        group: 45,
        inputs: [
            [0x402bffffffff216b, 0x3cc613ee6d159a80, 0x0000000000000000],
            [0x3de2672640000000, 0x0000000000000000, 0x0000000000000000],
            [0x402bfffffe670550, 0xbccc3187adf50000, 0x0000000000000000],
            [0x3e70e83a8fe00000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3da5082bb6dc14e7, 0x3da5082bb6dc14e8],
                center: [0x3da5082bb6dc14e7, 0x3a43385c84ebf43f],
                residual: 0x36c90adaf7710185,
                non_dyadic: true,
            }),
            Some(Reference {
                bounds: [0x3e33528c133ecd8a, 0x3e33528c133ecd8b],
                center: [0x3e33528c133ecd8b, 0xbad09f6a13d32a72],
                residual: 0x372253039c4c0fd4,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3e334807fd635f80, 0x3e334807fd635f81],
            center: [0x3e334807fd635f80, 0x3ad276f9bdea5f94],
            residual: 0x3760eb72f0829d3e,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3c20990d6eece25f),
    },
    Case {
        id: "trace_origin_before_retained_history_face_197",
        owner: 0,
        group: 29,
        inputs: [
            [0x402bfffffe670550, 0xbccc3187adf50000, 0x0000000000000000],
            [0x3e70e83a8fe00000, 0x0000000000000000, 0x0000000000000000],
            [0x402bfffe6f05ec68, 0xbcc521cbcf800000, 0x0000000000000000],
            [0x3ef09364b4a8c000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3e33528c133ecd8a, 0x3e33528c133ecd8b],
                center: [0x3e33528c133ecd8b, 0xbad09f6a13d32a72],
                residual: 0x372253039c4c0fd4,
                non_dyadic: true,
            }),
            Some(Reference {
                bounds: [0x3eb2f198b92e926a, 0x3eb2f198b92e926b],
                center: [0x3eb2f198b92e926b, 0xbb3580605626a867],
                residual: 0x37cd5abc0cfd5ce0,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3eb2de462d1b539d, 0x3eb2de462d1b539e],
            center: [0x3eb2de462d1b539d, 0x3b57f087548a2911],
            residual: 0x37f72ac4e982c93c,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3ca045e98f483d79),
    },
];
