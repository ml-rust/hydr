use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "scalar_planned_three_bands_both_knots_near_pure_fan",
        owner: 0,
        group: 26,
        inputs: [
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x400451bec8c475f1, 0x0000000000000000, 0x0000000000000000],
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
                bounds: [0x400451bec8c475f1, 0x400451bec8c475f1],
                center: [0x400451bec8c475f1, 0x0000000000000000],
                residual: 0x0000000000000000,
                non_dyadic: false,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x400451bec8c475f1, 0x400451bec8c475f1],
            center: [0x400451bec8c475f1, 0x0000000000000000],
            residual: 0x0000000000000000,
            non_dyadic: false,
        }),
        difference_zero: false,
        difference_allowance: None,
    },
    Case {
        id: "trace_origin_before_native_face_194",
        owner: 0,
        group: 10,
        inputs: [
            [0x402c000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x0000000000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402bffffffffffce, 0x0000000000000000, 0x0000000000000000],
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
                center: [0x3d031924924924b4, 0x39a8f0539782abf6],
                residual: 0x3644b61c5c85849c,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3d031924924924b4, 0x3d031924924924b5],
            center: [0x3d031924924924b4, 0x39a8f0539782abf6],
            residual: 0x3644b61c5c85849c,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3af067bde25eaffd),
    },
    Case {
        id: "trace_origin_before_native_face_195",
        owner: 0,
        group: 22,
        inputs: [
            [0x402bffffffffffce, 0x0000000000000000, 0x0000000000000000],
            [0x3d40b60000000000, 0x0000000000000000, 0x0000000000000000],
            [0x402bffffffff216b, 0x0000000000000000, 0x0000000000000000],
            [0x3de2672640000000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3d031924924924b4, 0x3d031924924924b5],
                center: [0x3d031924924924b4, 0x39a8f0539782abf6],
                residual: 0x3644b61c5c85849c,
                non_dyadic: true,
            }),
            Some(Reference {
                bounds: [0x3da5082bb6dc14e7, 0x3da5082bb6dc14e8],
                center: [0x3da5082bb6dc14e8, 0xba4c323f5aeb7dc9],
                residual: 0x368b44a5cbb9fba0,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3da503656db7829e, 0x3da503656db7829f],
            center: [0x3da503656db7829e, 0x3a488784902ea18c],
            residual: 0x367c0d77f41cb956,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3b9210fdc05d045c),
    },
    Case {
        id: "trace_origin_before_native_face_196",
        owner: 0,
        group: 60,
        inputs: [
            [0x402bffffffff216b, 0x0000000000000000, 0x0000000000000000],
            [0x3de2672640000000, 0x0000000000000000, 0x0000000000000000],
            [0x402bfffffe670550, 0x0000000000000000, 0x0000000000000000],
            [0x3e70e83a8fe00000, 0x0000000000000000, 0x0000000000000000],
        ],
        dry: [false, false],
        uniform: false,
        velocity: [
            Some(Reference {
                bounds: [0x3da5082bb6dc14e7, 0x3da5082bb6dc14e8],
                center: [0x3da5082bb6dc14e8, 0xba4c323f5aeb7dc9],
                residual: 0x368b44a5cbb9fba0,
                non_dyadic: true,
            }),
            Some(Reference {
                bounds: [0x3e33528c133ecd8a, 0x3e33528c133ecd8b],
                center: [0x3e33528c133ecd8a, 0x3adbebdbc137ffa2],
                residual: 0x3778749d946e3cc8,
                non_dyadic: true,
            }),
        ],
        difference: Some(Reference {
            bounds: [0x3e334807fd635f7f, 0x3e334807fd635f80],
            center: [0x3e334807fd635f80, 0xba9060b1f1a8a9ec],
            residual: 0x37394d41db9d43bb,
            non_dyadic: true,
        }),
        difference_zero: false,
        difference_allowance: Some(0x3c20990d6a593ae8),
    },
];
