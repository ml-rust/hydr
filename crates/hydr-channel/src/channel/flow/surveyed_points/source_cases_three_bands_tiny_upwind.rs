use super::model::{Case, Reference};

pub(crate) const CASES: &[Case] = &[Case {
    id: "captured_tiny_upwind_face_64",
    owner: 2,
    group: 6,
    inputs: [
        [0x399777f21e1a835c, 0x0000000000000000, 0x0000000000000000],
        [0x39bd89ee2425974f, 0x0000000000000000, 0x0000000000000000],
        [0x38e0225a9152b801, 0x0000000000000000, 0x0000000000000000],
        [0x39044cd69a2d92bf, 0x0000000000000000, 0x0000000000000000],
    ],
    dry: [false, false],
    uniform: false,
    velocity: [
        Some(Reference {
            bounds: [0x401423734b85b73c, 0x401423734b85b73d],
            center: [0x401423734b85b73c, 0x3cb5513d793c4951],
            residual: 0x3955c9f6003e7cfc,
            non_dyadic: true,
        }),
        Some(Reference {
            bounds: [0x4014219d38292ac6, 0x4014219d38292ac7],
            center: [0x4014219d38292ac6, 0x3c6cc1ecf6af5caf],
            residual: 0x3904cd05d6d5a56b,
            non_dyadic: true,
        }),
    ],
    difference: Some(Reference {
        bounds: [0xbf5d6135c8c7651b, 0xbf5d6135c8c7651a],
        center: [0xbf5d6135c8c7651b, 0x3bea3dcf26329992],
        residual: 0x387c8b9e0c0bd5e2,
        non_dyadic: true,
    }),
    difference_zero: false,
    difference_allowance: None,
}];
