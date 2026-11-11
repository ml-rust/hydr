use super::model::{InverseCase, inverse, truth};

// Exact area comparisons bracket each inverse independently of production recovery.
pub(super) const CASES: [InverseCase; 8] = [
    inverse(
        3,
        (0x0000000000000000, Some(0x0000000000000000)),
        truth(0x0000000000000000, 0x0000000000000000, 0x3df49da7e361ce4c),
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
    ),
    inverse(
        3,
        (0x8000000000000000, Some(0x8000000000000000)),
        truth(0x0000000000000000, 0x0000000000000000, 0x3df49da7e361ce4c),
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
    ),
    inverse(
        3,
        (0x3ff4000000000000, Some(0x3fe0000000000000)),
        truth(0x3fe0000000000000, 0x3fe0000000000000, 0x3df49da7e361ce4c),
        truth(0x3ff4000000000000, 0x3ff4000000000000, 0x3de12e0be826d694),
        Some(truth(
            0x4008000000000000,
            0x4008000000000000,
            0x3df49da7e361ce4c,
        )),
        truth(0x3fd2aaaaaaaaaaaa, 0x3fd2aaaaaaaaaaab, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x3fdaaaaaaaaaaaaa,
            0x3fdaaaaaaaaaaaab,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    inverse(
        3,
        (0x4008000000000000, Some(0x3ff0000000000000)),
        truth(0x3ff0000000000000, 0x3ff0000000000000, 0x3df49da7e361ce4c),
        truth(0x4008000000000000, 0x4008000000000000, 0x3df49da7e361ce4c),
        Some(truth(
            0x4010000000000000,
            0x4010000000000000,
            0x3dfb7cdfd9d7bdba,
        )),
        truth(0x3ff5555555555555, 0x3ff5555555555556, 0x3de2533fe68fd3d1),
        Some(truth(
            0x3fe8000000000000,
            0x3fe8000000000000,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    inverse(
        3,
        (0x4014800000000000, Some(0x3ff8000000000000)),
        truth(0x3ff8000000000000, 0x3ff8000000000000, 0x3df49da7e361ce4c),
        truth(0x4014800000000000, 0x4014800000000000, 0x3e019bff678e358b),
        Some(truth(
            0x4012000000000000,
            0x4012000000000000,
            0x3dfeec7bd512b572,
        )),
        truth(0x400ad55555555555, 0x400ad55555555556, 0x3df70cb66000e86d),
        Some(truth(
            0x3ff238e38e38e38e,
            0x3ff238e38e38e38f,
            0x3ddf4e37d48b09db,
        )),
    ),
    inverse(
        3,
        (0x401e000000000000, Some(0x4000000000000000)),
        truth(0x4000000000000000, 0x4000000000000000, 0x3df49da7e361ce4c),
        truth(0x401e000000000000, 0x401e000000000000, 0x3e09c511dc3a41df),
        Some(truth(
            0x4014000000000000,
            0x4014000000000000,
            0x3e012e0be826d694,
        )),
        truth(0x401a000000000000, 0x401a000000000000, 0x3e065575e0ff4a27),
        Some(truth(
            0x3ff8000000000000,
            0x3ff8000000000000,
            0x3de49da7e361ce4c,
        )),
    ),
    inverse(
        3,
        (0x4024c00000000000, Some(0x4004000000000000)),
        truth(0x4004000000000000, 0x4004000000000000, 0x3df49da7e361ce4c),
        truth(0x4024c00000000000, 0x4024c00000000000, 0x3e11d2f92741e507),
        Some(truth(
            0x401a000000000000,
            0x401a000000000000,
            0x3e065575e0ff4a27,
        )),
        truth(0x4025e00000000000, 0x4025e00000000000, 0x3e12ca5d05ea7ab2),
        Some(truth(
            0x3ff989d89d89d89d,
            0x3ff989d89d89d89e,
            0x3de5eff792c74143,
        )),
    ),
    inverse(
        3,
        (0x402c000000000000, Some(0x4008000000000000)),
        truth(0x4008000000000000, 0x4008000000000000, 0x3df49da7e361ce4c),
        truth(0x402c000000000000, 0x402c000000000000, 0x3e180d43de9cc603),
        Some(truth(
            0x4020000000000000,
            0x4020000000000000,
            0x3e0b7cdfd9d7bdba,
        )),
        truth(0x4031000000000000, 0x4031000000000000, 0x3e1d34add7753996),
        Some(truth(
            0x3ffc000000000000,
            0x3ffc000000000000,
            0x3de80d43de9cc603,
        )),
    ),
];
