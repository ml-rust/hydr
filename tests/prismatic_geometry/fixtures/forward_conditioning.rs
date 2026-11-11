use super::model::{ForwardCase, forward, truth};

// Exact clipped-segment references use the original F64 coordinates.
pub(super) const CASES: [ForwardCase; 10] = [
    forward(
        9,
        0x3fe0000000000000,
        truth(0x3fdccccccccccccc, 0x3fdccccccccccccd, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x3feccccccccccccc,
            0x3feccccccccccccd,
            0x3ddb7cdfd9d7bdba,
        )),
        truth(0x3fbccccccccccccc, 0x3fbccccccccccccd, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x3fe0000000000000,
            0x3fe0000000000000,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    forward(
        9,
        0x3ff0000000000000,
        truth(0x3feccccccccccccc, 0x3feccccccccccccd, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x3feccccccccccccc,
            0x3feccccccccccccd,
            0x3ddb7cdfd9d7bdba,
        )),
        truth(0x3fdccccccccccccc, 0x3fdccccccccccccd, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x3ff0000000000000,
            0x3ff0000000000000,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    forward(
        9,
        0x4008000000000000,
        truth(0x4005999999999999, 0x400599999999999a, 0x3df28de3e63e6cde),
        Some(truth(
            0x3feccccccccccccc,
            0x3feccccccccccccd,
            0x3ddb7cdfd9d7bdba,
        )),
        truth(0x4010333333333333, 0x4010333333333334, 0x3dfbd4d5d95da34d),
        Some(truth(
            0x4008000000000000,
            0x4008000000000000,
            0x3df49da7e361ce4c,
        )),
    ),
    forward(
        10,
        0x0000000000000000,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
    ),
    forward(
        10,
        0x8000000000000000,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
    ),
    forward(
        10,
        0x3feccccccccccccc,
        truth(0x4005999999999998, 0x4005999999999999, 0x3df28de3e63e6cdd),
        Some(truth(
            0x400fffffffffffff,
            0x4010000000000000,
            0x3dfb7cdfd9d7bdba,
        )),
        truth(0x3ff147ae147ae146, 0x3ff147ae147ae147, 0x3dddafd309fd7afb),
        Some(truth(
            0x3fe5999999999999,
            0x3fe599999999999a,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    forward(
        10,
        0x3feccccccccccccd,
        truth(0x4005999999999999, 0x400599999999999a, 0x3df28de3e63e6cde),
        Some(truth(
            0x4010000000000000,
            0x4010000000000001,
            0x3dfb7cdfd9d7bdba,
        )),
        truth(0x3ff147ae147ae147, 0x3ff147ae147ae148, 0x3dddafd309fd7afd),
        Some(truth(
            0x3fe5999999999999,
            0x3fe599999999999a,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    forward(
        10,
        0x3fecccccccccccce,
        truth(0x400599999999999a, 0x400599999999999b, 0x3df28de3e63e6cdf),
        Some(truth(
            0x4010000000000000,
            0x4010000000000001,
            0x3dfb7cdfd9d7bdba,
        )),
        truth(0x3ff147ae147ae149, 0x3ff147ae147ae14a, 0x3dddafd309fd7aff),
        Some(truth(
            0x3fe599999999999a,
            0x3fe599999999999b,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    forward(
        10,
        0x3ff8000000000000,
        truth(0x40151eb851eb851e, 0x40151eb851eb851f, 0x3e0224564d375961),
        Some(truth(
            0x4012666666666666,
            0x4012666666666667,
            0x3dff9c67d41e8096,
        )),
        truth(0x400ba5e353f7ced9, 0x400ba5e353f7ceda, 0x3df7bfdc07fdfbfd),
        Some(truth(
            0x3ff25d7ee30f9525,
            0x3ff25d7ee30f9526,
            0x3ddf8d1ba7ae2c24,
        )),
    ),
    forward(
        10,
        0x4004000000000000,
        truth(0x4024c28f5c28f5c2, 0x4024c28f5c28f5c3, 0x3e11d52c1a720ac4),
        Some(truth(
            0x4016666666666666,
            0x4016666666666667,
            0x3e033dcfe54a3802,
        )),
        truth(0x402667c3ece2a534, 0x402667c3ece2a535, 0x3e133efc22b93b12),
        Some(truth(
            0x3ffda83a83a83a83,
            0x3ffda83a83a83a84,
            0x3de979acb810a1aa,
        )),
    ),
];
