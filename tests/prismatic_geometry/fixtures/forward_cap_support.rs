use super::model::{ForwardCase, forward, truth};

// Exact clipped-segment references use the original F64 coordinates.
pub(super) const CASES: [ForwardCase; 12] = [
    forward(
        7,
        0x0000000000000000,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
    ),
    forward(
        7,
        0x8000000000000000,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
    ),
    forward(
        7,
        0x3fe0000000000000,
        truth(0x3ff0000000000000, 0x3ff0000000000000, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x4000000000000000,
            0x4000000000000000,
            0x3deb7cdfd9d7bdba,
        )),
        truth(0x3fd0000000000000, 0x3fd0000000000000, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x3fe0000000000000,
            0x3fe0000000000000,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    forward(
        7,
        0x3ff0000000000000,
        truth(0x4000000000000000, 0x4000000000000000, 0x3deb7cdfd9d7bdba),
        Some(truth(
            0x4000000000000000,
            0x4000000000000000,
            0x3deb7cdfd9d7bdba,
        )),
        truth(0x3ff0000000000000, 0x3ff0000000000000, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x3ff0000000000000,
            0x3ff0000000000000,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    forward(
        8,
        0x0000000000000000,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
    ),
    forward(
        8,
        0x8000000000000000,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
        truth(0x0000000000000000, 0x0000000000000000, 0x3ddb7cdfd9d7bdba),
        None,
    ),
    forward(
        8,
        0x3fe0000000000000,
        truth(0x3ff0000000000000, 0x3ff0000000000000, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x4000000000000000,
            0x4000000000000000,
            0x3deb7cdfd9d7bdba,
        )),
        truth(0x3fd0000000000000, 0x3fd0000000000000, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x3fe0000000000000,
            0x3fe0000000000000,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    forward(
        8,
        0x3ff0000000000000,
        truth(0x4000000000000000, 0x4000000000000000, 0x3deb7cdfd9d7bdba),
        Some(truth(
            0x4000000000000000,
            0x4000000000000000,
            0x3deb7cdfd9d7bdba,
        )),
        truth(0x3ff0000000000000, 0x3ff0000000000000, 0x3ddb7cdfd9d7bdba),
        Some(truth(
            0x3ff0000000000000,
            0x3ff0000000000000,
            0x3ddb7cdfd9d7bdba,
        )),
    ),
    forward(
        8,
        0x3ff8000000000000,
        truth(0x4008000000000000, 0x4008000000000000, 0x3df49da7e361ce4c),
        Some(truth(
            0x4000000000000000,
            0x4000000000000000,
            0x3deb7cdfd9d7bdba,
        )),
        truth(0x4002000000000000, 0x4002000000000000, 0x3deeec7bd512b572),
        Some(truth(
            0x3ff8000000000000,
            0x3ff8000000000000,
            0x3de49da7e361ce4c,
        )),
    ),
    forward(
        8,
        0x4000000000000000,
        truth(0x4010000000000000, 0x4010000000000000, 0x3dfb7cdfd9d7bdba),
        Some(truth(
            0x4000000000000000,
            0x4000000000000000,
            0x3deb7cdfd9d7bdba,
        )),
        truth(0x4010000000000000, 0x4010000000000000, 0x3dfb7cdfd9d7bdba),
        Some(truth(
            0x4000000000000000,
            0x4000000000000000,
            0x3deb7cdfd9d7bdba,
        )),
    ),
    forward(
        8,
        0x4004000000000000,
        truth(0x4014000000000000, 0x4014000000000000, 0x3e012e0be826d694),
        Some(truth(
            0x4000000000000000,
            0x4000000000000000,
            0x3deb7cdfd9d7bdba,
        )),
        truth(0x4019000000000000, 0x4019000000000000, 0x3e05798ee2308c39),
        Some(truth(
            0x4004000000000000,
            0x4004000000000000,
            0x3df12e0be826d694,
        )),
    ),
    forward(
        8,
        0x4008000000000000,
        truth(0x4018000000000000, 0x4018000000000000, 0x3e049da7e361ce4c),
        Some(truth(
            0x4000000000000000,
            0x4000000000000000,
            0x3deb7cdfd9d7bdba,
        )),
        truth(0x4022000000000000, 0x4022000000000000, 0x3e0eec7bd512b572),
        Some(truth(
            0x4008000000000000,
            0x4008000000000000,
            0x3df49da7e361ce4c,
        )),
    ),
];
