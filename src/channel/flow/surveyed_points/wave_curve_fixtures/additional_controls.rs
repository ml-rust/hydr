use super::model::{Case, Expected, ProfileRef, local, relative};

pub(super) const CASES: &[Case] = &[
    relative(
        "rectangle_weak_low_target",
        ProfileRef::Query(1),
        0x3ff0000000000000,
        [0x3ff0000000000000, 0x0000000000000000],
        [0x3af0000000000000, 0x0000000000000000],
        Expected::Shock {
            curve: local(0x3ae6a09e667f3bcc, 0x3ae6a09e667f3bcd, 0x38d36fd255a2213a),
            offset: local(0x3fe6a09e667f3bcc, 0x3fe6a09e667f3bcd, 0x3dd36fd255a2213a),
        },
    ),
    relative(
        "triangle_weak_low_target",
        ProfileRef::Query(4),
        0x3ff0000000000000,
        [0x3ff0000000000000, 0x0000000000000000],
        [0x3af0000000000000, 0x0000000000000000],
        Expected::Shock {
            curve: local(0x3ae47238cf17c239, 0x3ae47238cf17c23a, 0x38d19029920ad502),
            offset: local(0x3fe47238cf17c239, 0x3fe47238cf17c23a, 0x3dd19029920ad502),
        },
    ),
    relative(
        "physical_knot_weak_crossing",
        ProfileRef::Query(2),
        0x3ff0000000000000,
        [0x4008000000000000, 0x0000000000000000],
        [0x3af0000000000000, 0x0000000000000000],
        Expected::Shock {
            curve: local(0x3ad279a74590331c, 0x3ad279a74590331d, 0x38bfbd7b90d4d7c1),
            offset: local(0x3febb67ae8584caa, 0x3febb67ae8584cab, 0x3dd7ce1cac9fa1d1),
        },
    ),
    relative(
        "rectangle_weak_both_low",
        ProfileRef::Query(1),
        0x3ff0000000000000,
        [0x3ff0000000000000, 0x3af0000000000000],
        [0x3af0000000000000, 0x0000000000000000],
        Expected::Shock {
            curve: local(0x3ae6a09e667f3bcc, 0x3ae6a09e667f3bcd, 0x38d36fd255a2213a),
            offset: local(0x3fe6a09e667f3bcc, 0x3fe6a09e667f3bcd, 0x3dd36fd255a2213a),
        },
    ),
    relative(
        "legacy_stationary_compression",
        ProfileRef::OriginalRectangle,
        0x4008000000000000,
        [0x3ff0000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x0000000000000000],
        Expected::Shock {
            curve: local(0x3ff8000000000000, 0x3ff8000000000000, 0x3de49da7e361ce4c),
            offset: local(0x4008000000000000, 0x4008000000000000, 0x3df49da7e361ce4c),
        },
    ),
    relative(
        "legacy_stationary_expansion",
        ProfileRef::OriginalRectangle,
        0x4008000000000000,
        [0x4000000000000000, 0x0000000000000000],
        [0xbff0000000000000, 0x0000000000000000],
        Expected::Rare {
            curve: local(0xbff6f5427f73eb66, 0xbff6f5427f73eb65, 0x3de3b88716a90ae2),
        },
    ),
    relative(
        "rectangle_tiny_source_compression",
        ProfileRef::OriginalRectangle,
        0x3ff0000000000000,
        [0x1a70000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x9a70000000000000],
        Expected::Shock {
            curve: local(0x52a6a09e667f3bcc, 0x52a6a09e667f3bcd, 0x50936fd255a2213a),
            offset: local(0x52a6a09e667f3bcc, 0x52a6a09e667f3bcd, 0x50936fd255a2213a),
        },
    ),
    relative(
        "rectangle_contraction_underflow",
        ProfileRef::LargeRectangle,
        0x6974e718d7d7625a,
        [0x54b249ad2594c37d, 0x0000000000000000],
        [0x0000000000000001, 0x0000000000000000],
        Expected::Shock {
            curve: local(0x07311b0ec57e6499, 0x07311b0ec57e649a, 0x051d6329f1c35ca4),
            offset: local(0x5f138d352e5096af, 0x5f138d352e5096b0, 0x5d00cb70d24b7378),
        },
    ),
    relative(
        "rectangle_wave_range_exceeded",
        ProfileRef::OriginalRectangle,
        0x7fe1ccf385ebc8a0,
        [0x0000000000000001, 0x0000000000000000],
        [0x3ff0000000000000, 0x8000000000000001],
        Expected::NumericalRange,
    ),
];
