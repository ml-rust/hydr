use super::model::{Case, Expected, baseline, local};

pub(super) const CASES: &[Case] = &[
    baseline(
        "first_knot_from_second_source__physical_lower",
        17,
        0,
        Expected::Rare {
            curve: local(0xbfee7af084543f14, 0xbfee7af084543f13, 0x3dda2eac8be32391),
        },
    ),
    baseline(
        "first_knot_from_second_source__physical_upper",
        17,
        1,
        Expected::Rare {
            curve: local(0xbfee7af084543f14, 0xbfee7af084543f13, 0x3dda2eac8be32391),
        },
    ),
    baseline(
        "second_knot_from_first_source__physical_lower",
        18,
        1,
        Expected::Shock {
            curve: local(0x3ff043b5024ac92b, 0x3ff043b5024ac92c, 0x3ddbf131bbcc5ca0),
            offset: local(0x3ffb1b8303d1f9f3, 0x3ffb1b8303d1f9f4, 0x3de748fec72a4d30),
        },
    ),
    baseline(
        "second_knot_from_first_source__physical_upper",
        18,
        0,
        Expected::Shock {
            curve: local(0x3ff043b5024ac92b, 0x3ff043b5024ac92c, 0x3ddbf131bbcc5ca0),
            offset: local(0x3ffb1b8303d1f9f3, 0x3ffb1b8303d1f9f4, 0x3de748fec72a4d30),
        },
    ),
    baseline(
        "rectangle_dry_from_wet_source__physical_lower",
        19,
        0,
        Expected::Rare {
            curve: local(0xbff6a09e667f3bcd, 0xbff6a09e667f3bcc, 0x3de36fd255a2213a),
        },
    ),
    baseline(
        "rectangle_dry_from_wet_source__physical_upper",
        19,
        1,
        Expected::Rare {
            curve: local(0xbff6a09e667f3bcd, 0xbff6a09e667f3bcc, 0x3de36fd255a2213a),
        },
    ),
    baseline(
        "triangle_dry_from_wet_source__physical_lower",
        20,
        0,
        Expected::Rare {
            curve: local(0xc00cea4ebfc356e6, 0xc00cea4ebfc356e5, 0x3df8d688a8a08651),
        },
    ),
    baseline(
        "triangle_dry_from_wet_source__physical_upper",
        20,
        1,
        Expected::Rare {
            curve: local(0xc00cea4ebfc356e6, 0xc00cea4ebfc356e5, 0x3df8d688a8a08651),
        },
    ),
    baseline(
        "decimal_knot_from_wet_source__physical_lower",
        21,
        0,
        Expected::Rare {
            curve: local(0xbff52024fa8f763c, 0xbff52024fa8f763b, 0x3de2258f8a8611c0),
        },
    ),
    baseline(
        "decimal_knot_from_wet_source__physical_upper",
        21,
        1,
        Expected::Rare {
            curve: local(0xbff52024fa8f763c, 0xbff52024fa8f763b, 0x3de2258f8a8611c0),
        },
    ),
];
