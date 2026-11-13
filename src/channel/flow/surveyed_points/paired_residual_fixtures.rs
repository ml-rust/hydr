#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Anchor {
    LeftSource,
    Zero,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Query {
    pub anchor: Anchor,
    pub add: bool,
    pub scale: [u64; 2],
    pub parameter: [u64; 2],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sign {
    Negative,
    Positive,
    ContainsZero,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reference {
    pub bounds: [u64; 2],
    pub allowance: u64,
    pub sign: Sign,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Expected {
    Unavailable,
    ExactZero,
    Point(Reference),
    Range { lower: Reference, upper: Reference },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Partition {
    NotApplicable,
    Local,
    Below,
    Exact,
    Above,
    Crossing,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Request {
    pub id: &'static str,
    pub source_case_id: &'static str,
    pub gravity_bits: u64,
    pub query: Query,
    pub expected: Expected,
    pub partition: Partition,
}
const ONE: u64 = 0x3ff0000000000000;
const fn q(anchor: Anchor, add: bool, scale: [u64; 2], parameter: [u64; 2]) -> Query {
    Query {
        anchor,
        add,
        scale,
        parameter,
    }
}
const fn r(lower: u64, upper: u64, allowance: u64, sign: Sign) -> Reference {
    Reference {
        bounds: [lower, upper],
        allowance,
        sign,
    }
}
const fn request(
    id: &'static str,
    source_case_id: &'static str,
    gravity_bits: u64,
    query: Query,
    expected: Expected,
    partition: Partition,
) -> Request {
    Request {
        id,
        source_case_id,
        gravity_bits,
        query,
        expected,
        partition,
    }
}

const UNIFORM_SOURCE: &str = "scalar_rectangle_uniform_rest";
const STATIONARY_SOURCE: &str = "scalar_rectangle_stationary_shock";
const FILM_SOURCE: &str = "retained_vacuum_inside_film";
const LEFT_DRY_SOURCE: &str = "scalar_rectangle_left_dry_sonic";
const RIGHT_DRY_SOURCE: &str = "scalar_rectangle_right_dry_sonic";

pub(crate) const FILM_SOURCE_INVARIANT: [u64; 2] = [0xbaf0000000000000, 0x0000000000000000];

pub(crate) const REQUESTS: &[Request] = &[
    request(
        "uniform_source_identity",
        UNIFORM_SOURCE,
        ONE,
        q(Anchor::LeftSource, true, [0, 0], [ONE, ONE]),
        Expected::ExactZero,
        Partition::NotApplicable,
    ),
    request(
        "stationary_shock_equality",
        STATIONARY_SOURCE,
        0x4008000000000000,
        q(Anchor::LeftSource, true, [ONE, 0], [ONE, ONE]),
        Expected::Point(r(0, 0, 0x3de49da7e361ce4c, Sign::ContainsZero)),
        Partition::NotApplicable,
    ),
    request(
        "stationary_shock_rarefaction",
        STATIONARY_SOURCE,
        0x4008000000000000,
        q(
            Anchor::LeftSource,
            true,
            [0x3fe0000000000000, 0],
            [ONE, ONE],
        ),
        Expected::Point(r(
            0xbff5da30ff39a3ab,
            0xbff5da30ff39a3aa,
            0x3de2c55fb27e03d8,
            Sign::Negative,
        )),
        Partition::NotApplicable,
    ),
    request(
        "film_dry_target",
        FILM_SOURCE,
        ONE,
        q(Anchor::Zero, true, [0x35f0000000000000, 0], [0, 0]),
        Expected::Point(r(
            0xbaf0000000000000,
            0xbaf0000000000000,
            0x38db7cdfd9d7bdba,
            Sign::Negative,
        )),
        Partition::NotApplicable,
    ),
    request(
        "film_star",
        FILM_SOURCE,
        ONE,
        q(
            Anchor::Zero,
            true,
            [0x35f0000000000000, 0],
            [0x3fb0000000000000, 0x3fb0000000000000],
        ),
        Expected::Point(r(0, 0, 0x38db7cdfd9d7bdba, Sign::ContainsZero)),
        Partition::NotApplicable,
    ),
    request(
        "film_existing_sonic_lower",
        FILM_SOURCE,
        ONE,
        q(
            Anchor::Zero,
            true,
            [0x35f0000000000000, 0],
            [0x3fbc71c71c71c71c, 0x3fbc71c71c71c71c],
        ),
        Expected::Point(r(
            0x3ad5555555555554,
            0x3ad5555555555555,
            0x38db7cdfd9d7bdba,
            Sign::Positive,
        )),
        Partition::NotApplicable,
    ),
    request(
        "scalar_rectangle_left_dry_sonic_unavailable",
        LEFT_DRY_SOURCE,
        ONE,
        q(Anchor::Zero, true, [ONE, 0], [ONE, ONE]),
        Expected::Unavailable,
        Partition::NotApplicable,
    ),
    request(
        "scalar_rectangle_right_dry_sonic_unavailable",
        RIGHT_DRY_SOURCE,
        ONE,
        q(Anchor::Zero, true, [ONE, 0], [ONE, ONE]),
        Expected::Unavailable,
        Partition::NotApplicable,
    ),
    request(
        "uniform_weak_local_rarefactions",
        UNIFORM_SOURCE,
        ONE,
        q(
            Anchor::LeftSource,
            false,
            [0x3af0000000000000, 0],
            [ONE, ONE],
        ),
        Expected::Point(r(
            0xbb00000000000001,
            0xbb00000000000000,
            0x38eb7cdfd9d7bdba,
            Sign::Negative,
        )),
        Partition::Local,
    ),
    request(
        "uniform_half_source_below",
        UNIFORM_SOURCE,
        ONE,
        q(
            Anchor::LeftSource,
            false,
            [0x3fe0000000000000, 0x3c90000000000000],
            [ONE, ONE],
        ),
        Expected::Point(r(
            0xbff2bec333018868,
            0xbff2bec333018867,
            0x3de01a1b086b3900,
            Sign::Negative,
        )),
        Partition::Below,
    ),
    request(
        "uniform_half_source_exact",
        UNIFORM_SOURCE,
        ONE,
        q(
            Anchor::LeftSource,
            false,
            [0x3fe0000000000000, 0],
            [ONE, ONE],
        ),
        Expected::Point(r(
            0xbff2bec333018867,
            0xbff2bec333018866,
            0x3de01a1b086b38ff,
            Sign::Negative,
        )),
        Partition::Exact,
    ),
    request(
        "uniform_half_source_above",
        UNIFORM_SOURCE,
        ONE,
        q(
            Anchor::LeftSource,
            false,
            [0x3fdffffffffffffe, 0],
            [ONE, ONE],
        ),
        Expected::Point(r(
            0xbff2bec333018866,
            0xbff2bec333018865,
            0x3de01a1b086b38fe,
            Sign::Negative,
        )),
        Partition::Above,
    ),
    request(
        "uniform_half_source_crossing",
        UNIFORM_SOURCE,
        ONE,
        q(
            Anchor::Zero,
            true,
            [ONE, 0],
            [0x3fdfffffffffffff, 0x3fe0000000000001],
        ),
        Expected::Range {
            lower: r(
                0xbff2bec333018868,
                0xbff2bec333018867,
                0x3de01a1b086b3900,
                Sign::Negative,
            ),
            upper: r(
                0xbff2bec333018866,
                0xbff2bec333018865,
                0x3de01a1b086b38fe,
                Sign::Negative,
            ),
        },
        Partition::Crossing,
    ),
];
