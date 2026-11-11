use super::model::{GeometryRejection, ProfileCase, Quantity, QueryRejection, UnresolvedInverse};

pub(crate) const QUERY_REJECTIONS: [QueryRejection; 90] = [
    QueryRejection {
        profile: 0,
        quantity: Quantity::Area,
        bits: 0x4018000000000001,
    },
    QueryRejection {
        profile: 0,
        quantity: Quantity::Depth,
        bits: 0x4008000000000001,
    },
    QueryRejection {
        profile: 0,
        quantity: Quantity::Depth,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 0,
        quantity: Quantity::Area,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 0,
        quantity: Quantity::Depth,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 0,
        quantity: Quantity::Area,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 0,
        quantity: Quantity::Depth,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 0,
        quantity: Quantity::Area,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 0,
        quantity: Quantity::Depth,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 0,
        quantity: Quantity::Area,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Area,
        bits: 0x402b000000000001,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Depth,
        bits: 0x4008000000000001,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Depth,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Area,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Depth,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Area,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Depth,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Area,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Depth,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 1,
        quantity: Quantity::Area,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Area,
        bits: 0x4033800000000001,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Depth,
        bits: 0x4008000000000001,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Depth,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Area,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Depth,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Area,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Depth,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Area,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Depth,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 2,
        quantity: Quantity::Area,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Area,
        bits: 0x402c000000000001,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Depth,
        bits: 0x4008000000000001,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Depth,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Area,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Depth,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Area,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Depth,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Area,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Depth,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 3,
        quantity: Quantity::Area,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Area,
        bits: 0x4018000000000001,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Depth,
        bits: 0x4008000000000001,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Depth,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Area,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Depth,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Area,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Depth,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Area,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Depth,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 4,
        quantity: Quantity::Area,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Area,
        bits: 0x4033800000000001,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Depth,
        bits: 0x4008000000000001,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Depth,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Area,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Depth,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Area,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Depth,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Area,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Depth,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 5,
        quantity: Quantity::Area,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Area,
        bits: 0x402c000000000001,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Depth,
        bits: 0x4008000000000001,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Depth,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Area,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Depth,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Area,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Depth,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Area,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Depth,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 6,
        quantity: Quantity::Area,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Area,
        bits: 0x4000000000000001,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Depth,
        bits: 0x3ff0000000000001,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Depth,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Area,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Depth,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Area,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Depth,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Area,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Depth,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 7,
        quantity: Quantity::Area,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Area,
        bits: 0x4018000000000001,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Depth,
        bits: 0x4008000000000001,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Depth,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Area,
        bits: 0xbff0000000000000,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Depth,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Area,
        bits: 0x7ff0000000000000,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Depth,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Area,
        bits: 0xfff0000000000000,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Depth,
        bits: 0x7ff8000000000001,
    },
    QueryRejection {
        profile: 8,
        quantity: Quantity::Area,
        bits: 0x7ff8000000000001,
    },
];

pub(crate) const ADMISSION_REJECTIONS: [GeometryRejection; 6] = [
    GeometryRejection {
        profile: ProfileCase {
            id: "positive_height_bench",
            points: &[
                [0xc010000000000000, 0x4008000000000000],
                [0xc000000000000000, 0x3ff0000000000000],
                [0xbff0000000000000, 0x3ff0000000000000],
                [0xbff0000000000000, 0x0000000000000000],
                [0x3ff0000000000000, 0x0000000000000000],
                [0x3ff0000000000000, 0x3ff0000000000000],
                [0x4000000000000000, 0x3ff0000000000000],
                [0x4010000000000000, 0x4008000000000000],
            ],
            maximum_stage_bits: 0x4008000000000000,
        },
        reason: "Positive-height horizontal bench.",
    },
    GeometryRejection {
        profile: ProfileCase {
            id: "disconnected_depressions",
            points: &[
                [0xc010000000000000, 0x4008000000000000],
                [0xc008000000000000, 0x0000000000000000],
                [0xc000000000000000, 0x4000000000000000],
                [0xbff0000000000000, 0x0000000000000000],
                [0x0000000000000000, 0x4008000000000000],
            ],
            maximum_stage_bits: 0x4008000000000000,
        },
        reason: "Internal ridge separates wet pockets below height two.",
    },
    GeometryRejection {
        profile: ProfileCase {
            id: "hidden_pockets_above_cap_ridge",
            points: &[
                [0xc014000000000000, 0x4008000000000000],
                [0xc010000000000000, 0x0000000000000000],
                [0xc008000000000000, 0x4010000000000000],
                [0xc000000000000000, 0x0000000000000000],
                [0xbff0000000000000, 0x4008000000000000],
            ],
            maximum_stage_bits: 0x4008000000000000,
        },
        reason: "Above-cap ridge separates two wetted depressions throughout the admitted range.",
    },
    GeometryRejection {
        profile: ProfileCase {
            id: "vertical_retracing",
            points: &[
                [0xbff0000000000000, 0x4008000000000000],
                [0xbff0000000000000, 0x0000000000000000],
                [0xbff0000000000000, 0x3ff0000000000000],
                [0xbff0000000000000, 0x0000000000000000],
                [0x3ff0000000000000, 0x0000000000000000],
                [0x3ff0000000000000, 0x4008000000000000],
            ],
            maximum_stage_bits: 0x4008000000000000,
        },
        reason: "Vertical path retraces an existing wall.",
    },
    GeometryRejection {
        profile: ProfileCase {
            id: "unordered_stations",
            points: &[
                [0xbff0000000000000, 0x4008000000000000],
                [0x0000000000000000, 0x0000000000000000],
                [0xbfe0000000000000, 0x3ff0000000000000],
                [0x3ff0000000000000, 0x4008000000000000],
            ],
            maximum_stage_bits: 0x4008000000000000,
        },
        reason: "Horizontal station order decreases.",
    },
    GeometryRejection {
        profile: ProfileCase {
            id: "nonconvex_pressure",
            points: &[
                [0xc01c000000000000, 0x4000000000000000],
                [0xc000000000000000, 0x3ff0000000000000],
                [0xbff0000000000000, 0x0000000000000000],
                [0x3ff0000000000000, 0x0000000000000000],
                [0x4000000000000000, 0x3ff0000000000000],
                [0x401c000000000000, 0x4000000000000000],
            ],
            maximum_stage_bits: 0x4000000000000000,
        },
        reason: "Width slope jumps from two to ten at height one. T squared minus A times slope equals minus fourteen.",
    },
];

pub(crate) const UNRESOLVED_INVERSE: UnresolvedInverse = UnresolvedInverse {
    profile: ProfileCase {
        id: "unrepresented_positive_depth",
        points: &[
            [0xbff0000000000000, 0x3ff0000000000000],
            [0xbff0000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x0000000000000000],
            [0x3ff0000000000000, 0x3ff0000000000000],
        ],
        maximum_stage_bits: 0x3ff0000000000000,
    },
    area_bits: 0x0000000000000001,
};
