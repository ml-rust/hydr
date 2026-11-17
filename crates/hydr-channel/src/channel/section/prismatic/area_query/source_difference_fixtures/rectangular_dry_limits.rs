use super::model::{Case, DeltaAreaAuthority, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "rectangle_exact_dry__physical_anchor",
        query_id: "rectangle_exact_dry",
        query_index: 7,
        source_components: [0x0000000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0x0000000000000000, 0x0000000000000000],
                delta_area_allowance: 0x0000000000000000,
                delta_area_authority: DeltaAreaAuthority::ExactZero,
                structural_equality: true,
            },
            Endpoint {
                delta_area: [0x0000000000000000, 0x0000000000000000],
                delta_area_allowance: 0x0000000000000000,
                delta_area_authority: DeltaAreaAuthority::ExactZero,
                structural_equality: true,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "rectangle_touching_dry__physical_anchor",
        query_id: "rectangle_touching_dry",
        query_index: 8,
        source_components: [0x0000000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0x0000000000000000, 0x0000000000000000],
                delta_area_allowance: 0x0000000000000000,
                delta_area_authority: DeltaAreaAuthority::ExactZero,
                structural_equality: true,
            },
            Endpoint {
                delta_area: [0x1a70000000000000, 0x1a70000000000000],
                delta_area_allowance: 0x185b7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "rectangle_scaled_tiny_positive__physical_anchor",
        query_id: "rectangle_scaled_tiny_positive",
        query_index: 14,
        source_components: [0x0000000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0x0170000000000000, 0x0170000000000000],
                delta_area_allowance: 0x000001b7cdfd9d7b,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0x1a70000000000000, 0x1a70000000000000],
                delta_area_allowance: 0x185b7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "rectangle_dry_from_wet_source",
        query_id: "rectangle_exact_dry",
        query_index: 7,
        source_components: [0x3ff0000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xbff0000000000000, 0xbff0000000000000],
                delta_area_allowance: 0x3ddb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0xbff0000000000000, 0xbff0000000000000],
                delta_area_allowance: 0x3ddb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
];
