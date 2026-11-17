use super::model::{Case, DeltaAreaAuthority, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "triangle_exact_dry__physical_anchor",
        query_id: "triangle_exact_dry",
        query_index: 9,
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
        id: "triangle_touching_dry__physical_anchor",
        query_id: "triangle_touching_dry",
        query_index: 10,
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
        id: "triangle_scaled_tiny_positive__physical_anchor",
        query_id: "triangle_scaled_tiny_positive",
        query_index: 15,
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
        id: "triangle_dry_from_wet_source",
        query_id: "triangle_exact_dry",
        query_index: 9,
        source_components: [0x4010000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xc010000000000000, 0xc010000000000000],
                delta_area_allowance: 0x3dfb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0xc010000000000000, 0xc010000000000000],
                delta_area_allowance: 0x3dfb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
];
