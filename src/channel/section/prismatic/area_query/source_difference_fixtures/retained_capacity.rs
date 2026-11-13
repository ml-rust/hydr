use super::model::{Case, DeltaAreaAuthority, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "three_band_cap_deficit__physical_anchor",
        query_id: "three_band_cap_deficit",
        query_index: 2,
        source_components: [0x402c000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xbaf0000000000000, 0xbaf0000000000000],
                delta_area_allowance: 0x38db7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0xbb00000000000000, 0xbb00000000000000],
                delta_area_allowance: 0x38eb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 1,
        maximum_endpoint: 0,
    },
    Case {
        id: "rectangle_cap_deficit__physical_anchor",
        query_id: "rectangle_cap_deficit",
        query_index: 3,
        source_components: [0x4018000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xbaf0000000000000, 0xbaf0000000000000],
                delta_area_allowance: 0x38db7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0xbb00000000000000, 0xbb00000000000000],
                delta_area_allowance: 0x38eb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 1,
        maximum_endpoint: 0,
    },
];
