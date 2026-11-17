use super::model::{Case, DeltaAreaAuthority, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "three_band_broad__physical_anchor",
        query_id: "three_band_broad",
        query_index: 11,
        source_components: [0x0000000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0x3ff0000000000000, 0x3ff0000000000000],
                delta_area_allowance: 0x3ddb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0x4028000000000000, 0x4028000000000000],
                delta_area_allowance: 0x3e149da7e361ce4c,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "three_band_broad_reverse__physical_anchor",
        query_id: "three_band_broad_reverse",
        query_index: 13,
        source_components: [0x402c000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xbff0000000000000, 0xbff0000000000000],
                delta_area_allowance: 0x3ddb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0xc02a000000000000, 0xc02a000000000000],
                delta_area_allowance: 0x3e165575e0ff4a27,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 1,
        maximum_endpoint: 0,
    },
    Case {
        id: "first_knot_from_second_source",
        query_id: "source_first_knot_forward",
        query_index: 0,
        source_components: [0x401e000000000000, 0x3af0000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xc012000000000001, 0xc012000000000000],
                delta_area_allowance: 0x3dfeec7bd512b572,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0xc012000000000000, 0xc012000000000000],
                delta_area_allowance: 0x3dfeec7bd512b572,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "second_knot_from_first_source",
        query_id: "source_second_knot_reverse",
        query_index: 1,
        source_components: [0x4008000000000000, 0xbaf0000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0x4012000000000000, 0x4012000000000001],
                delta_area_allowance: 0x3dfeec7bd512b572,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0x4012000000000000, 0x4012000000000000],
                delta_area_allowance: 0x3dfeec7bd512b572,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 1,
        maximum_endpoint: 0,
    },
];
