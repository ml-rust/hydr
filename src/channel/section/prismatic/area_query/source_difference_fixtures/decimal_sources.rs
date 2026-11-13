use super::model::{Case, DeltaAreaAuthority, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "decimal_knot_signed_offsets__physical_anchor",
        query_id: "decimal_knot_signed_offsets",
        query_index: 6,
        source_components: [0x3feccccccccccccd, 0xbc80000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xbaf0000000000000, 0xbaf0000000000000],
                delta_area_allowance: 0x3ddb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0x3af0000000000000, 0x3af0000000000000],
                delta_area_allowance: 0x3ddb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "decimal_broad__physical_anchor",
        query_id: "decimal_broad",
        query_index: 12,
        source_components: [0x0000000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0x0000000000000000, 0x0000000000000000],
                delta_area_allowance: 0x0000000000000000,
                delta_area_authority: DeltaAreaAuthority::ExactZero,
                structural_equality: true,
            },
            Endpoint {
                delta_area: [0x4010000000000000, 0x4010000000000000],
                delta_area_allowance: 0x3dfb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "decimal_knot_from_wet_source",
        query_id: "decimal_knot_signed_offsets",
        query_index: 6,
        source_components: [0x4010000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xc008cccccccccccd, 0xc008cccccccccccc],
                delta_area_allowance: 0x3df54d93e26d9970,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0xc008cccccccccccd, 0xc008cccccccccccc],
                delta_area_allowance: 0x3df54d93e26d9970,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
];
