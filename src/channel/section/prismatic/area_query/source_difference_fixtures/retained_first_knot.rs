use super::model::{Case, DeltaAreaAuthority, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "source_first_knot_forward__physical_anchor",
        query_id: "source_first_knot_forward",
        query_index: 0,
        source_components: [0x4008000000000000, 0xbaf0000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0x0000000000000000, 0x0000000000000000],
                delta_area_allowance: 0x0000000000000000,
                delta_area_authority: DeltaAreaAuthority::ExactZero,
                structural_equality: true,
            },
            Endpoint {
                delta_area: [0x3b00000000000000, 0x3b00000000000000],
                delta_area_allowance: 0x38eb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "first_band_start_signed_offsets__physical_anchor",
        query_id: "first_band_start_signed_offsets",
        query_index: 4,
        source_components: [0x4008000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xbaf0000000000000, 0xbaf0000000000000],
                delta_area_allowance: 0x38db7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
            Endpoint {
                delta_area: [0x3af0000000000000, 0x3af0000000000000],
                delta_area_allowance: 0x38db7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
];
