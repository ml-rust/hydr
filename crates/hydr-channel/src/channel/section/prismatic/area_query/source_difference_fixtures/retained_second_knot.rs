use super::model::{Case, DeltaAreaAuthority, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "source_second_knot_reverse__physical_anchor",
        query_id: "source_second_knot_reverse",
        query_index: 1,
        source_components: [0x401e000000000000, 0x3af0000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0x0000000000000000, 0x0000000000000000],
                delta_area_allowance: 0x0000000000000000,
                delta_area_authority: DeltaAreaAuthority::ExactZero,
                structural_equality: true,
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
        id: "middle_band_end_signed_offsets__physical_anchor",
        query_id: "middle_band_end_signed_offsets",
        query_index: 5,
        source_components: [0x401e000000000000, 0x0000000000000000],
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
