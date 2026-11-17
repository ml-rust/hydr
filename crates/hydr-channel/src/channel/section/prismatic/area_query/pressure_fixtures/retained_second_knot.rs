use super::model::{Case, DeltaAreaAuthority, Endpoint, QuantityPolicy, QuantityRef};

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
                delta_pressure: QuantityRef {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: Some(0x0000000000000000),
                    policy: QuantityPolicy::ExactZero,
                },
                secant: QuantityRef {
                    bounds: [0x3ff8000000000000, 0x3ff8000000000001],
                    native_allowance: Some(0x3de49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0xbb00000000000000, 0xbb00000000000000],
                delta_area_allowance: 0x38eb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0xbb08000000000000, 0xbb07ffffffffffff],
                    native_allowance: Some(0x38f49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ff7ffffffffffff, 0x3ff8000000000000],
                    native_allowance: Some(0x3de49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
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
                delta_pressure: QuantityRef {
                    bounds: [0xbaf8000000000000, 0xbaf7ffffffffffff],
                    native_allowance: Some(0x38e49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ff7ffffffffffff, 0x3ff8000000000000],
                    native_allowance: Some(0x3de49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0x3af0000000000000, 0x3af0000000000000],
                delta_area_allowance: 0x38db7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0x3af8000000000000, 0x3af8000000000001],
                    native_allowance: Some(0x38e49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ff8000000000000, 0x3ff8000000000001],
                    native_allowance: Some(0x3de49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
];
