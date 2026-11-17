use super::model::{Case, DeltaAreaAuthority, Endpoint, QuantityPolicy, QuantityRef};

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
                delta_pressure: QuantityRef {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: Some(0x0000000000000000),
                    policy: QuantityPolicy::ExactZero,
                },
                secant: QuantityRef {
                    bounds: [0x3fe7ffffffffffff, 0x3fe8000000000000],
                    native_allowance: Some(0x3dd49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0x3b00000000000000, 0x3b00000000000000],
                delta_area_allowance: 0x38eb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0x3af8000000000000, 0x3af8000000000001],
                    native_allowance: Some(0x38e49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3fe8000000000000, 0x3fe8000000000001],
                    native_allowance: Some(0x3dd49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
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
                delta_pressure: QuantityRef {
                    bounds: [0xbae8000000000000, 0xbae7ffffffffffff],
                    native_allowance: Some(0x38d49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3fe7ffffffffffff, 0x3fe8000000000000],
                    native_allowance: Some(0x3dd49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0x3af0000000000000, 0x3af0000000000000],
                delta_area_allowance: 0x38db7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0x3ae8000000000000, 0x3ae8000000000001],
                    native_allowance: Some(0x38d49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3fe8000000000000, 0x3fe8000000000001],
                    native_allowance: Some(0x3dd49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
];
