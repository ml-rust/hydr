use super::model::{Case, DeltaAreaAuthority, Endpoint, QuantityPolicy, QuantityRef};

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
                delta_pressure: QuantityRef {
                    bounds: [0xbafc000000000000, 0xbafbffffffffffff],
                    native_allowance: Some(0x38f80d43de9cc603),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ffbffffffffffff, 0x3ffc000000000000],
                    native_allowance: Some(0x3de80d43de9cc603),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0xbb00000000000000, 0xbb00000000000000],
                delta_area_allowance: 0x38eb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0xbb0c000000000000, 0xbb0bffffffffffff],
                    native_allowance: Some(0x38f80d43de9cc603),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ffbffffffffffff, 0x3ffc000000000000],
                    native_allowance: Some(0x3de80d43de9cc603),
                    policy: QuantityPolicy::Local,
                },
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
                delta_pressure: QuantityRef {
                    bounds: [0xbb08000000000000, 0xbb07ffffffffffff],
                    native_allowance: Some(0x39049da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x4007ffffffffffff, 0x4008000000000000],
                    native_allowance: Some(0x3df49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0xbb00000000000000, 0xbb00000000000000],
                delta_area_allowance: 0x38eb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0xbb18000000000000, 0xbb17ffffffffffff],
                    native_allowance: Some(0x39049da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x4007ffffffffffff, 0x4008000000000000],
                    native_allowance: Some(0x3df49da7e361ce4c),
                    policy: QuantityPolicy::Local,
                },
            },
        ],
        minimum_endpoint: 1,
        maximum_endpoint: 0,
    },
];
