use super::model::{Case, DeltaAreaAuthority, Endpoint, QuantityPolicy, QuantityRef};

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
                delta_pressure: QuantityRef {
                    bounds: [0x3fc8fe5999576089, 0x3fc8fe599957608a],
                    native_allowance: Some(0x3db578240b39a154),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3fc8fe5999576089, 0x3fc8fe599957608a],
                    native_allowance: Some(0x3db578240b39a154),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0x4028000000000000, 0x4028000000000000],
                delta_area_allowance: 0x3e149da7e361ce4c,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0x402b2c06f1945d48, 0x402b2c06f1945d49],
                    native_allowance: Some(0x3e17572e8a87711f),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ff21d59f662e8da, 0x3ff21d59f662e8db],
                    native_allowance: Some(0x3ddf1ee8b8b496d5),
                    policy: QuantityPolicy::Local,
                },
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
                delta_pressure: QuantityRef {
                    bounds: [0xbffba7eb37cc6774, 0xbffba7eb37cc6773],
                    native_allowance: Some(0x3de7c19a9d010372),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ffba7eb37cc6773, 0x3ffba7eb37cc6774],
                    native_allowance: Some(0x3de7c19a9d010372),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0xc02a000000000000, 0xc02a000000000000],
                delta_area_allowance: 0x3e165575e0ff4a27,
                delta_area_authority: DeltaAreaAuthority::OriginalGeometry,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0xc030ce034ccd513f, 0xc030ce034ccd513e],
                    native_allowance: Some(0x3e1cdecd47485311),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ff4aec8fc1063fe, 0x3ff4aec8fc1063ff],
                    native_allowance: Some(0x3de1c42f8e53e459),
                    policy: QuantityPolicy::Local,
                },
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
                delta_pressure: QuantityRef {
                    bounds: [0xc014aaaaaaaaaaab, 0xc014aaaaaaaaaaaa],
                    native_allowance: Some(0x3e07315cdfce0815),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ff25ed097b425ed, 0x3ff25ed097b425ee],
                    native_allowance: Some(0x3ddf8f5fd4309777),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0xc012000000000000, 0xc012000000000000],
                delta_area_allowance: 0x3dfeec7bd512b572,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0xc014aaaaaaaaaaab, 0xc014aaaaaaaaaaaa],
                    native_allowance: Some(0x3e07315cdfce0815),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ff25ed097b425ed, 0x3ff25ed097b425ee],
                    native_allowance: Some(0x3ddf8f5fd4309777),
                    policy: QuantityPolicy::Local,
                },
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
                delta_pressure: QuantityRef {
                    bounds: [0x4014aaaaaaaaaaaa, 0x4014aaaaaaaaaaab],
                    native_allowance: Some(0x3df7315cdfce0815),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ff25ed097b425ed, 0x3ff25ed097b425ee],
                    native_allowance: Some(0x3ddf8f5fd4309777),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0x4012000000000000, 0x4012000000000000],
                delta_area_allowance: 0x3dfeec7bd512b572,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0x4014aaaaaaaaaaaa, 0x4014aaaaaaaaaaab],
                    native_allowance: Some(0x3df7315cdfce0815),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3ff25ed097b425ed, 0x3ff25ed097b425ee],
                    native_allowance: Some(0x3ddf8f5fd4309777),
                    policy: QuantityPolicy::Local,
                },
            },
        ],
        minimum_endpoint: 1,
        maximum_endpoint: 0,
    },
];
