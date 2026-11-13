use super::model::{Case, DeltaAreaAuthority, Endpoint, QuantityPolicy, QuantityRef};

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
                delta_pressure: QuantityRef {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: Some(0x0000000000000000),
                    policy: QuantityPolicy::ExactZero,
                },
                secant: QuantityRef {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: Some(0x0000000000000000),
                    policy: QuantityPolicy::ExactZero,
                },
            },
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
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: Some(0x0000000000000000),
                    policy: QuantityPolicy::ExactZero,
                },
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
                delta_pressure: QuantityRef {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: Some(0x0000000000000000),
                    policy: QuantityPolicy::ExactZero,
                },
                secant: QuantityRef {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: Some(0x0000000000000000),
                    policy: QuantityPolicy::ExactZero,
                },
            },
            Endpoint {
                delta_area: [0x1a70000000000000, 0x1a70000000000000],
                delta_area_allowance: 0x185b7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0x07916b28f55d72d3, 0x07916b28f55d72d4],
                    native_allowance: Some(0x057decc74e2e5bc9),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x2d116b28f55d72d3, 0x2d116b28f55d72d4],
                    native_allowance: Some(0x2afdecc74e2e5bc9),
                    policy: QuantityPolicy::Local,
                },
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
                delta_pressure: QuantityRef {
                    bounds: [0x0000000000000000, 0x0000000000000001],
                    native_allowance: None,
                    policy: QuantityPolicy::Underflow {
                        physical_projected_bounds: [0x0000000000000000, 0x0000000000000001],
                        maximum_physical_width: 0x0000000000000001,
                        nonzero_physical: true,
                    },
                },
                secant: QuantityRef {
                    bounds: [0x20916b28f55d72d3, 0x20916b28f55d72d4],
                    native_allowance: Some(0x1e7decc74e2e5bc9),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0x1a70000000000000, 0x1a70000000000000],
                delta_area_allowance: 0x185b7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0x07916b28f55d72d3, 0x07916b28f55d72d4],
                    native_allowance: Some(0x057decc74e2e5bc9),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x2d116b28f55d72d3, 0x2d116b28f55d72d4],
                    native_allowance: Some(0x2afdecc74e2e5bc9),
                    policy: QuantityPolicy::Local,
                },
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
                delta_pressure: QuantityRef {
                    bounds: [0xc0016b28f55d72d4, 0xc0016b28f55d72d3],
                    native_allowance: Some(0x3dedecc74e2e5bc9),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3fe16b28f55d72d3, 0x3fe16b28f55d72d4],
                    native_allowance: Some(0x3dcdecc74e2e5bc9),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0xc010000000000000, 0xc010000000000000],
                delta_area_allowance: 0x3dfb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0xc0016b28f55d72d4, 0xc0016b28f55d72d3],
                    native_allowance: Some(0x3dedecc74e2e5bc9),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3fe16b28f55d72d3, 0x3fe16b28f55d72d4],
                    native_allowance: Some(0x3dcdecc74e2e5bc9),
                    policy: QuantityPolicy::Local,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
];
