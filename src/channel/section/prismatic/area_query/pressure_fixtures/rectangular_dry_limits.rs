use super::model::{Case, DeltaAreaAuthority, Endpoint, QuantityPolicy, QuantityRef};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "rectangle_exact_dry__physical_anchor",
        query_id: "rectangle_exact_dry",
        query_index: 7,
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
        id: "rectangle_touching_dry__physical_anchor",
        query_id: "rectangle_touching_dry",
        query_index: 8,
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
                    bounds: [0x0000000000000000, 0x0000000000000001],
                    native_allowance: None,
                    policy: QuantityPolicy::Underflow {
                        physical_projected_bounds: [0x0000000000000000, 0x0000000000000001],
                        maximum_physical_width: 0x0000000000000001,
                        nonzero_physical: true,
                    },
                },
                secant: QuantityRef {
                    bounds: [0x1a50000000000000, 0x1a50000000000000],
                    native_allowance: Some(0x183b7cdfd9d7bdba),
                    policy: QuantityPolicy::Local,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "rectangle_scaled_tiny_positive__physical_anchor",
        query_id: "rectangle_scaled_tiny_positive",
        query_index: 14,
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
                    bounds: [0x0150000000000000, 0x0150000000000000],
                    native_allowance: Some(0x0000006df37f675e),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0x1a70000000000000, 0x1a70000000000000],
                delta_area_allowance: 0x185b7cdfd9d7bdba,
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
                    bounds: [0x1a50000000000000, 0x1a50000000000000],
                    native_allowance: Some(0x183b7cdfd9d7bdba),
                    policy: QuantityPolicy::Local,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
    Case {
        id: "rectangle_dry_from_wet_source",
        query_id: "rectangle_exact_dry",
        query_index: 7,
        source_components: [0x3ff0000000000000, 0x0000000000000000],
        endpoints: [
            Endpoint {
                delta_area: [0xbff0000000000000, 0xbff0000000000000],
                delta_area_allowance: 0x3ddb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0xbfd0000000000000, 0xbfd0000000000000],
                    native_allowance: Some(0x3dbb7cdfd9d7bdba),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3fd0000000000000, 0x3fd0000000000000],
                    native_allowance: Some(0x3dbb7cdfd9d7bdba),
                    policy: QuantityPolicy::Local,
                },
            },
            Endpoint {
                delta_area: [0xbff0000000000000, 0xbff0000000000000],
                delta_area_allowance: 0x3ddb7cdfd9d7bdba,
                delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
                structural_equality: false,
                delta_pressure: QuantityRef {
                    bounds: [0xbfd0000000000000, 0xbfd0000000000000],
                    native_allowance: Some(0x3dbb7cdfd9d7bdba),
                    policy: QuantityPolicy::Local,
                },
                secant: QuantityRef {
                    bounds: [0x3fd0000000000000, 0x3fd0000000000000],
                    native_allowance: Some(0x3dbb7cdfd9d7bdba),
                    policy: QuantityPolicy::Local,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
    },
];
