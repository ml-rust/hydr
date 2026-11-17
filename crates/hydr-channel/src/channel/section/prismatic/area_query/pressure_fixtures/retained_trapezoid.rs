use super::model::{Case, DeltaAreaAuthority, Endpoint, QuantityPolicy, QuantityRef};

pub(crate) const CASES: &[Case] = &[Case {
    id: "trapezoid_source_retained_offsets__physical_anchor",
    query_id: "trapezoid_source_retained_offsets",
    query_index: 16,
    source_components: [0x3ff0000000000000, 0xbaf0000000000000],
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
                bounds: [0x3fd43d136248490e, 0x3fd43d136248490f],
                native_allowance: Some(0x3dc16282996f8872),
                policy: QuantityPolicy::Local,
            },
        },
        Endpoint {
            delta_area: [0x3b00000000000000, 0x3b00000000000000],
            delta_area_allowance: 0x38eb7cdfd9d7bdba,
            delta_area_authority: DeltaAreaAuthority::ConditionedLocal,
            structural_equality: false,
            delta_pressure: QuantityRef {
                bounds: [0x3ae43d136248490e, 0x3ae43d136248490f],
                native_allowance: Some(0x38d16282996f8872),
                policy: QuantityPolicy::Local,
            },
            secant: QuantityRef {
                bounds: [0x3fd43d136248490e, 0x3fd43d136248490f],
                native_allowance: Some(0x3dc16282996f8872),
                policy: QuantityPolicy::Local,
            },
        },
    ],
    minimum_endpoint: 0,
    maximum_endpoint: 1,
}];
