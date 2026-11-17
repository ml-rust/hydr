use super::super::model::{Reference, ReferencePolicy};
use super::model::{Case, Endpoint};

pub(crate) const CASES: &[Case] = &[Case {
    id: "trapezoid_source_retained_offsets__physical_anchor",
    query_id: "trapezoid_source_retained_offsets",
    query_index: 16,
    source_components: [0x3ff0000000000000, 0xbaf0000000000000],
    source_potential: Reference {
        bounds: [0x3ff4b74966b07ce2, 0x3ff4b74966b07ce3],
        native_allowance: 0x3de1cb7d1560ecea,
        policy: ReferencePolicy::PhysicalLocal,
    },
    source_sound: Reference {
        bounds: [0x3fe1feb33c1c381e, 0x3fe1feb33c1c381f],
        native_allowance: 0x3dceea40256491f9,
        policy: ReferencePolicy::PhysicalLocal,
    },
    endpoints: [
        Endpoint {
            structural_equality: true,
            potential: Reference {
                bounds: [0x3ff4b74966b07ce2, 0x3ff4b74966b07ce3],
                native_allowance: 0x3de1cb7d1560ecea,
                policy: ReferencePolicy::PhysicalLocal,
            },
            sound: Reference {
                bounds: [0x3fe1feb33c1c381e, 0x3fe1feb33c1c381f],
                native_allowance: 0x3dceea40256491f9,
                policy: ReferencePolicy::PhysicalLocal,
            },
            increment: Reference {
                bounds: [0x0000000000000000, 0x0000000000000000],
                native_allowance: 0x0000000000000000,
                policy: ReferencePolicy::ExactZero,
            },
        },
        Endpoint {
            structural_equality: false,
            potential: Reference {
                bounds: [0x3ff4b74966b07ce2, 0x3ff4b74966b07ce3],
                native_allowance: 0x3de1cb7d1560ecea,
                policy: ReferencePolicy::PhysicalLocal,
            },
            sound: Reference {
                bounds: [0x3fe1feb33c1c381e, 0x3fe1feb33c1c381f],
                native_allowance: 0x3dceea40256491f9,
                policy: ReferencePolicy::PhysicalLocal,
            },
            increment: Reference {
                bounds: [0x3af1feb33c1c381e, 0x3af1feb33c1c381f],
                native_allowance: 0x38deea40256491f9,
                policy: ReferencePolicy::PhysicalLocal,
            },
        },
    ],
    minimum_endpoint: 0,
    maximum_endpoint: 1,
    wet_possible: true,
    dry_possible: false,
}];
