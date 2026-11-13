use super::super::model::{Reference, ReferencePolicy};
use super::model::{Case, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "three_band_cap_deficit__physical_anchor",
        query_id: "three_band_cap_deficit",
        query_index: 2,
        source_components: [0x402c000000000000, 0x0000000000000000],
        source_potential: Reference {
            bounds: [0x400eed016f605088, 0x400eed016f605089],
            native_allowance: 0x3dfa90a7f1b28acb,
            policy: ReferencePolicy::PhysicalLocal,
        },
        source_sound: Reference {
            bounds: [0x3ff52a7fa9d2f8e9, 0x3ff52a7fa9d2f8ea],
            native_allowance: 0x3de22e74763c9fbf,
            policy: ReferencePolicy::PhysicalLocal,
        },
        endpoints: [
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x400eed016f605088, 0x400eed016f605089],
                    native_allowance: 0x3dfa90a7f1b28acb,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3ff52a7fa9d2f8e9, 0x3ff52a7fa9d2f8ea],
                    native_allowance: 0x3de22e74763c9fbf,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbab83091e6a7f7e7, 0xbab83091e6a7f7e6],
                    native_allowance: 0x38a4c7608720b692,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x400eed016f605088, 0x400eed016f605089],
                    native_allowance: 0x3dfa90a7f1b28acb,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3ff52a7fa9d2f8e9, 0x3ff52a7fa9d2f8ea],
                    native_allowance: 0x3de22e74763c9fbf,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbac83091e6a7f7e7, 0xbac83091e6a7f7e6],
                    native_allowance: 0x38b4c7608720b692,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
        ],
        minimum_endpoint: 1,
        maximum_endpoint: 0,
        wet_possible: true,
        dry_possible: false,
    },
    Case {
        id: "rectangle_cap_deficit__physical_anchor",
        query_id: "rectangle_cap_deficit",
        query_index: 3,
        source_components: [0x4018000000000000, 0x0000000000000000],
        source_potential: Reference {
            bounds: [0x400bb67ae8584caa, 0x400bb67ae8584cab],
            native_allowance: 0x3df7ce1cac9fa1d1,
            policy: ReferencePolicy::PhysicalLocal,
        },
        source_sound: Reference {
            bounds: [0x3ffbb67ae8584caa, 0x3ffbb67ae8584cab],
            native_allowance: 0x3de7ce1cac9fa1d1,
            policy: ReferencePolicy::PhysicalLocal,
        },
        endpoints: [
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x400bb67ae8584caa, 0x400bb67ae8584cab],
                    native_allowance: 0x3df7ce1cac9fa1d1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3ffbb67ae8584caa, 0x3ffbb67ae8584cab],
                    native_allowance: 0x3de7ce1cac9fa1d1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbad279a74590331d, 0xbad279a74590331c],
                    native_allowance: 0x38bfbd7b90d4d7c1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x400bb67ae8584caa, 0x400bb67ae8584cab],
                    native_allowance: 0x3df7ce1cac9fa1d1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3ffbb67ae8584caa, 0x3ffbb67ae8584cab],
                    native_allowance: 0x3de7ce1cac9fa1d1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbae279a74590331d, 0xbae279a74590331c],
                    native_allowance: 0x38cfbd7b90d4d7c1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
        ],
        minimum_endpoint: 1,
        maximum_endpoint: 0,
        wet_possible: true,
        dry_possible: false,
    },
];
