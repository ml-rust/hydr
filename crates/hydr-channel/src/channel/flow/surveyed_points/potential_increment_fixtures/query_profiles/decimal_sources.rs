use super::super::model::{Reference, ReferencePolicy};
use super::model::{Case, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "decimal_knot_signed_offsets__physical_anchor",
        query_id: "decimal_knot_signed_offsets",
        query_index: 6,
        source_components: [0x3feccccccccccccd, 0xbc80000000000000],
        source_potential: Reference {
            bounds: [0x4005775c544ff262, 0x4005775c544ff263],
            native_allowance: 0x3df2707a96d7b932,
            policy: ReferencePolicy::PhysicalLocal,
        },
        source_sound: Reference {
            bounds: [0x3fe5775c544ff262, 0x3fe5775c544ff263],
            native_allowance: 0x3dd2707a96d7b932,
            policy: ReferencePolicy::PhysicalLocal,
        },
        endpoints: [
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x4005775c544ff262, 0x4005775c544ff263],
                    native_allowance: 0x3df2707a96d7b932,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3fe5775c544ff262, 0x3fe5775c544ff263],
                    native_allowance: 0x3dd2707a96d7b932,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbae7d9f4cf754636, 0xbae7d9f4cf754635],
                    native_allowance: 0x38d47cf9fcefb154,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x4005775c544ff262, 0x4005775c544ff263],
                    native_allowance: 0x3df2707a96d7b932,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3fe5775c544ff262, 0x3fe5775c544ff263],
                    native_allowance: 0x3dd2707a96d7b932,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x3ae7d9f4cf754635, 0x3ae7d9f4cf754636],
                    native_allowance: 0x38d47cf9fcefb154,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
        wet_possible: true,
        dry_possible: false,
    },
    Case {
        id: "decimal_broad__physical_anchor",
        query_id: "decimal_broad",
        query_index: 12,
        source_components: [0x0000000000000000, 0x0000000000000000],
        source_potential: Reference {
            bounds: [0x0000000000000000, 0x0000000000000000],
            native_allowance: 0x0000000000000000,
            policy: ReferencePolicy::ExactZero,
        },
        source_sound: Reference {
            bounds: [0x0000000000000000, 0x0000000000000000],
            native_allowance: 0x0000000000000000,
            policy: ReferencePolicy::ExactZero,
        },
        endpoints: [
            Endpoint {
                structural_equality: true,
                potential: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: 0x0000000000000000,
                    policy: ReferencePolicy::ExactZero,
                },
                sound: Reference {
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: 0x0000000000000000,
                    policy: ReferencePolicy::ExactZero,
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
                    bounds: [0x401003b768cbd6c0, 0x401003b768cbd6c1],
                    native_allowance: 0x3dfb83425c1ac212,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3ff1e7f34723e875, 0x3ff1e7f34723e876],
                    native_allowance: 0x3ddec32aaa0de2b2,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x401003b768cbd6c0, 0x401003b768cbd6c1],
                    native_allowance: 0x3dfb83425c1ac212,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
        wet_possible: true,
        dry_possible: true,
    },
    Case {
        id: "decimal_knot_from_wet_source",
        query_id: "decimal_knot_signed_offsets",
        query_index: 6,
        source_components: [0x4010000000000000, 0x0000000000000000],
        source_potential: Reference {
            bounds: [0x401003b768cbd6c0, 0x401003b768cbd6c1],
            native_allowance: 0x3dfb83425c1ac212,
            policy: ReferencePolicy::PhysicalLocal,
        },
        source_sound: Reference {
            bounds: [0x3ff1e7f34723e875, 0x3ff1e7f34723e876],
            native_allowance: 0x3ddec32aaa0de2b2,
            policy: ReferencePolicy::PhysicalLocal,
        },
        endpoints: [
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x4005775c544ff262, 0x4005775c544ff263],
                    native_allowance: 0x3df2707a96d7b932,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3fe5775c544ff262, 0x3fe5775c544ff263],
                    native_allowance: 0x3dd2707a96d7b932,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbff52024fa8f763c, 0xbff52024fa8f763b],
                    native_allowance: 0x3de2258f8a8611c0,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x4005775c544ff262, 0x4005775c544ff263],
                    native_allowance: 0x3df2707a96d7b932,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3fe5775c544ff262, 0x3fe5775c544ff263],
                    native_allowance: 0x3dd2707a96d7b932,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbff52024fa8f763c, 0xbff52024fa8f763b],
                    native_allowance: 0x3de2258f8a8611c0,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
        wet_possible: true,
        dry_possible: false,
    },
];
