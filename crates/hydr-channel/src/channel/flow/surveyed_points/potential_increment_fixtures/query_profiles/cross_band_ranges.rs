use super::super::model::{Reference, ReferencePolicy};
use super::model::{Case, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "three_band_broad__physical_anchor",
        query_id: "three_band_broad",
        query_index: 11,
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
                structural_equality: false,
                potential: Reference {
                    bounds: [0x3ff536854d37b52f, 0x3ff536854d37b530],
                    native_allowance: 0x3de238c8220be6e3,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3fe306fe0a31b715, 0x3fe306fe0a31b716],
                    native_allowance: 0x3dd058268b437caa,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x3ff536854d37b52f, 0x3ff536854d37b530],
                    native_allowance: 0x3de238c8220be6e3,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x400d50adf2cca326, 0x400d50adf2cca327],
                    native_allowance: 0x3df92e7878b67619,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3ff4a3d7477d1583, 0x3ff4a3d7477d1584],
                    native_allowance: 0x3de1bac8e8ccfc15,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x400d50adf2cca326, 0x400d50adf2cca327],
                    native_allowance: 0x3df92e7878b67619,
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
        id: "three_band_broad_reverse__physical_anchor",
        query_id: "three_band_broad_reverse",
        query_index: 13,
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
                    bounds: [0x400e257b8f093f4a, 0x400e257b8f093f4b],
                    native_allowance: 0x3df9e54461e88a7a,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3ff4e77d25f25b23, 0x3ff4e77d25f25b24],
                    native_allowance: 0x3de1f4e4d8766063,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbfb8f0bc0ae227c7, 0xbfb8f0bc0ae227c6],
                    native_allowance: 0x3da56c71f9400a3a,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x3ff536854d37b52f, 0x3ff536854d37b530],
                    native_allowance: 0x3de238c8220be6e3,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3fe306fe0a31b715, 0x3fe306fe0a31b716],
                    native_allowance: 0x3dd058268b437caa,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xc00451bec8c475f1, 0xc00451bec8c475f0],
                    native_allowance: 0x3df17443e0ac975a,
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
        id: "first_knot_from_second_source",
        query_id: "source_first_knot_forward",
        query_index: 0,
        source_components: [0x401e000000000000, 0x3af0000000000000],
        source_potential: Reference {
            bounds: [0x40089b70bbdbbd40, 0x40089b70bbdbbd41],
            native_allowance: 0x3df5232d9616f30c,
            policy: ReferencePolicy::PhysicalLocal,
        },
        source_sound: Reference {
            bounds: [0x3ff3988e1409212e, 0x3ff3988e1409212f],
            native_allowance: 0x3de0d5301bfa13a1,
            policy: ReferencePolicy::PhysicalLocal,
        },
        endpoints: [
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x4000fcb49ac6ad7b, 0x4000fcb49ac6ad7c],
                    native_allowance: 0x3ded2f04e63c544f,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3febb67ae8584caa, 0x3febb67ae8584cab],
                    native_allowance: 0x3dd7ce1cac9fa1d1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbfee7af084543f14, 0xbfee7af084543f13],
                    native_allowance: 0x3dda2eac8be32391,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x4000fcb49ac6ad7b, 0x4000fcb49ac6ad7c],
                    native_allowance: 0x3ded2f04e63c544f,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3febb67ae8584caa, 0x3febb67ae8584cab],
                    native_allowance: 0x3dd7ce1cac9fa1d1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0xbfee7af084543f14, 0xbfee7af084543f13],
                    native_allowance: 0x3dda2eac8be32391,
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
        id: "second_knot_from_first_source",
        query_id: "source_second_knot_reverse",
        query_index: 1,
        source_components: [0x4008000000000000, 0xbaf0000000000000],
        source_potential: Reference {
            bounds: [0x4000fcb49ac6ad7b, 0x4000fcb49ac6ad7c],
            native_allowance: 0x3ded2f04e63c544f,
            policy: ReferencePolicy::PhysicalLocal,
        },
        source_sound: Reference {
            bounds: [0x3febb67ae8584caa, 0x3febb67ae8584cab],
            native_allowance: 0x3dd7ce1cac9fa1d1,
            policy: ReferencePolicy::PhysicalLocal,
        },
        endpoints: [
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x40089b70bbdbbd40, 0x40089b70bbdbbd41],
                    native_allowance: 0x3df5232d9616f30c,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3ff3988e1409212e, 0x3ff3988e1409212f],
                    native_allowance: 0x3de0d5301bfa13a1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x3fee7af084543f13, 0x3fee7af084543f14],
                    native_allowance: 0x3dda2eac8be32391,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x40089b70bbdbbd40, 0x40089b70bbdbbd41],
                    native_allowance: 0x3df5232d9616f30c,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x3ff3988e1409212e, 0x3ff3988e1409212f],
                    native_allowance: 0x3de0d5301bfa13a1,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x3fee7af084543f13, 0x3fee7af084543f14],
                    native_allowance: 0x3dda2eac8be32391,
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
