use super::super::model::{Reference, ReferencePolicy};
use super::model::{Case, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "source_first_knot_forward__physical_anchor",
        query_id: "source_first_knot_forward",
        query_index: 0,
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
                structural_equality: true,
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
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: 0x0000000000000000,
                    policy: ReferencePolicy::ExactZero,
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
                    bounds: [0x3ae279a74590331c, 0x3ae279a74590331d],
                    native_allowance: 0x38cfbd7b90d4d7c1,
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
        id: "first_band_start_signed_offsets__physical_anchor",
        query_id: "first_band_start_signed_offsets",
        query_index: 4,
        source_components: [0x4008000000000000, 0x0000000000000000],
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
                    bounds: [0xbad279a74590331d, 0xbad279a74590331c],
                    native_allowance: 0x38bfbd7b90d4d7c1,
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
                    bounds: [0x3ad279a74590331c, 0x3ad279a74590331d],
                    native_allowance: 0x38bfbd7b90d4d7c1,
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
