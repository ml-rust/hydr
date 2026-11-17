use super::super::model::{Reference, ReferencePolicy};
use super::model::{Case, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "triangle_exact_dry__physical_anchor",
        query_id: "triangle_exact_dry",
        query_index: 9,
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
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
        wet_possible: false,
        dry_possible: true,
    },
    Case {
        id: "triangle_touching_dry__physical_anchor",
        query_id: "triangle_touching_dry",
        query_index: 10,
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
                    bounds: [0x36a47238cf17c239, 0x36a47238cf17c23a],
                    native_allowance: 0x34919029920ad502,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x36847238cf17c239, 0x36847238cf17c23a],
                    native_allowance: 0x34719029920ad502,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x36a47238cf17c239, 0x36a47238cf17c23a],
                    native_allowance: 0x34919029920ad502,
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
        id: "triangle_scaled_tiny_positive__physical_anchor",
        query_id: "triangle_scaled_tiny_positive",
        query_index: 15,
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
                    bounds: [0x30647238cf17c239, 0x30647238cf17c23a],
                    native_allowance: 0x2e519029920ad502,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x30447238cf17c239, 0x30447238cf17c23a],
                    native_allowance: 0x2e319029920ad502,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x30647238cf17c239, 0x30647238cf17c23a],
                    native_allowance: 0x2e519029920ad502,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x36a47238cf17c239, 0x36a47238cf17c23a],
                    native_allowance: 0x34919029920ad502,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x36847238cf17c239, 0x36847238cf17c23a],
                    native_allowance: 0x34719029920ad502,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x36a47238cf17c239, 0x36a47238cf17c23a],
                    native_allowance: 0x34919029920ad502,
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
        id: "triangle_dry_from_wet_source",
        query_id: "triangle_exact_dry",
        query_index: 9,
        source_components: [0x4010000000000000, 0x0000000000000000],
        source_potential: Reference {
            bounds: [0x400cea4ebfc356e5, 0x400cea4ebfc356e6],
            native_allowance: 0x3df8d688a8a08651,
            policy: ReferencePolicy::PhysicalLocal,
        },
        source_sound: Reference {
            bounds: [0x3fecea4ebfc356e5, 0x3fecea4ebfc356e6],
            native_allowance: 0x3dd8d688a8a08651,
            policy: ReferencePolicy::PhysicalLocal,
        },
        endpoints: [
            Endpoint {
                structural_equality: false,
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
                    bounds: [0xc00cea4ebfc356e6, 0xc00cea4ebfc356e5],
                    native_allowance: 0x3df8d688a8a08651,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
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
                    bounds: [0xc00cea4ebfc356e6, 0xc00cea4ebfc356e5],
                    native_allowance: 0x3df8d688a8a08651,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
        ],
        minimum_endpoint: 0,
        maximum_endpoint: 1,
        wet_possible: false,
        dry_possible: true,
    },
];
