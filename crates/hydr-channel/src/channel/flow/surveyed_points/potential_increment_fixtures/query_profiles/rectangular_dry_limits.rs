use super::super::model::{Reference, ReferencePolicy};
use super::model::{Case, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "rectangle_exact_dry__physical_anchor",
        query_id: "rectangle_exact_dry",
        query_index: 7,
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
        id: "rectangle_touching_dry__physical_anchor",
        query_id: "rectangle_touching_dry",
        query_index: 8,
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
                    bounds: [0x2d36a09e667f3bcc, 0x2d36a09e667f3bcd],
                    native_allowance: 0x2b236fd255a2213a,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x2d26a09e667f3bcc, 0x2d26a09e667f3bcd],
                    native_allowance: 0x2b136fd255a2213a,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x2d36a09e667f3bcc, 0x2d36a09e667f3bcd],
                    native_allowance: 0x2b236fd255a2213a,
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
        id: "rectangle_scaled_tiny_positive__physical_anchor",
        query_id: "rectangle_scaled_tiny_positive",
        query_index: 14,
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
                    bounds: [0x20b6a09e667f3bcc, 0x20b6a09e667f3bcd],
                    native_allowance: 0x1ea36fd255a2213a,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x20a6a09e667f3bcc, 0x20a6a09e667f3bcd],
                    native_allowance: 0x1e936fd255a2213a,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x20b6a09e667f3bcc, 0x20b6a09e667f3bcd],
                    native_allowance: 0x1ea36fd255a2213a,
                    policy: ReferencePolicy::PhysicalLocal,
                },
            },
            Endpoint {
                structural_equality: false,
                potential: Reference {
                    bounds: [0x2d36a09e667f3bcc, 0x2d36a09e667f3bcd],
                    native_allowance: 0x2b236fd255a2213a,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                sound: Reference {
                    bounds: [0x2d26a09e667f3bcc, 0x2d26a09e667f3bcd],
                    native_allowance: 0x2b136fd255a2213a,
                    policy: ReferencePolicy::PhysicalLocal,
                },
                increment: Reference {
                    bounds: [0x2d36a09e667f3bcc, 0x2d36a09e667f3bcd],
                    native_allowance: 0x2b236fd255a2213a,
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
        id: "rectangle_dry_from_wet_source",
        query_id: "rectangle_exact_dry",
        query_index: 7,
        source_components: [0x3ff0000000000000, 0x0000000000000000],
        source_potential: Reference {
            bounds: [0x3ff6a09e667f3bcc, 0x3ff6a09e667f3bcd],
            native_allowance: 0x3de36fd255a2213a,
            policy: ReferencePolicy::PhysicalLocal,
        },
        source_sound: Reference {
            bounds: [0x3fe6a09e667f3bcc, 0x3fe6a09e667f3bcd],
            native_allowance: 0x3dd36fd255a2213a,
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
                    bounds: [0xbff6a09e667f3bcd, 0xbff6a09e667f3bcc],
                    native_allowance: 0x3de36fd255a2213a,
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
                    bounds: [0xbff6a09e667f3bcd, 0xbff6a09e667f3bcc],
                    native_allowance: 0x3de36fd255a2213a,
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
