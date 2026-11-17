use super::super::model::{Reference, ReferencePolicy};
use super::model::{Case, Endpoint};

pub(crate) const CASES: &[Case] = &[
    Case {
        id: "source_second_knot_reverse__physical_anchor",
        query_id: "source_second_knot_reverse",
        query_index: 1,
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
                structural_equality: true,
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
                    bounds: [0x0000000000000000, 0x0000000000000000],
                    native_allowance: 0x0000000000000000,
                    policy: ReferencePolicy::ExactZero,
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
                    bounds: [0xbad4e6fdf33cf032, 0xbad4e6fdf33cf031],
                    native_allowance: 0x38c1f477954f03df,
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
        id: "middle_band_end_signed_offsets__physical_anchor",
        query_id: "middle_band_end_signed_offsets",
        query_index: 5,
        source_components: [0x401e000000000000, 0x0000000000000000],
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
                    bounds: [0xbac4e6fdf33cf032, 0xbac4e6fdf33cf031],
                    native_allowance: 0x38b1f477954f03df,
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
                    bounds: [0x3ac4e6fdf33cf031, 0x3ac4e6fdf33cf032],
                    native_allowance: 0x38b1f477954f03df,
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
