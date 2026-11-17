use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
        expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
        id: "uniform_cap_represented_tenth_forward",
        config: 6,
        source_group: "new_exact_point_controls",
        group: "15193f9a2f93d81eb8aad2c58a13c30ac0f4839faef1e1a12e8fc8d5e978e3cd",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x402c000000000000, 0x402c000000000000],
            depth: Some([0x4008000000000000, 0x4008000000000000]),
            velocity: Some([0x3fb999999999999a, 0x3fb999999999999a]),
            discharge: [0x3ff6666666666666, 0x3ff6666666666667],
            velocity_provenance: "direct",
        }],
        star: Some(State {
            label: "star",
            area: [0x402c000000000000, 0x402c000000000000],
            depth: None,
            velocity: Some([0x3fb999999999999a, 0x3fb999999999999a]),
            discharge: [0x3ff6666666666666, 0x3ff6666666666667],
            velocity_provenance: "direct",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xbff390e610395f51, 0xbff390e610395f50],
            },
            Edge {
                label: "edge_1",
                bounds: [0xbff390e610395f51, 0xbff390e610395f50],
            },
            Edge {
                label: "edge_2",
                bounds: [0xbff390e610395f51, 0xbff390e610395f50],
            },
            Edge {
                label: "edge_3",
                bounds: [0x3ff6c419436c9283, 0x3ff6c419436c9284],
            },
            Edge {
                label: "edge_4",
                bounds: [0x3ff6c419436c9283, 0x3ff6c419436c9284],
            },
            Edge {
                label: "edge_5",
                bounds: [0x3ff6c419436c9283, 0x3ff6c419436c9284],
            },
        ],
    },
    Execution {
        config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
        expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
        id: "uniform_cap_represented_tenth_reverse",
        config: 6,
        source_group: "new_exact_point_controls",
        group: "084fe2f6678024804dea1ada47a617464fb3bcfbc5bfe22e9ccc97f540dc5171",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x402c000000000000, 0x402c000000000000],
            depth: Some([0x4008000000000000, 0x4008000000000000]),
            velocity: Some([0xbfb999999999999a, 0xbfb999999999999a]),
            discharge: [0xbff6666666666667, 0xbff6666666666666],
            velocity_provenance: "direct",
        }],
        star: Some(State {
            label: "star",
            area: [0x402c000000000000, 0x402c000000000000],
            depth: None,
            velocity: Some([0xbfb999999999999a, 0xbfb999999999999a]),
            discharge: [0xbff6666666666667, 0xbff6666666666666],
            velocity_provenance: "direct",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xbff6c419436c9284, 0xbff6c419436c9283],
            },
            Edge {
                label: "edge_1",
                bounds: [0xbff6c419436c9284, 0xbff6c419436c9283],
            },
            Edge {
                label: "edge_2",
                bounds: [0xbff6c419436c9284, 0xbff6c419436c9283],
            },
            Edge {
                label: "edge_3",
                bounds: [0x3ff390e610395f50, 0x3ff390e610395f51],
            },
            Edge {
                label: "edge_4",
                bounds: [0x3ff390e610395f50, 0x3ff390e610395f51],
            },
            Edge {
                label: "edge_5",
                bounds: [0x3ff390e610395f50, 0x3ff390e610395f51],
            },
        ],
    },
    Execution {
        config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
        expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
        id: "uniform_cap_retained_deficit_dyadic_velocity",
        config: 6,
        source_group: "new_exact_point_controls",
        group: "4865512702cb2fc8486f5a9a44c49a68524ab10b3e605b80f7bf37b586304ae2",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x402bffffffffffff, 0x402c000000000000],
            depth: Some([0x4007ffffffffffff, 0x4008000000000000]),
            velocity: Some([0x3fc0000000000000, 0x3fc0000000000000]),
            discharge: [0x3ffbffffffffffff, 0x3ffc000000000000],
            velocity_provenance: "direct",
        }],
        star: Some(State {
            label: "star",
            area: [0x402bffffffffffff, 0x402c000000000000],
            depth: None,
            velocity: Some([0x3fc0000000000000, 0x3fc0000000000000]),
            discharge: [0x3ffbffffffffffff, 0x3ffc000000000000],
            velocity_provenance: "direct",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xbff32a7fa9d2f8ea, 0xbff32a7fa9d2f8e9],
            },
            Edge {
                label: "edge_1",
                bounds: [0xbff32a7fa9d2f8ea, 0xbff32a7fa9d2f8e9],
            },
            Edge {
                label: "edge_2",
                bounds: [0xbff32a7fa9d2f8ea, 0xbff32a7fa9d2f8e9],
            },
            Edge {
                label: "edge_3",
                bounds: [0x3ff72a7fa9d2f8e9, 0x3ff72a7fa9d2f8ea],
            },
            Edge {
                label: "edge_4",
                bounds: [0x3ff72a7fa9d2f8e9, 0x3ff72a7fa9d2f8ea],
            },
            Edge {
                label: "edge_5",
                bounds: [0x3ff72a7fa9d2f8e9, 0x3ff72a7fa9d2f8ea],
            },
        ],
    },
];
