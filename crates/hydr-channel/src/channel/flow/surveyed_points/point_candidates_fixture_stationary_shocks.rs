use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "438de0538495334ff6b85aa3672869bec4c28c507ccd4e1a5bf588e79ec7afcd",
        expected_identity: "9361f343d7ea2def24c7738b6021367ee97b5072c379fd3ab4beab70d147d6d1",
        id: "scalar_rectangle_stationary_shock",
        config: 2,
        source_group: "existing_scalar_37",
        group: "809ff4033cb602b9049a91c7d5a325a14ffa57bf6b1ed1c002d3c3084e96b2e0",
        declared: "stationary_shock",
        outcome: "resolved",
        physical: "stationary_shock",
        traces: &[
            State {
                label: "left-exterior",
                area: [0x3ff0000000000000, 0x3ff0000000000000],
                depth: Some([0x3ff0000000000000, 0x3ff0000000000000]),
                velocity: Some([0x4008000000000000, 0x4008000000000000]),
                discharge: [0x4008000000000000, 0x4008000000000000],
                velocity_provenance: "Q_over_A",
            },
            State {
                label: "star",
                area: [0x4000000000000000, 0x4000000000000000],
                depth: Some([0x4000000000000000, 0x4000000000000000]),
                velocity: Some([0x3ff8000000000000, 0x3ff8000000000000]),
                discharge: [0x4008000000000000, 0x4008000000000000],
                velocity_provenance: "Q_over_A",
            },
        ],
        star: Some(State {
            label: "star",
            area: [0x4000000000000000, 0x4000000000000000],
            depth: Some([0x4000000000000000, 0x4000000000000000]),
            velocity: Some([0x3ff8000000000000, 0x3ff8000000000000]),
            discharge: [0x4008000000000000, 0x4008000000000000],
            velocity_provenance: "Q_over_A",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0x0000000000000000, 0x0000000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x400f988e1409212e, 0x400f988e1409212f],
            },
            Edge {
                label: "edge_2",
                bounds: [0x400f988e1409212e, 0x400f988e1409212f],
            },
            Edge {
                label: "edge_3",
                bounds: [0x400f988e1409212e, 0x400f988e1409212f],
            },
        ],
    },
    Execution {
        config_identity: "e47297287da213a8b0074b6b22766b06111a74dc8f839450f998cbbcebce8131",
        expected_identity: "e3f09ab424108e6b64110f0699484d67ab12a5fa5d1faba6bd6fa7bcabebf3d5",
        id: "scalar_rectangle_stationary_shock_reverse",
        config: 2,
        source_group: "existing_scalar_37",
        group: "161a1067e2774f21a5a500d4e4380364d6f5878bcc99d44ce8868f487f8b2500",
        declared: "stationary_shock",
        outcome: "resolved",
        physical: "stationary_shock",
        traces: &[
            State {
                label: "right-exterior",
                area: [0x3ff0000000000000, 0x3ff0000000000000],
                depth: Some([0x3ff0000000000000, 0x3ff0000000000000]),
                velocity: Some([0xc008000000000000, 0xc008000000000000]),
                discharge: [0xc008000000000000, 0xc008000000000000],
                velocity_provenance: "Q_over_A",
            },
            State {
                label: "star",
                area: [0x4000000000000000, 0x4000000000000000],
                depth: Some([0x4000000000000000, 0x4000000000000000]),
                velocity: Some([0xbff8000000000000, 0xbff8000000000000]),
                discharge: [0xc008000000000000, 0xc008000000000000],
                velocity_provenance: "Q_over_A",
            },
        ],
        star: Some(State {
            label: "star",
            area: [0x4000000000000000, 0x4000000000000000],
            depth: Some([0x4000000000000000, 0x4000000000000000]),
            velocity: Some([0xbff8000000000000, 0xbff8000000000000]),
            discharge: [0xc008000000000000, 0xc008000000000000],
            velocity_provenance: "Q_over_A",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc00f988e1409212f, 0xc00f988e1409212e],
            },
            Edge {
                label: "edge_1",
                bounds: [0xc00f988e1409212f, 0xc00f988e1409212e],
            },
            Edge {
                label: "edge_2",
                bounds: [0xc00f988e1409212f, 0xc00f988e1409212e],
            },
            Edge {
                label: "edge_3",
                bounds: [0x0000000000000000, 0x0000000000000000],
            },
        ],
    },
    Execution {
        config_identity: "3635fcf0c7e98822ded25ee515479412913b7a792a33d17714c2d6b832772f58",
        expected_identity: "6d7d4582319fdff5109ecd874c12e4a4a1b6c7f975ed7db85c62a1cd8745fc50",
        id: "scalar_trapezoid_stationary_shock",
        config: 3,
        source_group: "existing_scalar_37",
        group: "c3e7bee966b616cbf5d064d34ba041d472d495644426c26d61fb20d19b7bf419",
        declared: "stationary_shock",
        outcome: "resolved",
        physical: "stationary_shock",
        traces: &[
            State {
                label: "left-exterior",
                area: [0x4000000000000000, 0x4000000000000000],
                depth: Some([0x3ff0000000000000, 0x3ff0000000000000]),
                velocity: Some([0x4027000000000000, 0x4027000000000000]),
                discharge: [0x4037000000000000, 0x4037000000000000],
                velocity_provenance: "Q_over_A",
            },
            State {
                label: "star",
                area: [0x4018000000000000, 0x4018000000000000],
                depth: Some([0x4000000000000000, 0x4000000000000000]),
                velocity: Some([0x400eaaaaaaaaaaaa, 0x400eaaaaaaaaaaab]),
                discharge: [0x4036ffffffffffff, 0x4037000000000001],
                velocity_provenance: "Q_over_A",
            },
        ],
        star: Some(State {
            label: "star",
            area: [0x4018000000000000, 0x4018000000000000],
            depth: Some([0x4000000000000000, 0x4000000000000000]),
            velocity: Some([0x400eaaaaaaaaaaaa, 0x400eaaaaaaaaaaab]),
            discharge: [0x4036ffffffffffff, 0x4037000000000001],
            velocity_provenance: "Q_over_A",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0x0000000000000000, 0x0000000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x402686a867cc92f9, 0x402686a867cc92fa],
            },
            Edge {
                label: "edge_2",
                bounds: [0x402686a867cc92f9, 0x402686a867cc92fa],
            },
            Edge {
                label: "edge_3",
                bounds: [0x402686a867cc92f9, 0x402686a867cc92fa],
            },
        ],
    },
];
