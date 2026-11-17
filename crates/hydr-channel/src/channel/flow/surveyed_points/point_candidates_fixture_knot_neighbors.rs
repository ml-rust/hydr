use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "7d854333d3b8f34f63692b12a6ff3e32a63b1c94bb44b34ffa8b453c4e286da9",
        expected_identity: "b7af7090fb767df6165dba4a9f9150459329ec69501631233610b03abbcf41ac",
        id: "scalar_three_bands_first_knot_neighbors",
        config: 0,
        source_group: "existing_scalar_37",
        group: "9f2c313a856c90d1fae697dd41dce2b6e16751ff243e7ad2d79e5875a87599e4",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x4000000000000000, 0x4000000000000001],
            depth: Some([0x3ff0000000000000, 0x3ff0000000000001]),
            velocity: Some([0xbca3988e1409212f, 0xbca3988e1409212e]),
            discharge: [0xbcb3988e1409212f, 0xbcb3988e1409212e],
            velocity_provenance: "Q_over_A",
        }],
        star: Some(State {
            label: "star",
            area: [0x4000000000000000, 0x4000000000000001],
            depth: Some([0x3ff0000000000000, 0x3ff0000000000001]),
            velocity: Some([0xbca3988e1409212f, 0xbca3988e1409212e]),
            discharge: [0xbcb3988e1409212f, 0xbcb3988e1409212e],
            velocity_provenance: "Q_over_A",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xbfea20bd700c2c3f, 0xbfea20bd700c2c3e],
            },
            Edge {
                label: "edge_1",
                bounds: [0x3fea20bd700c2c3e, 0x3fea20bd700c2c3f],
            },
            Edge {
                label: "edge_2",
                bounds: [0x3fea20bd700c2c3c, 0x3fea20bd700c2c3d],
            },
        ],
    },
    Execution {
        config_identity: "ec49cf932c91314e9843522df614c92fdee23c00b1a0a22d452fb8d7ed0b7fb5",
        expected_identity: "cdaf3cc837a342ac4e3057fd0619c38c2645593966a39728569c0fe4b500d8a0",
        id: "scalar_three_bands_second_knot_neighbors",
        config: 0,
        source_group: "existing_scalar_37",
        group: "b69914a0086fab7b035ce511ea816b1aab274b110e11b7be6a77f9019a6a6fff",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x401bffffffffffff, 0x401c000000000000],
            depth: Some([0x3fffffffffffffff, 0x4000000000000000]),
            velocity: Some([0xbca2492492492493, 0xbca2492492492492]),
            discharge: [0xbcd0000000000001, 0xbcd0000000000000],
            velocity_provenance: "Q_over_A",
        }],
        star: Some(State {
            label: "star",
            area: [0x401bffffffffffff, 0x401c000000000000],
            depth: Some([0x3fffffffffffffff, 0x4000000000000000]),
            velocity: Some([0xbca2492492492493, 0xbca2492492492492]),
            discharge: [0xbcd0000000000001, 0xbcd0000000000000],
            velocity_provenance: "Q_over_A",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xbff0000000000001, 0xbff0000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x3ff0000000000000, 0x3ff0000000000001],
            },
            Edge {
                label: "edge_2",
                bounds: [0x3feffffffffffffe, 0x3fefffffffffffff],
            },
        ],
    },
];
