use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[Execution {
    config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
    expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
    id: "tiny_rectangle_compression",
    config: 6,
    source_group: "new_exact_point_controls",
    group: "078abd9a23ad5af2ff2ce6d81676ea7bed0eb1d2c71177d60005c3d7b3964cd6",
    declared: "wet_wet",
    outcome: "resolved",
    physical: "wet_wet",
    traces: &[State {
        label: "left-exterior",
        area: [0x1a70000000000000, 0x1a70000000000000],
        depth: Some([0x1a70000000000000, 0x1a70000000000000]),
        velocity: Some([0x4014000000000000, 0x4014000000000000]),
        discharge: [0x1a94000000000000, 0x1a94000000000000],
        velocity_provenance: "direct",
    }],
    star: Some(State {
        label: "star",
        area: [0x2d26a09e667f3bcc, 0x2d26a09e667f3bcd],
        depth: None,
        velocity: Some([0x4011ffffffffffff, 0x4012000000000001]),
        discharge: [0x2d4974b2334f2346, 0x2d4974b2334f2347],
        velocity_provenance: "direct",
    }),
    edges: &[
        Edge {
            label: "edge_0",
            bounds: [0x4011ffffffffffff, 0x4012000000000001],
        },
        Edge {
            label: "edge_1",
            bounds: [0x4011ffffffffffff, 0x4012000000000001],
        },
    ],
}];
