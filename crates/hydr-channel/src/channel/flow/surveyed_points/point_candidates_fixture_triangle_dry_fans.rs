use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "fcc933c63bfc458ecbd2ffb0fbb7a4b4365728fc7e9197f70bbb1b583cfaf99d",
        expected_identity: "146e2f01c2b1361448229c4e3d52a1906000b182540081dbd9c0bdd3c8880992",
        id: "scalar_triangle_right_dry_sonic",
        config: 0,
        source_group: "existing_scalar_37",
        group: "0d573ba941bae7ac982d07abfdbd7fa31d9ac58fa9ada6287b19f4015406cd0c",
        declared: "right_dry",
        outcome: "resolved",
        physical: "right_dry",
        traces: &[State {
            label: "left-sonic",
            area: [0x3ffa36e2eb1c432c, 0x3ffa36e2eb1c432d],
            depth: Some([0x3ff47ae147ae147a, 0x3ff47ae147ae147b]),
            velocity: Some([0x3fe9999999999999, 0x3fe999999999999a]),
            discharge: [0x3ff4f8b588e368f0, 0x3ff4f8b588e368f1],
            velocity_provenance: "Q_over_A",
        }],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xbff0000000000000, 0xbff0000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x4010000000000000, 0x4010000000000000],
            },
        ],
    },
    Execution {
        config_identity: "eb258aeb82d3f99c112f95e626b9948213fa12156e54b4410eff7ba526df7ceb",
        expected_identity: "3a69830f62c3706b2f226a32a1a341e0bb6bba035d97193b8ce4d13a0f919dcd",
        id: "scalar_triangle_left_dry_sonic",
        config: 0,
        source_group: "existing_scalar_37",
        group: "f1d051650ade88b9a0c1224d75a5efbf193f7be5b1e1dab64da5c2dfca90507d",
        declared: "left_dry",
        outcome: "resolved",
        physical: "left_dry",
        traces: &[State {
            label: "right-sonic",
            area: [0x3ffa36e2eb1c432c, 0x3ffa36e2eb1c432d],
            depth: Some([0x3ff47ae147ae147a, 0x3ff47ae147ae147b]),
            velocity: Some([0xbfe999999999999a, 0xbfe9999999999999]),
            discharge: [0xbff4f8b588e368f1, 0xbff4f8b588e368f0],
            velocity_provenance: "Q_over_A",
        }],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0x3ff0000000000000, 0x3ff0000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0xc010000000000000, 0xc010000000000000],
            },
        ],
    },
];
