use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
        expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
        id: "tiny_rectangle_right_dry_fan",
        config: 6,
        source_group: "new_exact_point_controls",
        group: "63b64f74291a3c2a9f684a6e7ce127615a3c579d0bb9bc3c29fa4377f761c341",
        declared: "right_dry",
        outcome: "resolved",
        physical: "right_dry",
        traces: &[State {
            label: "left-sonic",
            area: [0x1a5c71c71c71c71c, 0x1a5c71c71c71c71d],
            depth: Some([0x1a5c71c71c71c71c, 0x1a5c71c71c71c71d]),
            velocity: Some([0x2d25555555555555, 0x2d25555555555556]),
            discharge: [0x0792f684bda12f68, 0x0792f684bda12f69],
            velocity_provenance: "direct",
        }],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xad30000000000000, 0xad30000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x2d40000000000000, 0x2d40000000000000],
            },
        ],
    },
    Execution {
        config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
        expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
        id: "retained_vacuum_inside_film",
        config: 6,
        source_group: "new_exact_point_controls",
        group: "533a69771241e58d702e64d195c21e9e998bfae79d72c6a93d0b3f0159f68b05",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "right-sonic",
            area: [0x35bc71c71c71c71c, 0x35bc71c71c71c71d],
            depth: Some([0x35bc71c71c71c71c, 0x35bc71c71c71c71d]),
            velocity: Some([0xbad5555555555556, 0xbad5555555555555]),
            discharge: [0xb0a2f684bda12f69, 0xb0a2f684bda12f68],
            velocity_provenance: "direct",
        }],
        star: Some(State {
            label: "star",
            area: [0x35b0000000000000, 0x35b0000000000000],
            depth: None,
            velocity: Some([0xbae0000000000000, 0xbae0000000000000]),
            discharge: [0xb0a0000000000000, 0xb0a0000000000000],
            velocity_provenance: "direct",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc008000000000000, 0xc008000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0xbae8000000000000, 0xbae8000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0x4007ffffffffffff, 0x4008000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0xbad0000000000000, 0xbad0000000000000],
            },
        ],
    },
];
