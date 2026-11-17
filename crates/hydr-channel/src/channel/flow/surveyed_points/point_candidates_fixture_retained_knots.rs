use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
        expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
        id: "retained_knot_2_crossing",
        config: 6,
        source_group: "new_exact_point_controls",
        group: "4e6456843534b5d5a53a154530cfc4f9aa3a7d1b1c85bc7ac8a0e60f8a8f7e10",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x3fffffffffffffff, 0x4000000000000000],
            depth: Some([0x3fefffffffffffff, 0x3ff0000000000000]),
            velocity: Some([0xbada20bd700c2c3e, 0xbada20bd700c2c3d]),
            discharge: [0xbaea20bd700c2c3e, 0xbaea20bd700c2c3d],
            velocity_provenance: "direct",
        }],
        star: Some(State {
            label: "star",
            area: [0x3fffffffffffffff, 0x4000000000000000],
            depth: None,
            velocity: Some([0xbada20bd700c2c3e, 0xbada20bd700c2c3d]),
            discharge: [0xbaea20bd700c2c3e, 0xbaea20bd700c2c3d],
            velocity_provenance: "direct",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xbfea20bd700c2c3e, 0xbfea20bd700c2c3d],
            },
            Edge {
                label: "edge_1",
                bounds: [0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e],
            },
            Edge {
                label: "edge_2",
                bounds: [0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e],
            },
        ],
    },
    Execution {
        config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
        expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
        id: "retained_knot_7_crossing",
        config: 6,
        source_group: "new_exact_point_controls",
        group: "db149f122f5f88267753b50d2f7e9635bdfdb0850608eb161eb0c8ec44c67358",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x401bffffffffffff, 0x401c000000000000],
            depth: Some([0x3fffffffffffffff, 0x4000000000000000]),
            velocity: Some([0xbac2492492492493, 0xbac2492492492492]),
            discharge: [0xbaf0000000000001, 0xbaf0000000000000],
            velocity_provenance: "direct",
        }],
        star: Some(State {
            label: "star",
            area: [0x401bffffffffffff, 0x401c000000000000],
            depth: None,
            velocity: Some([0xbac2492492492493, 0xbac2492492492492]),
            discharge: [0xbaf0000000000001, 0xbaf0000000000000],
            velocity_provenance: "direct",
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
                bounds: [0x3fefffffffffffff, 0x3ff0000000000000],
            },
        ],
    },
];
