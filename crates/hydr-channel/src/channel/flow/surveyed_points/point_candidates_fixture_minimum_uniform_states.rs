use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
        expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
        id: "pointed_minimum_uniform_moving",
        config: 6,
        source_group: "new_exact_point_controls",
        group: "8ee5e3a061304cf5952979a2ec1861e53678efc8fb092b52f7ae518eec2dfbec",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[
            State {
                label: "left-exterior",
                area: [0x0000000000000001, 0x0000000000000001],
                depth: Some([0x1e60000000000000, 0x1e60000000000000]),
                velocity: Some([0x4014000000000000, 0x4014000000000000]),
                discharge: [0x0000000000000005, 0x0000000000000005],
                velocity_provenance: "direct",
            },
            State {
                label: "left-exterior",
                area: [0x0000000000000001, 0x0000000000000001],
                depth: Some([0x1e60000000000000, 0x1e60000000000000]),
                velocity: Some([0x4014000000000000, 0x4014000000000000]),
                discharge: [0x0000000000000005, 0x0000000000000005],
                velocity_provenance: "direct",
            },
        ],
        star: Some(State {
            label: "star",
            area: [0x0000000000000001, 0x0000000000000001],
            depth: None,
            velocity: Some([0x4014000000000000, 0x4014000000000000]),
            discharge: [0x0000000000000005, 0x0000000000000005],
            velocity_provenance: "direct",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0x4013ffffffffffff, 0x4014000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x4013ffffffffffff, 0x4014000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0x4013ffffffffffff, 0x4014000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0x4014000000000000, 0x4014000000000001],
            },
            Edge {
                label: "edge_4",
                bounds: [0x4014000000000000, 0x4014000000000001],
            },
            Edge {
                label: "edge_5",
                bounds: [0x4014000000000000, 0x4014000000000001],
            },
        ],
    },
    Execution {
        config_identity: "26d1239de8e8f786b72c7d733c30c2b724869844c3279a3a5b02f17090436122",
        expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
        id: "low_only_rectangle_uniform_moving",
        config: 6,
        source_group: "new_exact_point_controls",
        group: "a43da1d07f4e06fdc968bcdd42c34050f6575c98d1846845dd7e0dc69ffc446f",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[
            State {
                label: "left-exterior",
                area: [0x0170000000000000, 0x0170000000000000],
                depth: Some([0x0170000000000000, 0x0170000000000000]),
                velocity: Some([0x4010000000000000, 0x4010000000000000]),
                discharge: [0x0190000000000000, 0x0190000000000000],
                velocity_provenance: "direct",
            },
            State {
                label: "left-exterior",
                area: [0x0170000000000000, 0x0170000000000000],
                depth: Some([0x0170000000000000, 0x0170000000000000]),
                velocity: Some([0x4010000000000000, 0x4010000000000000]),
                discharge: [0x0190000000000000, 0x0190000000000000],
                velocity_provenance: "direct",
            },
        ],
        star: Some(State {
            label: "star",
            area: [0x0170000000000000, 0x0170000000000000],
            depth: None,
            velocity: Some([0x4010000000000000, 0x4010000000000000]),
            discharge: [0x0190000000000000, 0x0190000000000000],
            velocity_provenance: "direct",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0x400fffffffffffff, 0x4010000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x400fffffffffffff, 0x4010000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0x400fffffffffffff, 0x4010000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0x4010000000000000, 0x4010000000000001],
            },
            Edge {
                label: "edge_4",
                bounds: [0x4010000000000000, 0x4010000000000001],
            },
            Edge {
                label: "edge_5",
                bounds: [0x4010000000000000, 0x4010000000000001],
            },
        ],
    },
];
