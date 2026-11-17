use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "c54190676f7f901fc9dfc866fdc36c707ff090830e3b9fbf445137d50d84d4be",
        expected_identity: "104ee14240ce25ffb9ea4a2518dd42eba313df54ba8a70f12f959c21653d6257",
        id: "scalar_rectangle_vacuum",
        config: 0,
        source_group: "existing_scalar_37",
        group: "425ba4c96a1b7b34e65003d4b917165390c44af74b68fbbe7c6029019cdd1f31",
        declared: "vacuum",
        outcome: "resolved",
        physical: "vacuum",
        traces: &[State {
            label: "vacuum",
            area: [0x0000000000000000, 0x0000000000000000],
            depth: Some([0x0000000000000000, 0x0000000000000000]),
            velocity: None,
            discharge: [0x0000000000000000, 0x0000000000000000],
            velocity_provenance: "none",
        }],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc010000000000000, 0xc010000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0xbff0000000000000, 0xbff0000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0x4010000000000000, 0x4010000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0x3ff0000000000000, 0x3ff0000000000000],
            },
        ],
    },
    Execution {
        config_identity: "ae466aaa46eea05f3ac57a098770efa531b11dd58d8b999a33beb21f602a3ec7",
        expected_identity: "d02a32711768714022fb3a970a7c73edd229b51b658d7ad393af0055c927bc7a",
        id: "scalar_rectangle_vacuum_threshold",
        config: 0,
        source_group: "existing_scalar_37",
        group: "67b657e0a5704ec19c1d034b601929679d6425503b6054e67367aad234695036",
        declared: "vacuum",
        outcome: "resolved",
        physical: "vacuum",
        traces: &[
            State {
                label: "sonic-dry-limit",
                area: [0x0000000000000000, 0x0000000000000000],
                depth: Some([0x0000000000000000, 0x0000000000000000]),
                velocity: None,
                discharge: [0x0000000000000000, 0x0000000000000000],
                velocity_provenance: "none",
            },
            State {
                label: "sonic-dry-limit",
                area: [0x0000000000000000, 0x0000000000000000],
                depth: Some([0x0000000000000000, 0x0000000000000000]),
                velocity: None,
                discharge: [0x0000000000000000, 0x0000000000000000],
                velocity_provenance: "none",
            },
            State {
                label: "vacuum",
                area: [0x0000000000000000, 0x0000000000000000],
                depth: Some([0x0000000000000000, 0x0000000000000000]),
                velocity: None,
                discharge: [0x0000000000000000, 0x0000000000000000],
                velocity_provenance: "none",
            },
        ],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc008000000000000, 0xc008000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x0000000000000000, 0x0000000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0x4008000000000000, 0x4008000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0x0000000000000000, 0x0000000000000000],
            },
        ],
    },
    Execution {
        config_identity: "e0fb7497f60eedf14e6cdbf7a747754b272684c1cc16e8e51367dcbb88c29b62",
        expected_identity: "95b18d710f9988462c113b20118a7f2109682c5a71c4b8e3af489ebfc8e76b3c",
        id: "scalar_rectangle_vacuum_threshold_outside",
        config: 0,
        source_group: "existing_scalar_37",
        group: "e08fbc49546b5dc9102bb1451bd1fc782a8b4c9cf23ea52161eca4d4c9279ef7",
        declared: "vacuum",
        outcome: "resolved",
        physical: "vacuum",
        traces: &[
            State {
                label: "sonic-dry-limit",
                area: [0x0000000000000000, 0x0000000000000000],
                depth: Some([0x0000000000000000, 0x0000000000000000]),
                velocity: None,
                discharge: [0x0000000000000000, 0x0000000000000000],
                velocity_provenance: "none",
            },
            State {
                label: "vacuum",
                area: [0x0000000000000000, 0x0000000000000000],
                depth: Some([0x0000000000000000, 0x0000000000000000]),
                velocity: None,
                discharge: [0x0000000000000000, 0x0000000000000000],
                velocity_provenance: "none",
            },
        ],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc008000000000000, 0xc008000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x0000000000000000, 0x0000000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0x4008000000000001, 0x4008000000000001],
            },
            Edge {
                label: "edge_3",
                bounds: [0x3cc0000000000000, 0x3cc0000000000000],
            },
        ],
    },
    Execution {
        config_identity: "b97ef2ec51b1fc4a9c1404a8e292266835dc8c73829b799e057db7a4df7cf24c",
        expected_identity: "2281dab56bcc7c9ff79cf142d5a737392e9d43c1e2855bde5b9f5644aea484d7",
        id: "scalar_rectangle_vacuum_threshold_inside",
        config: 0,
        source_group: "existing_scalar_37",
        group: "7520fee02ddece3836987073d85a3b10d26ded7e602496d13d13b967156fc13d",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "right-sonic",
            area: [0x393c71c71c71c71c, 0x393c71c71c71c71d],
            depth: Some([0x393c71c71c71c71c, 0x393c71c71c71c71d]),
            velocity: Some([0xbc95555555555556, 0xbc95555555555555]),
            discharge: [0xb5e2f684bda12f69, 0xb5e2f684bda12f68],
            velocity_provenance: "Q_over_A",
        }],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc008000000000000, 0xc008000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0xbca8000000000000, 0xbca8000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0x4007ffffffffffff, 0x4008000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0xbc90000000000000, 0xbc90000000000000],
            },
        ],
    },
];
