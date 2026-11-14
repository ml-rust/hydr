use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "65723f34802ccecde57761cdbd4b04e954748ce360a1ea65e2b5dca293a57b81",
        expected_identity: "24be30194d6ecdefd652b39f2bc9044fc3460205f98514f8ce37fdc772137b89",
        id: "scalar_rectangle_exact_wet_pure_fan",
        config: 0,
        source_group: "existing_scalar_37",
        group: "d50c5ef29bac9e6b3bbe1581f6fba06c9405eb8f5dfaed47345c6eb7261c98e5",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "left-sonic",
            area: [0x3ffc71c71c71c71c, 0x3ffc71c71c71c71d],
            depth: Some([0x3ffc71c71c71c71c, 0x3ffc71c71c71c71d]),
            velocity: Some([0x3ff5555555555555, 0x3ff5555555555556]),
            discharge: [0x4002f684bda12f68, 0x4002f684bda12f69],
            velocity_provenance: "Q_over_A",
        }],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc000000000000000, 0xc000000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x3ff0000000000000, 0x3ff0000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0x4008000000000000, 0x4008000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0x4008000000000000, 0x4008000000000000],
            },
            Edge {
                label: "edge_4",
                bounds: [0x4008000000000000, 0x4008000000000000],
            },
        ],
    },
    Execution {
        config_identity: "31c096aec2509629afc960d52743013bcfa9a96c7c2d8f1c263be32665457fd0",
        expected_identity: "f70fedb03774458bbe8b12cac3a1ba678912894754da85b52411fb2d1e4b61f5",
        id: "scalar_planned_three_bands_both_knots_near_pure_fan",
        config: 0,
        source_group: "existing_scalar_37",
        group: "7c2f64d6aa18b17011ff297b2a46b5b8f581651880f54eb40d0b3a849f5ac5dc",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "left-sonic",
            area: [0x4016ccf1ffa9e4d9, 0x4016ccf1ffa9e4da],
            depth: Some([0x3ffa0446cb515322, 0x3ffa0446cb515323]),
            velocity: Some([0x3ff1c2b77c2e69b2, 0x3ff1c2b77c2e69b3]),
            discharge: [0x40194f3b8ee6c299, 0x40194f3b8ee6c29a],
            velocity_provenance: "Q_over_A",
        }],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xbff52a7fa9d2f8ea, 0xbff52a7fa9d2f8e9],
            },
            Edge {
                label: "edge_1",
                bounds: [0x3fff1ffe8c701057, 0x3fff1ffe8c701058],
            },
            Edge {
                label: "edge_2",
                bounds: [0x4009137e4b50e3b6, 0x4009137e4b50e3b7],
            },
            Edge {
                label: "edge_3",
                bounds: [0x4009137e4b50e3b6, 0x4009137e4b50e3b7],
            },
        ],
    },
];
