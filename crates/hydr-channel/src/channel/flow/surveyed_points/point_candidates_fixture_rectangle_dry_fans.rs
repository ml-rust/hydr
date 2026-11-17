use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "84d0b32fb26ef19d3c1971c8a15c00aee19dd38d4b31f6262e1a86346256b750",
        expected_identity: "8468514dc33ff17802eb8acb7ddc9cabec782b01c89cd7b454677384eb34e485",
        id: "scalar_rectangle_right_dry_sonic",
        config: 0,
        source_group: "existing_scalar_37",
        group: "2f2969cc963fe64e0a6edf76693afc850e617d9da3e5f8ab1daf44d6b3ad2c0e",
        declared: "right_dry",
        outcome: "resolved",
        physical: "right_dry",
        traces: &[State {
            label: "left-sonic",
            area: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
            depth: Some([0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d]),
            velocity: Some([0x3fe5555555555555, 0x3fe5555555555556]),
            discharge: [0x3fd2f684bda12f68, 0x3fd2f684bda12f69],
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
                bounds: [0x4000000000000000, 0x4000000000000000],
            },
        ],
    },
    Execution {
        config_identity: "84905a70ddb89b5ebae33398a170a97bc9cd8da26fba52067c603371a4d54bcb",
        expected_identity: "227c5de0b4fca640c1fc77871524f01943eeab8c4a1cb9222b3bc200484ed8f0",
        id: "scalar_rectangle_left_dry_sonic",
        config: 0,
        source_group: "existing_scalar_37",
        group: "8d5358737d1a189b47e58b9db7a33e6471fc5322b27eb6943a6d1bebd3d3f92d",
        declared: "left_dry",
        outcome: "resolved",
        physical: "left_dry",
        traces: &[State {
            label: "right-sonic",
            area: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
            depth: Some([0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d]),
            velocity: Some([0xbfe5555555555556, 0xbfe5555555555555]),
            discharge: [0xbfd2f684bda12f69, 0xbfd2f684bda12f68],
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
                bounds: [0xc000000000000000, 0xc000000000000000],
            },
        ],
    },
];
