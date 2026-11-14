use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[
    Execution {
        config_identity: "fabf1a89754bae85ce604602211d2046fdc41970b627ec62100c86352e026964",
        expected_identity: "ae36539d917e8545987274cc7f14f560a7cfabd5d15615fad868fb6cdec23a1e",
        id: "scalar_rectangle_uniform_rest",
        config: 0,
        source_group: "existing_scalar_37",
        group: "365b9b0b67682e1e3663d7086275df84a922bc361e48fd955d4b92838bb0508f",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x3ff0000000000000, 0x3ff0000000000000],
            depth: Some([0x3ff0000000000000, 0x3ff0000000000000]),
            velocity: Some([0x0000000000000000, 0x0000000000000000]),
            discharge: [0x0000000000000000, 0x0000000000000000],
            velocity_provenance: "Q_over_A",
        }],
        star: Some(State {
            label: "star",
            area: [0x3ff0000000000000, 0x3ff0000000000000],
            depth: Some([0x3ff0000000000000, 0x3ff0000000000000]),
            velocity: Some([0x0000000000000000, 0x0000000000000000]),
            discharge: [0x0000000000000000, 0x0000000000000000],
            velocity_provenance: "Q_over_A",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xbff0000000000000, 0xbff0000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0xbff0000000000000, 0xbff0000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0xbff0000000000000, 0xbff0000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0x3ff0000000000000, 0x3ff0000000000000],
            },
            Edge {
                label: "edge_4",
                bounds: [0x3ff0000000000000, 0x3ff0000000000000],
            },
            Edge {
                label: "edge_5",
                bounds: [0x3ff0000000000000, 0x3ff0000000000000],
            },
        ],
    },
    Execution {
        config_identity: "90a434772d282e704918337f86b9c7add40257a19ff151b300cb8f0a887e149e",
        expected_identity: "3c6f701154510656b9630baaf04d97eb19b46b0d3cc6d01d4bc689b90d99a454",
        id: "scalar_rectangle_uniform_forward",
        config: 0,
        source_group: "existing_scalar_37",
        group: "7f11aad6e34c45ed67603e10f9c30453d66982ecb9d11f56aef07641251c19e2",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
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
                label: "left-exterior",
                area: [0x3ff0000000000000, 0x3ff0000000000000],
                depth: Some([0x3ff0000000000000, 0x3ff0000000000000]),
                velocity: Some([0x4008000000000000, 0x4008000000000000]),
                discharge: [0x4008000000000000, 0x4008000000000000],
                velocity_provenance: "Q_over_A",
            },
        ],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0x4000000000000000, 0x4000000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0x4000000000000000, 0x4000000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0x4000000000000000, 0x4000000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0x4010000000000000, 0x4010000000000000],
            },
            Edge {
                label: "edge_4",
                bounds: [0x4010000000000000, 0x4010000000000000],
            },
            Edge {
                label: "edge_5",
                bounds: [0x4010000000000000, 0x4010000000000000],
            },
        ],
    },
    Execution {
        config_identity: "f4dffc0a94f3514e36c51f8cab8936e858bbddf4ec35efb6dbd0c209e7627cd8",
        expected_identity: "92b8aff7eabb4a75010d82e4d4b25c07675cd083e25398d06cf2d73f19f39bee",
        id: "scalar_rectangle_uniform_reverse",
        config: 0,
        source_group: "existing_scalar_37",
        group: "3968b19df9eac044fcf6dd53ec687125d0bb8d026a729f8f5332a21daeafc7d3",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
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
                label: "right-exterior",
                area: [0x3ff0000000000000, 0x3ff0000000000000],
                depth: Some([0x3ff0000000000000, 0x3ff0000000000000]),
                velocity: Some([0xc008000000000000, 0xc008000000000000]),
                discharge: [0xc008000000000000, 0xc008000000000000],
                velocity_provenance: "Q_over_A",
            },
        ],
        star: None,
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc010000000000000, 0xc010000000000000],
            },
            Edge {
                label: "edge_1",
                bounds: [0xc010000000000000, 0xc010000000000000],
            },
            Edge {
                label: "edge_2",
                bounds: [0xc010000000000000, 0xc010000000000000],
            },
            Edge {
                label: "edge_3",
                bounds: [0xc000000000000000, 0xc000000000000000],
            },
            Edge {
                label: "edge_4",
                bounds: [0xc000000000000000, 0xc000000000000000],
            },
            Edge {
                label: "edge_5",
                bounds: [0xc000000000000000, 0xc000000000000000],
            },
        ],
    },
    Execution {
        config_identity: "b998d70f595664f307d166a2ba665cda88795d6d578075461db5454f57739e85",
        expected_identity: "c253bf0147457d1fa98812e5e6185564f96e9963e262ae91b2d120790ade6e79",
        id: "scalar_triangle_uniform_flow",
        config: 1,
        source_group: "existing_scalar_37",
        group: "e3c87f6bb276701d642ee807a2113d7d964370d147559649a77770c098c5e7e9",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x4010000000000000, 0x4010000000000000],
            depth: Some([0x4000000000000000, 0x4000000000000000]),
            velocity: Some([0x3fe0000000000000, 0x3fe0000000000000]),
            discharge: [0x4000000000000000, 0x4000000000000000],
            velocity_provenance: "Q_over_A",
        }],
        star: Some(State {
            label: "star",
            area: [0x4010000000000000, 0x4010000000000000],
            depth: Some([0x4000000000000000, 0x4000000000000000]),
            velocity: Some([0x3fe0000000000000, 0x3fe0000000000000]),
            discharge: [0x4000000000000000, 0x4000000000000000],
            velocity_provenance: "Q_over_A",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc0050e8639c65e42, 0xc0050e8639c65e41],
            },
            Edge {
                label: "edge_1",
                bounds: [0xc0050e8639c65e42, 0xc0050e8639c65e41],
            },
            Edge {
                label: "edge_2",
                bounds: [0xc0050e8639c65e42, 0xc0050e8639c65e41],
            },
            Edge {
                label: "edge_3",
                bounds: [0x400d0e8639c65e41, 0x400d0e8639c65e42],
            },
            Edge {
                label: "edge_4",
                bounds: [0x400d0e8639c65e41, 0x400d0e8639c65e42],
            },
            Edge {
                label: "edge_5",
                bounds: [0x400d0e8639c65e41, 0x400d0e8639c65e42],
            },
        ],
    },
    Execution {
        config_identity: "6525d640f1773f214e2e88642ad3eb3d3ea5c16d012e9f799040e68d1f6daec0",
        expected_identity: "818146a8a0ea28d63268f02c865125493a4338bf7912eafdcbbfae0141812cc2",
        id: "scalar_trapezoid_uniform_flow",
        config: 1,
        source_group: "existing_scalar_37",
        group: "f39a6718cba34bfed2697388e7e41cc17c95f0f12369eecbb1298197dea20275",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x4018000000000000, 0x4018000000000000],
            depth: Some([0x4000000000000000, 0x4000000000000000]),
            velocity: Some([0x3fe0000000000000, 0x3fe0000000000000]),
            discharge: [0x4008000000000000, 0x4008000000000000],
            velocity_provenance: "Q_over_A",
        }],
        star: Some(State {
            label: "star",
            area: [0x4018000000000000, 0x4018000000000000],
            depth: Some([0x4000000000000000, 0x4000000000000000]),
            velocity: Some([0x3fe0000000000000, 0x3fe0000000000000]),
            discharge: [0x4008000000000000, 0x4008000000000000],
            velocity_provenance: "Q_over_A",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc00772c2637ae566, 0xc00772c2637ae565],
            },
            Edge {
                label: "edge_1",
                bounds: [0xc00772c2637ae566, 0xc00772c2637ae565],
            },
            Edge {
                label: "edge_2",
                bounds: [0xc00772c2637ae566, 0xc00772c2637ae565],
            },
            Edge {
                label: "edge_3",
                bounds: [0x400f72c2637ae565, 0x400f72c2637ae566],
            },
            Edge {
                label: "edge_4",
                bounds: [0x400f72c2637ae565, 0x400f72c2637ae566],
            },
            Edge {
                label: "edge_5",
                bounds: [0x400f72c2637ae565, 0x400f72c2637ae566],
            },
        ],
    },
    Execution {
        config_identity: "6a77ad8623e7338be3485da33ba2d80d53ad3ca1ab424a1bd4c235d8cbf834b8",
        expected_identity: "612406a129abe1007159954bcdc4e21f4daad3550a9870c5abbdbe68997b115d",
        id: "scalar_three_bands_uniform_upper",
        config: 1,
        source_group: "existing_scalar_37",
        group: "ec6890537fc7135156eeac25f9600ad9b509379a9c555ef597af05009e71d700",
        declared: "wet_wet",
        outcome: "resolved",
        physical: "wet_wet",
        traces: &[State {
            label: "star",
            area: [0x4028000000000000, 0x4028000000000000],
            depth: Some([0x400572fc8ab73db6, 0x400572fc8ab73db7]),
            velocity: Some([0x3fd0000000000000, 0x3fd0000000000000]),
            discharge: [0x4008000000000000, 0x4008000000000000],
            velocity_provenance: "Q_over_A",
        }],
        star: Some(State {
            label: "star",
            area: [0x4028000000000000, 0x4028000000000000],
            depth: Some([0x400572fc8ab73db6, 0x400572fc8ab73db7]),
            velocity: Some([0x3fd0000000000000, 0x3fd0000000000000]),
            discharge: [0x4008000000000000, 0x4008000000000000],
            velocity_provenance: "Q_over_A",
        }),
        edges: &[
            Edge {
                label: "edge_0",
                bounds: [0xc00d518eafa820de, 0xc00d518eafa820dd],
            },
            Edge {
                label: "edge_1",
                bounds: [0xc00d518eafa820de, 0xc00d518eafa820dd],
            },
            Edge {
                label: "edge_2",
                bounds: [0xc00d518eafa820de, 0xc00d518eafa820dd],
            },
            Edge {
                label: "edge_3",
                bounds: [0x4010a8c757d4106e, 0x4010a8c757d4106f],
            },
            Edge {
                label: "edge_4",
                bounds: [0x4010a8c757d4106e, 0x4010a8c757d4106f],
            },
            Edge {
                label: "edge_5",
                bounds: [0x4010a8c757d4106e, 0x4010a8c757d4106f],
            },
        ],
    },
];
