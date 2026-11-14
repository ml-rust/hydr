use super::super::fixture_model::{Edge, Execution, State};
pub(super) const ROWS: &[Execution] = &[Execution {
    config_identity: "e6096a4fad707b1508ea5faa6b7b78845b95d8c4894f4c5ab193ee29a5cab613",
    expected_identity: "2916ea9be525ec4883800ba1541c4074e233bcbab656b310f4f96e8c342332f4",
    id: "captured_tiny_upwind_face_64",
    config: 7,
    source_group: "new_exact_point_controls",
    group: "1d2b328ef55c55bdcb3d90b0d268669579b60f2f5145a5ee70b1c697e7476f3f",
    declared: "wet_wet",
    outcome: "resolved",
    physical: "wet_wet",
    traces: &[State {
        label: "left-exterior",
        area: [0x399777f21e1a835c, 0x399777f21e1a835c],
        depth: Some([0x399777f21e1a835b, 0x399777f21e1a835c]),
        velocity: Some([0x401423734b85b73c, 0x401423734b85b73d]),
        discharge: [0x39bd89ee2425974f, 0x39bd89ee2425974f],
        velocity_provenance: "direct",
    }],
    star: Some(State {
        label: "star",
        area: [0x3bc280007c90d340, 0x3bc280007c90d341],
        depth: None,
        velocity: Some([0x4014236ad65ed986, 0x4014236ad65ed987]),
        discharge: [0x3be748f424a67026, 0x3be748f424a67027],
        velocity_provenance: "direct",
    }),
    edges: &[
        Edge {
            label: "edge_0",
            bounds: [0x4014236ad65ed985, 0x4014236ad65ed986],
        },
        Edge {
            label: "edge_1",
            bounds: [0x4014236ad65ed986, 0x4014236ad65ed987],
        },
    ],
}];
