//! Tests for the pure-Rust 1D dynamic-wave Preissmann-slot pressurized-pipe solver.

use hydr::drainage::{
    NATIVE_PIPE_CONTRACT_VERSION, NATIVE_PIPE_SNAPSHOT_SCHEMA, NativePipeConfig, NativePipeNetwork,
    NativePipeNetworkSpec, PipeConduitSpec, PipeCrossSection, PipeNodeKind, PipeNodeSpec,
    PipeOutfallBoundary,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn two_conduit_network() -> NativePipeNetworkSpec {
    NativePipeNetworkSpec {
        vertical_datum: "NAVD88".to_string(),
        nodes: vec![
            PipeNodeSpec {
                id: "J1".to_string(),
                invert_m: 10.0,
                initial_depth_m: 0.0,
                kind: PipeNodeKind::Junction {
                    max_depth_m: 2.0,
                    surcharge_depth_m: 3.0,
                    shaft_area_m2: 2.0,
                    ponding_area_m2: 0.0,
                },
            },
            PipeNodeSpec {
                id: "J2".to_string(),
                invert_m: 9.5,
                initial_depth_m: 0.0,
                kind: PipeNodeKind::Junction {
                    max_depth_m: 2.0,
                    surcharge_depth_m: 3.0,
                    shaft_area_m2: 2.0,
                    ponding_area_m2: 0.0,
                },
            },
            PipeNodeSpec {
                id: "OUT1".to_string(),
                invert_m: 9.0,
                initial_depth_m: 0.0,
                kind: PipeNodeKind::Outfall {
                    boundary: PipeOutfallBoundary::Free,
                },
            },
        ],
        conduits: vec![
            PipeConduitSpec {
                id: "C1".to_string(),
                from_node: "J1".to_string(),
                to_node: "J2".to_string(),
                length_m: 50.0,
                section: PipeCrossSection::Circular { diameter_m: 0.6 },
                manning_n: 0.013,
                in_offset_m: 0.0,
                out_offset_m: 0.0,
                initial_flow_m3_per_s: 0.0,
            },
            PipeConduitSpec {
                id: "C2".to_string(),
                from_node: "J2".to_string(),
                to_node: "OUT1".to_string(),
                length_m: 50.0,
                section: PipeCrossSection::Box {
                    width_m: 0.6,
                    height_m: 0.6,
                },
                manning_n: 0.013,
                in_offset_m: 0.0,
                out_offset_m: 0.0,
                initial_flow_m3_per_s: 0.0,
            },
        ],
        config: NativePipeConfig {
            max_substep_s: 0.25,
            ..NativePipeConfig::default()
        },
    }
}

#[test]
fn native_pipe_gravity_surcharge_control_rollback_and_restart() -> TestResult {
    let mut net = NativePipeNetwork::new(two_conduit_network())?;
    let j1 = net.node_index("J1").ok_or("missing J1")?;
    let c1 = net.conduit_index("C1").ok_or("missing C1")?;

    // 1. Gravity flow under modest inflow (0.20 m³/s) -> positive conduit flow & outfall outflow.
    let s1 = net.trial_to(90.0, &[(j1, 0.20)], &[])?.accept();
    assert!(s1.conduits[0].flow_m3_per_s > 0.05);
    assert!(s1.conduits[1].flow_m3_per_s > 0.01);
    assert!(s1.ledger.outfall_outflow_m3 > 0.1);

    // 2. Heavy storm inflow (1.2 m³/s) -> drives J1 & C1 into Preissmann-slot pressurized surcharge.
    let s2 = net.trial_to(150.0, &[(j1, 1.2)], &[])?.accept();
    assert!(
        s2.nodes[0].surcharged && s2.conduits[0].pressurized,
        "expected pressurized surcharge above 0.6 m crown, got J1 depth={} m, C1 depth={} m",
        s2.nodes[0].depth_m,
        s2.conduits[0].depth_m
    );

    // 3. Link control actuation: close C1 (setting = 0.0) -> flow drops to zero.
    let s3 = net.trial_to(160.0, &[(j1, 0.2)], &[(c1, 0.0)])?.accept();
    assert_eq!(s3.conduits[0].flow_m3_per_s, 0.0);
    assert_eq!(s3.conduits[0].setting, 0.0);

    // 4. Transactional rollback: drop trial without accepting -> bit-identical state preserved.
    let before = net.state();
    {
        let trial = net.trial_to(170.0, &[(j1, 0.5)], &[(c1, 1.0)])?;
        assert!(trial.state().conduits[0].flow_m3_per_s > 0.0);
    }
    assert_eq!(net.state(), before);

    // 5. Portable checkpoint & bit-identical restart.
    let snap = net.snapshot_data();
    assert_eq!(snap.schema, NATIVE_PIPE_SNAPSHOT_SCHEMA);
    assert_eq!(snap.contract, NATIVE_PIPE_CONTRACT_VERSION);

    let ref_next = net.trial_to(180.0, &[(j1, 0.3)], &[(c1, 1.0)])?.accept();
    net.import_snapshot(&snap)?;
    let replayed = net.trial_to(180.0, &[(j1, 0.3)], &[(c1, 1.0)])?.accept();
    assert_eq!(replayed, ref_next);
    Ok(())
}
