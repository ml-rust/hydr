//! Capability declaration of `NativePipeNetwork` checked against admission.

use hydr::capability::{ComponentId, UnsupportedReason};
use hydr::drainage::{
    NativePipeConfig, NativePipeNetwork, NativePipeNetworkSpec, PipeConduitSpec, PipeCrossSection,
    PipeNodeKind, PipeNodeSpec, PipeOutfallBoundary, native_pipe_drainage_capabilities,
};

#[test]
fn native_pipe_drainage_declaration_matches_admission() -> Result<(), Box<dyn std::error::Error>> {
    let caps = native_pipe_drainage_capabilities();
    assert_eq!(caps.component, ComponentId::NativePipeDrainage);

    let rejected_ids: Vec<&str> = caps
        .unsupported
        .iter()
        .filter(|u| u.reason == UnsupportedReason::RejectedByAdmission)
        .map(|u| u.id)
        .collect();
    assert_eq!(
        rejected_ids,
        ["invalid-pipe-network", "tampered-pipe-snapshot"]
    );

    // Self-loop conduit rejects.
    let bad = NativePipeNetwork::new(NativePipeNetworkSpec {
        vertical_datum: "NAVD88".to_string(),
        nodes: vec![PipeNodeSpec {
            id: "J1".to_string(),
            invert_m: 10.0,
            initial_depth_m: 0.0,
            kind: PipeNodeKind::Outfall {
                boundary: PipeOutfallBoundary::Free,
            },
        }],
        conduits: vec![PipeConduitSpec {
            id: "C1".to_string(),
            from_node: "J1".to_string(),
            to_node: "J1".to_string(),
            length_m: 50.0,
            section: PipeCrossSection::Circular { diameter_m: 1.0 },
            manning_n: 0.013,
            in_offset_m: 0.0,
            out_offset_m: 0.0,
            initial_flow_m3_per_s: 0.0,
        }],
        config: NativePipeConfig::default(),
    });
    assert!(bad.is_err());

    // Tampered snapshot rejects.
    let mut net = NativePipeNetwork::new(NativePipeNetworkSpec {
        vertical_datum: "NAVD88".to_string(),
        nodes: vec![PipeNodeSpec {
            id: "OUT".to_string(),
            invert_m: 0.0,
            initial_depth_m: 0.0,
            kind: PipeNodeKind::Outfall {
                boundary: PipeOutfallBoundary::Free,
            },
        }],
        conduits: Vec::new(),
        config: NativePipeConfig::default(),
    })?;
    let mut snap = net.snapshot_data();
    snap.schema = "tampered".to_string();
    assert!(net.import_snapshot(&snap).is_err());
    Ok(())
}
