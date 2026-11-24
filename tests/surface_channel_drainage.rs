//! End-to-end 3-way coupling test: 2D surface + 1D river channel + pinned EPA SWMM v5.2.4 drainage.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[allow(dead_code)]
#[path = "surface_channels/fixture.rs"]
mod fixture;

#[allow(dead_code)]
#[path = "surface_channels/scenario.rs"]
mod scenario;

use hydr::channel::ChannelBoundary;
use hydr::coupling::{
    SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION, SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA,
    SurfaceChannelBatch, SurfaceChannelDrainageBatch, SurfaceChannelDrainageSetup,
    SurfaceChannelForcingInterval, SurfaceChannelReachForcing, SurfaceDrainageConfig,
    SurfaceDrainageInlet,
};
use hydr::drainage::{
    SwmmControlAuthority, SwmmDrainage, SwmmDrainageDefinition, SwmmSurchargeMethod,
};
use runtime::Runtime;
use scenario::{ReachSpec, Spec, weir};
use std::path::PathBuf;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const PROVENANCE: &str = "EPA SWMM v5.2.4 commit 7952ca837988b1c32f791812eccc9fd64547e093";
const SURCHARGE_EVIDENCE: &str = "3-way surface-channel-drainage verification under EXTRAN";

fn worker_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resource/reference-engines/swmm/build-drainage-worker/swmm-drainage-worker")
}

fn swmm_pipe_inp() -> String {
    r#"[TITLE]
Three-way coupled urban drainage test

[OPTIONS]
FLOW_UNITS           CFS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
START_DATE           01/01/2025
START_TIME           00:00:00
REPORT_START_DATE    01/01/2025
REPORT_START_TIME    00:00:00
END_DATE             01/01/2025
END_TIME             01:00:00
ROUTING_STEP         00:00:01
MINIMUM_STEP         0.5
VARIABLE_STEP        0.75
SURCHARGE_METHOD     EXTRAN

[JUNCTIONS]
;;Name  Elev    MaxDepth  InitDepth  SurDepth  Aponded
J1      26.2467 6.5617    0.0        0.0       0.0

[OUTFALLS]
;;Name  Elev    Type  Gated
OUT1    22.9659 FREE  NO

[CONDUITS]
;;Name  From    To      Length  Roughness  InOffset  OutOffset  InitFlow  MaxFlow
C1      J1      OUT1    100.0   0.013      0.0       0.0        0.0       0.0

[XSECTIONS]
;;Link  Shape     Geom1  Geom2  Geom3  Geom4  Barrels
C1      CIRCULAR  3.0    0      0      0      1

[REPORT]
INPUT      NO
CONTROLS   NO
SUBCATCHMENTS NONE
NODES      ALL
LINKS      ALL
"#
    .to_string()
}

fn channel_forcing(start_s: f64, end_s: f64) -> SurfaceChannelForcingInterval {
    SurfaceChannelForcingInterval {
        start_s,
        end_s,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: "river-1".to_string(),
            left: Some(ChannelBoundary::Wall),
            right: Some(ChannelBoundary::Wall),
            lateral_inputs: Vec::new(),
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff: Vec::new(),
    }
}

#[test]
fn surface_channel_drainage_three_way_exchange_rollback_and_restart() -> TestResult {
    let worker = worker_binary();
    if !worker.is_file() {
        eprintln!(
            "skipping live 3-way SWMM test: worker not built at {}",
            worker.display()
        );
        return Ok(());
    }

    // 3x3 closed floodplain (bed = 10.0 m, depth = 0.25 m -> stage 10.25 m) +
    // 1D river reach (bed = 9.0 m, depth = 2.2 m -> stage 11.2 m) spilling over a 10.5 m levee
    // into floodplain cell 0.
    let (setup_sc, initial_sc) = Spec::new(3, 3, 10.0, 0.25)
        .reach(ReachSpec::new("river-1", 9.0, &[2.2; 3]))
        .link(weir("levee-0", 0, "river-1", 1, 10.5, 5.0))
        .parts()?;
    let sc = SurfaceChannelBatch::<Runtime>::new(runtime::client(), setup_sc, initial_sc)?;

    let drainage_def = SwmmDrainageDefinition::new(
        swmm_pipe_inp(),
        "NAVD88",
        SwmmSurchargeMethod::Extran,
        SwmmControlAuthority::NativeSwmm,
        worker,
        PROVENANCE,
        SURCHARGE_EVIDENCE,
    )?;
    let drainage = SwmmDrainage::open(drainage_def)?;

    // Inlet at floodplain cell 4 (bed = 10.0 m, rim = 10.0 m) connected to SWMM junction J1 (invert = 8.0 m).
    let setup = SurfaceChannelDrainageSetup {
        vertical_datum: "NAVD88".to_string(),
        inlets: vec![SurfaceDrainageInlet {
            id: "IN1".to_string(),
            surface_cell: 4,
            drainage_node_id: "J1".to_string(),
            rim_elevation_m: 10.0,
            weir_crest_length_m: 2.0,
            weir_cd: 1.7,
            orifice_area_m2: 0.5,
            orifice_cd: 0.62,
            allow_surcharge_to_surface: true,
        }],
        control_claims: Vec::new(),
        config: SurfaceDrainageConfig {
            max_step_s: 0.5,
            min_step_s: 0.05,
            ..SurfaceDrainageConfig::default()
        },
    };
    let mut batch = SurfaceChannelDrainageBatch::new(sc, drainage, setup)?;

    // 1. Advance 2 steps: river spills overbank onto floodplain AND floodplain drains into SWMM sewer.
    let mut total_bank_from_river = 0.0_f64;
    let mut total_sewer_capture = 0.0_f64;
    for _ in 0..2 {
        let t0 = batch.time_s();
        let target = t0 + 2.0;
        let f = channel_forcing(t0, target);
        let diag = batch.trial_to(target, &f)?.accept();
        assert!(diag.end_s > diag.start_s);
        let bank_vol: f64 = diag
            .surface_channel
            .links
            .iter()
            .map(|l| l.signed_volume_m3)
            .sum();
        total_bank_from_river += -bank_vol;
        total_sewer_capture += diag.inlet_transfers[0].transfer_volume_m3;
    }
    assert!(
        total_bank_from_river > 0.1,
        "expected river overbank spill onto 2D surface, got {total_bank_from_river} m³"
    );
    assert!(
        total_sewer_capture > 0.1,
        "expected 2D surface capture into SWMM sewer, got {total_sewer_capture} m³"
    );

    // 2. Transactional rollback: drop a trial without accepting and verify bit-identical state.
    let before_time = batch.time_s();
    let before_j1_head = batch.drainage().state().nodes[0].head_m;
    let before_surface_h: Vec<f64> = batch
        .surface_channels()
        .surface()
        .state()?
        .cells()
        .iter()
        .map(|c| c.h)
        .collect();
    {
        let f = channel_forcing(before_time, before_time + 2.0);
        let trial = batch.trial_to(before_time + 2.0, &f)?;
        assert!(trial.diagnostics().end_s > before_time);
    }
    assert_eq!(batch.time_s(), before_time);
    assert_eq!(batch.drainage().state().nodes[0].head_m, before_j1_head);
    let after_drop_h: Vec<f64> = batch
        .surface_channels()
        .surface()
        .state()?
        .cells()
        .iter()
        .map(|c| c.h)
        .collect();
    assert_eq!(after_drop_h, before_surface_h);

    // 3. Atomic 3-way checkpoint & restart.
    let snap = batch.snapshot_data()?;
    assert_eq!(snap.schema, SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA);
    assert_eq!(snap.contract, SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION);

    let f_next = channel_forcing(batch.time_s(), batch.time_s() + 2.0);
    let diag_ref = batch.trial_to(batch.time_s() + 2.0, &f_next)?.accept();

    batch.import_snapshot(runtime::client(), &snap)?;
    assert_eq!(batch.time_s(), before_time);
    let diag_replayed = batch.trial_to(batch.time_s() + 2.0, &f_next)?.accept();
    assert!((diag_replayed.end_s - diag_ref.end_s).abs() < 1e-12);
    assert!(
        (diag_replayed.inlet_transfers[0].transfer_volume_m3
            - diag_ref.inlet_transfers[0].transfer_volume_m3)
            .abs()
            < 1e-6
    );
    Ok(())
}
