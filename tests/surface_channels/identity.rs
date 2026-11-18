//! Package rejections name the foreign or forged field and leave no batch behind.

use crate::fixture::{Result, contains};
use crate::restoration::{network, run, spill};
use crate::scenario::{Batch, Spec, step};
use hydr::coupling::{
    SurfaceChannelError, SurfaceChannelIntervalSnapshotData, SurfaceChannelSnapshotData,
};
use hydr::model::FlowState;

/// Package after two accepted intervals, with exchange through every link.
fn package() -> Result<SurfaceChannelSnapshotData> {
    Ok(run(network().build()?, &[0.25, 0.5])?.snapshot_data()?)
}

/// Mutable last interval of `data`.
fn last_interval(
    data: &mut SurfaceChannelSnapshotData,
) -> Result<&mut SurfaceChannelIntervalSnapshotData> {
    Ok(data
        .last_interval
        .as_mut()
        .ok_or("the package lacks its last interval")?)
}

/// Input messages of import into a fresh network batch and of resume, in that order.
fn rejections(data: &SurfaceChannelSnapshotData) -> Result<[String; 2]> {
    rejections_from(&network(), data)
}

/// Input messages of import into a fresh `spec` batch and of resume, in that order.
fn rejections_from(spec: &Spec, data: &SurfaceChannelSnapshotData) -> Result<[String; 2]> {
    let message = |result: std::result::Result<Batch, SurfaceChannelError>| match result {
        Ok(_) => Err("the coordinator accepted a forged package".into()),
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    };
    let imported: Result<String> = message(spec.build()?.import_snapshot(data));
    let resumed: Result<String> = message(Batch::resume(
        crate::runtime::client(),
        data,
        data.surface.setup.forcing.clone(),
    ));
    Ok([imported?, resumed?])
}

/// Fails unless import and resume both reject `data` with every fragment.
fn rejects(data: &SurfaceChannelSnapshotData, fragments: &[&str]) -> Result {
    for message in rejections(data)? {
        contains(&message, fragments)?;
    }
    Ok(())
}

#[test]
fn foreign_schema_contract_and_nested_identity_reject_with_path() -> Result {
    let data = package()?;
    let mut forged = data.clone();
    forged.schema = "hydr-surface-channel-checkpoint-0".into();
    rejects(
        &forged,
        &["surface-channel checkpoint schema", "checkpoint-0"],
    )?;
    let mut forged = data.clone();
    forged.contract = "hydr-surface-channel-numr-cuda-1".into();
    rejects(
        &forged,
        &["surface-channel checkpoint contract", "numr-cuda-1"],
    )?;
    let mut forged = data.clone();
    forged.surface.schema = "hydr-checkpoint-2".into();
    rejects(
        &forged,
        &[
            "surface-channel checkpoint surface:",
            "surface solver checkpoint schema",
        ],
    )?;
    let mut forged = data.clone();
    forged.reaches[1].1.contract = "foreign".into();
    rejects(
        &forged,
        &[
            "surface-channel checkpoint reaches[1] (\"south\")",
            "channel checkpoint contract",
        ],
    )?;
    let mut forged = data.clone();
    let last = last_interval(&mut forged)?;
    last.previous.surface.contract = "foreign".into();
    rejects(
        &forged,
        &[
            "surface-channel checkpoint last_interval.previous.surface:",
            "surface solver checkpoint contract",
        ],
    )?;
    Ok(())
}

#[test]
fn forged_initial_surface_state_rejects() -> Result {
    let data = package()?;
    let mut forged = data.clone();
    let mut cells = forged.initial.surface.flow.cells().to_vec();
    cells[0].h += 1e-3;
    forged.initial.surface.flow = FlowState::new(&data.surface.setup.domain, cells)?;
    let [imported, resumed] = rejections(&forged)?;
    contains(
        &imported,
        &["initial.surface", "destination original state"],
    )?;
    contains(
        &resumed,
        &["initial.surface", "surface.ledger.initial_storage"],
    )?;
    // Momentum leaves storage unchanged, so only the zero-history state comparison sees it.
    let zero = network().build()?.snapshot_data()?;
    let mut forged = zero.clone();
    let mut cells = forged.initial.surface.flow.cells().to_vec();
    cells[0].qx = 1e-3;
    forged.initial.surface.flow = FlowState::new(&zero.surface.setup.domain, cells)?;
    let [imported, resumed] = rejections(&forged)?;
    contains(
        &imported,
        &["initial.surface", "destination original state"],
    )?;
    contains(
        &resumed,
        &["initial.surface", "zero-history surface package"],
    )?;
    Ok(())
}

#[test]
fn forged_last_interval_transfer_rejects() -> Result {
    let data = package()?;
    let mut forged = data.clone();
    let last = last_interval(&mut forged)?;
    let link = last.diagnostics.links.first_mut().ok_or("missing link")?;
    assert!(
        link.signed_volume_m3 != 0.0,
        "the forged link carries water"
    );
    link.signed_volume_m3 = link.signed_volume_m3.next_up();
    rejects(
        &forged,
        &[
            "last_interval.diagnostics.links[0] (\"n-0\").signed_volume_m3",
            "recalculated last interval",
        ],
    )?;
    let mut forged = data.clone();
    let last = last_interval(&mut forged)?;
    let link = last.diagnostics.links.get_mut(2).ok_or("missing link")?;
    link.midpoint_channel_stage_m = link.midpoint_channel_stage_m.next_down();
    rejects(
        &forged,
        &["last_interval.diagnostics.links[2] (\"s-0\").midpoint_channel_stage_m"],
    )?;
    let mut forged = data.clone();
    let last = last_interval(&mut forged)?;
    last.forcing
        .reaches
        .retain(|reach| reach.reach_id != "south");
    for message in rejections(&forged)? {
        contains(&message, &["south"])?;
    }
    Ok(())
}

#[test]
fn forged_last_interval_contraction_rejects() -> Result {
    let spec = spill();
    let mut batch = spec.build()?;
    for _ in 0..3 {
        batch = step(&batch, batch.time() + 0.2)?;
    }
    let data = batch.snapshot_data()?;
    let mut forged = data.clone();
    let diagnostics = &mut last_interval(&mut forged)?.diagnostics;
    // Two updates measure the contraction. The response step can converge exactly, so the
    // measured value can be zero.
    assert!(
        diagnostics.iterations >= 2,
        "the third spill interval takes two updates: {diagnostics:?}"
    );
    diagnostics.contraction = diagnostics.contraction.next_up();
    for message in rejections_from(&spec, &forged)? {
        contains(
            &message,
            &[
                "last_interval.diagnostics.contraction",
                "recalculated last interval",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn synthesized_binding_mismatch_rejects() -> Result {
    let data = package()?;
    let mut forged = data.clone();
    let binding = forged.reaches[0]
        .1
        .definition
        .lateral_bindings
        .last_mut()
        .ok_or("missing synthesized binding")?;
    assert!(binding.boundary_id.starts_with("surface-exchange:"));
    binding.cell_id = "cell-1".into();
    rejects(
        &forged,
        &[
            "reaches[0] (\"north\").definition",
            "synthesized coordinator reach definition",
        ],
    )?;
    let mut forged = data.clone();
    let last = last_interval(&mut forged)?;
    last.previous.reaches[0].1.definition.lateral_bindings.pop();
    rejects(
        &forged,
        &["last_interval.previous.reaches[0] (\"north\").definition"],
    )?;
    let mut forged = data.clone();
    forged.setup.links[1].crest_elevation_m = forged.setup.links[1].crest_elevation_m.next_up();
    let [imported, _] = rejections(&forged)?;
    contains(
        &imported,
        &["setup.links[1] (\"n-1\")", "destination setup"],
    )?;
    Ok(())
}
