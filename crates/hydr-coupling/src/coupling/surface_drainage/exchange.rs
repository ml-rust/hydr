//! Inlet transfer proposals, donor limiting, and surface/SWMM exchange assembly.

use super::authority::DrainageControlAuthorityTable;
use super::batch::SurfaceDrainageBatch;
use super::error::SurfaceDrainageError;
use super::inlet::{SurfaceDrainageInlet, inlet_discharge_m3_per_s};
use super::state::{
    SURFACE_DRAINAGE_CONTRACT_VERSION, SURFACE_DRAINAGE_SNAPSHOT_SCHEMA,
    SurfaceDrainageInletTransfer, SurfaceDrainageSnapshotData, SurfaceDrainageStepDiagnostics,
};
use crate::drainage::{
    SwmmControlAuthority, SwmmDrainage, SwmmDrainageSnapshotData, SwmmLinkCommand,
};
use crate::solver::Solver;
use hydr_solver::internal::{ExchangeSource, SurfaceLinks};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub(crate) struct ResolvedInlet {
    pub(crate) spec: SurfaceDrainageInlet,
    pub(crate) link_lane: usize,
    pub(crate) node_index: usize,
    pub(crate) surface_bed_m: f64,
}

/// Evaluates midpoint-head inlet transfers (`m³`) with proportional surface donor limiting.
#[allow(clippy::too_many_arguments)]
pub(crate) fn propose_transfers(
    inlets: &[ResolvedInlet],
    link_count: usize,
    dt_s: f64,
    gravity: f64,
    start_s: &[f64],
    end_s: &[f64],
    start_d: &[f64],
    end_d: &[f64],
    avail_vol_m3: &[f64],
) -> Vec<f64> {
    let mut raw = vec![0.0_f64; inlets.len()];
    let mut cell_demand = vec![0.0_f64; link_count];
    for (k, inlet) in inlets.iter().enumerate() {
        let h_s = 0.5 * (start_s[k] + end_s[k]);
        let h_d = 0.5 * (start_d[k] + end_d[k]);
        let q = inlet_discharge_m3_per_s(&inlet.spec, h_s, h_d, gravity);
        let vol = q * dt_s;
        raw[k] = vol;
        if vol > 0.0 {
            cell_demand[inlet.link_lane] += vol;
        }
    }
    for (k, inlet) in inlets.iter().enumerate() {
        if raw[k] > 0.0 {
            let demand = cell_demand[inlet.link_lane];
            let avail = avail_vol_m3[inlet.link_lane].max(0.0);
            if demand > avail && demand > 0.0 {
                raw[k] *= avail / demand;
            }
        }
    }
    raw
}

/// Assembles device exchange operands for `Solver::coupling_complete`.
pub(super) fn build_surface_exchange<R: Runtime<DType = DType>>(
    device: &R::Device,
    links: &SurfaceLinks<R>,
    inlets: &[ResolvedInlet],
    transfers_m3: &[f64],
) -> Result<ExchangeSource<R>, SurfaceDrainageError> {
    let n = links.len();
    let mut inflow = vec![0.0_f64; n];
    let mut withdrawal = vec![0.0_f64; n];
    for (k, inlet) in inlets.iter().enumerate() {
        let v = transfers_m3[k];
        if v >= 0.0 {
            withdrawal[inlet.link_lane] += v;
        } else {
            inflow[inlet.link_lane] += -v;
        }
    }
    let map_err = |err: numr::error::Error| SurfaceDrainageError::Surface(err.to_string());
    let inflow_t = Tensor::<R>::from_slice(&inflow, &[n], device).map_err(map_err)?;
    let withdrawal_t = Tensor::<R>::from_slice(&withdrawal, &[n], device).map_err(map_err)?;
    let zero_v = Tensor::<R>::zeros(&[n], DType::F64, device).map_err(map_err)?;
    ExchangeSource::new(links, inflow_t, withdrawal_t, [zero_v.clone(), zero_v])
        .map_err(SurfaceDrainageError::surface)
}

/// Aggregates signed per-inlet transfer volumes into per-node SWMM inflow rates (`m³/s`).
pub(crate) fn aggregate_node_inflows(
    inlets: &[ResolvedInlet],
    transfers_m3: &[f64],
    dt_s: f64,
) -> Vec<(usize, f64)> {
    let mut by_node: BTreeMap<usize, f64> = BTreeMap::new();
    for (k, inlet) in inlets.iter().enumerate() {
        *by_node.entry(inlet.node_index).or_insert(0.0) += transfers_m3[k] / dt_s;
    }
    by_node.into_iter().collect()
}

/// Builds per-inlet step transfer records for diagnostics.
pub(crate) fn build_transfer_records(
    inlets: &[ResolvedInlet],
    transfers_m3: &[f64],
    dt_s: f64,
    end_surface_stages: &[f64],
    end_drainage_heads: &[f64],
) -> Vec<SurfaceDrainageInletTransfer> {
    inlets
        .iter()
        .enumerate()
        .map(|(k, inlet)| SurfaceDrainageInletTransfer {
            inlet_id: inlet.spec.id.clone(),
            surface_cell: inlet.spec.surface_cell,
            drainage_node_index: inlet.node_index,
            drainage_node_id: inlet.spec.drainage_node_id.clone(),
            transfer_volume_m3: transfers_m3[k],
            average_rate_m3_per_s: transfers_m3[k] / dt_s,
            end_surface_stage_m: end_surface_stages[k],
            end_drainage_head_m: end_drainage_heads[k],
        })
        .collect()
}

pub(crate) fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

pub(crate) fn relaxation(iter: usize) -> f64 {
    match iter {
        0..=3 => 0.5,
        4..=9 => 0.25,
        _ => 0.125,
    }
}

/// Validates and converts `(link_id, target_setting)` commands against exclusive authority.
pub(crate) fn resolve_link_commands(
    authority: &DrainageControlAuthorityTable,
    drainage: &SwmmDrainage,
    link_commands: &[(String, f64)],
) -> Result<Vec<SwmmLinkCommand>, SurfaceDrainageError> {
    let mut out = Vec::with_capacity(link_commands.len());
    for (link_id, setting) in link_commands {
        if !setting.is_finite() || !(0.0..=1.0).contains(setting) {
            return Err(SurfaceDrainageError::input(format!(
                "link `{link_id}` command setting must be finite in [0.0, 1.0], got {setting}"
            )));
        }
        let Some(owner) = authority.authority_of(link_id) else {
            return Err(SurfaceDrainageError::input(format!(
                "duplicate-parameter-authority: link `{link_id}` has no declared external hydr control claim"
            )));
        };
        let impose_setting = match owner {
            SwmmControlAuthority::NativeSwmm => {
                return Err(SurfaceDrainageError::input(format!(
                    "duplicate-parameter-authority: link `{link_id}` is owned by native SWMM [CONTROLS]; external hydr command rejected"
                )));
            }
            SwmmControlAuthority::HydrCommandsNativeActuation => false,
            SwmmControlAuthority::HydrActuatorImposedPosition
            | SwmmControlAuthority::PrescribedHydrTrajectory => true,
        };
        let Some(link_index) = drainage.link_index(link_id) else {
            return Err(SurfaceDrainageError::input(format!(
                "link `{link_id}` not found in open SWMM drainage network"
            )));
        };
        out.push(SwmmLinkCommand {
            link_index,
            target_setting: *setting,
            impose_setting,
        });
    }
    Ok(out)
}

/// Checks schema, contract, vertical datum, transfer vector, and drainage time consistency.
pub(super) fn validate_snapshot_header(
    data: &SurfaceDrainageSnapshotData,
    expected_datum: &str,
    inlet_count: usize,
) -> Result<(), SurfaceDrainageError> {
    if data.schema != SURFACE_DRAINAGE_SNAPSHOT_SCHEMA {
        return Err(SurfaceDrainageError::input(format!(
            "snapshot schema `{}` does not match `{SURFACE_DRAINAGE_SNAPSHOT_SCHEMA}`",
            data.schema
        )));
    }
    if data.contract != SURFACE_DRAINAGE_CONTRACT_VERSION {
        return Err(SurfaceDrainageError::input(format!(
            "snapshot contract `{}` does not match `{SURFACE_DRAINAGE_CONTRACT_VERSION}`",
            data.contract
        )));
    }
    if data.vertical_datum != expected_datum {
        return Err(SurfaceDrainageError::input(format!(
            "snapshot vertical_datum `{}` does not match `{expected_datum}`",
            data.vertical_datum
        )));
    }
    if data.cumulative_inlet_transfers_m3.len() != inlet_count
        || !data
            .cumulative_inlet_transfers_m3
            .iter()
            .all(|v| v.is_finite())
    {
        return Err(SurfaceDrainageError::input(
            "snapshot cumulative_inlet_transfers_m3 length or finiteness invalid",
        ));
    }
    if (data.drainage.time_s - data.time_s).abs() > 1e-3 {
        return Err(SurfaceDrainageError::input(format!(
            "snapshot drainage time_s ({}) disagrees with coordinator time_s ({})",
            data.drainage.time_s, data.time_s
        )));
    }
    Ok(())
}

/// Uncommitted conservative coupled step of a [`super::SurfaceDrainageBatch`].
///
/// Call [`Self::accept`] to commit both components atomically. Dropping without accepting
/// leaves the surface solver unchanged and rolls back the SWMM worker bit-for-bit.
pub struct SurfaceDrainageTrial<'a, R: Runtime<DType = DType>> {
    pub(super) batch: &'a mut SurfaceDrainageBatch<R>,
    pub(super) drainage_backup: SwmmDrainageSnapshotData,
    pub(super) next_surface: Option<Solver<R>>,
    pub(super) step_transfers_m3: Vec<f64>,
    pub(super) diagnostics: SurfaceDrainageStepDiagnostics,
    pub(super) accepted: bool,
}

impl<R: Runtime<DType = DType>> SurfaceDrainageTrial<'_, R> {
    /// Proposed step diagnostics and shared inlet transfers.
    pub fn diagnostics(&self) -> &SurfaceDrainageStepDiagnostics {
        &self.diagnostics
    }

    /// Commits both the 2D surface solver and the SWMM drainage worker atomically.
    pub fn accept(mut self) -> SurfaceDrainageStepDiagnostics {
        self.accepted = true;
        if let Some(surface) = self.next_surface.take() {
            self.batch
                .commit_trial(surface, self.diagnostics.end_s, &self.step_transfers_m3);
        }
        self.diagnostics.clone()
    }
}

impl<R: Runtime<DType = DType>> Drop for SurfaceDrainageTrial<'_, R> {
    fn drop(&mut self) {
        if !self.accepted {
            self.batch.rollback_drainage(&self.drainage_backup);
        }
    }
}
