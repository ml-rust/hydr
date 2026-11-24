//! Conservative midpoint-head coordinator for 2D surface flow coupled to SWMM drainage.

use super::authority::DrainageControlAuthorityTable;
use super::error::SurfaceDrainageError;
use super::exchange::{
    ResolvedInlet, SurfaceDrainageTrial, aggregate_node_inflows, build_surface_exchange,
    build_transfer_records, max_abs_diff, propose_transfers, relaxation,
};
use super::inlet::SurfaceDrainageSetup;
use super::state::{
    SURFACE_DRAINAGE_CONTRACT_VERSION, SURFACE_DRAINAGE_SNAPSHOT_SCHEMA,
    SurfaceDrainageSnapshotData, SurfaceDrainageStepDiagnostics,
};
use crate::drainage::SwmmDrainage;
use crate::solver::Solver;
use hydr_solver::internal::{CouplingCompletion, CouplingPreparation, SurfaceLinks};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, ReduceOps, ScalarOps, ShapeOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};
use std::collections::BTreeMap;

const LIMIT_SAFETY: f64 = 0.95;

/// Conservative 2D surface / SWMM v5.2.4 piped drainage coordinator.
///
/// Advances both components through atomic midpoint-head trials with proportional donor
/// limiting, complete SWMM hidden-state rollback on unconverged iterations or dropped trials,
/// and dual-component portable checkpoints.
pub struct SurfaceDrainageBatch<R: Runtime<DType = DType>> {
    setup: SurfaceDrainageSetup,
    authority: DrainageControlAuthorityTable,
    surface: Solver<R>,
    drainage: SwmmDrainage,
    links: SurfaceLinks<R>,
    inlets: Vec<ResolvedInlet>,
    time_s: f64,
    accepted_steps: u64,
    cumulative_inlet_transfers_m3: Vec<f64>,
}

impl<R: Runtime<DType = DType>> SurfaceDrainageBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Constructs a validated surface-drainage coordinator over `surface` and `drainage`.
    ///
    /// Checks vertical datum alignment, common start time, inlet geometry, SWMM node lookup,
    /// active surface cell membership, and exclusive control/inlet parameter authority.
    pub fn new(
        surface: Solver<R>,
        drainage: SwmmDrainage,
        setup: SurfaceDrainageSetup,
    ) -> Result<Self, SurfaceDrainageError> {
        if setup.vertical_datum.trim().is_empty() {
            return Err(SurfaceDrainageError::input(
                "vertical_datum must not be empty",
            ));
        }
        if setup.vertical_datum != drainage.definition().vertical_datum() {
            return Err(SurfaceDrainageError::input(format!(
                "vertical datum mismatch: setup declares `{}` but SWMM drainage declares `{}`",
                setup.vertical_datum,
                drainage.definition().vertical_datum()
            )));
        }
        setup.config.validate()?;
        if setup.inlets.is_empty() {
            return Err(SurfaceDrainageError::input(
                "surface-drainage setup requires at least one inlet",
            ));
        }
        if (surface.time() - drainage.time_s()).abs() > 1e-9 {
            return Err(SurfaceDrainageError::input(format!(
                "component start times disagree: surface={} s, drainage={} s",
                surface.time(),
                drainage.time_s()
            )));
        }

        for inlet in &setup.inlets {
            inlet.validate()?;
        }

        let known_link_ids: Vec<String> = drainage
            .state()
            .links
            .iter()
            .map(|l| l.id.clone())
            .collect();
        let authority = DrainageControlAuthorityTable::build(
            drainage.definition().inp_content(),
            &known_link_ids,
            &setup.control_claims,
            &setup.inlets,
        )?;

        let mut unique_cells = Vec::new();
        let mut cell_to_lane = BTreeMap::new();
        for inlet in &setup.inlets {
            cell_to_lane.entry(inlet.surface_cell).or_insert_with(|| {
                let lane = unique_cells.len();
                unique_cells.push(inlet.surface_cell);
                lane
            });
        }
        let links = surface
            .coupling_links(&unique_cells)
            .map_err(SurfaceDrainageError::surface)?;

        let bed = surface.domain().bed();
        let mut resolved = Vec::with_capacity(setup.inlets.len());
        for inlet in &setup.inlets {
            let Some(node_index) = drainage.node_index(&inlet.drainage_node_id) else {
                return Err(SurfaceDrainageError::input(format!(
                    "inlet `{}` references unknown SWMM drainage node `{}`",
                    inlet.id, inlet.drainage_node_id
                )));
            };
            let node = &drainage.state().nodes[node_index];
            if inlet.rim_elevation_m < node.invert_m {
                return Err(SurfaceDrainageError::input(format!(
                    "inlet `{}` rim_elevation_m ({}) lies below SWMM node `{}` invert ({})",
                    inlet.id, inlet.rim_elevation_m, node.id, node.invert_m
                )));
            }
            let surface_bed_m = bed[inlet.surface_cell];
            if inlet.rim_elevation_m < surface_bed_m - 1e-6 {
                return Err(SurfaceDrainageError::input(format!(
                    "inlet `{}` rim_elevation_m ({}) lies below surface cell {} bed ({surface_bed_m})",
                    inlet.id, inlet.rim_elevation_m, inlet.surface_cell
                )));
            }
            let Some(&link_lane) = cell_to_lane.get(&inlet.surface_cell) else {
                return Err(SurfaceDrainageError::input(
                    "internal inlet cell lane lookup failed",
                ));
            };
            resolved.push(ResolvedInlet {
                spec: inlet.clone(),
                link_lane,
                node_index,
                surface_bed_m,
            });
        }

        let time_s = surface.time();
        let cumulative_inlet_transfers_m3 = vec![0.0; resolved.len()];
        Ok(Self {
            setup,
            authority,
            surface,
            drainage,
            links,
            inlets: resolved,
            time_s,
            accepted_steps: 0,
            cumulative_inlet_transfers_m3,
        })
    }

    /// Declared static setup.
    pub fn setup(&self) -> &SurfaceDrainageSetup {
        &self.setup
    }

    /// Checked exclusive control and inlet authority table.
    pub fn authority(&self) -> &DrainageControlAuthorityTable {
        &self.authority
    }

    /// Current accepted 2D surface solver.
    pub fn surface(&self) -> &Solver<R> {
        &self.surface
    }

    /// Current accepted SWMM drainage engine.
    pub fn drainage(&self) -> &SwmmDrainage {
        &self.drainage
    }

    /// Current accepted common simulation time in seconds.
    pub fn time_s(&self) -> f64 {
        self.time_s
    }

    /// Number of accepted coupled steps.
    pub fn accepted_steps(&self) -> u64 {
        self.accepted_steps
    }

    /// Cumulative signed transfer volume (`m³`) per inlet in setup order.
    pub fn cumulative_inlet_transfers_m3(&self) -> &[f64] {
        &self.cumulative_inlet_transfers_m3
    }

    /// Executes one conservative coupled trial step toward `requested_end_s`.
    ///
    /// Both components restore to the accepted state before each Picard iteration and on any
    /// dropped trial. Call [`SurfaceDrainageTrial::accept`] to commit the step atomically.
    pub fn trial_to<'a>(
        &'a mut self,
        requested_end_s: f64,
    ) -> Result<SurfaceDrainageTrial<'a, R>, SurfaceDrainageError> {
        if !requested_end_s.is_finite() || requested_end_s <= self.time_s {
            return Err(SurfaceDrainageError::input(format!(
                "requested_end_s ({requested_end_s}) must be finite and greater than time_s ({})",
                self.time_s
            )));
        }
        let cfg = self.setup.config;
        let forcing_end = self
            .surface
            .coupling_forcing()
            .intervals()
            .iter()
            .find(|iv| iv.start <= self.time_s && self.time_s < iv.end)
            .map(|iv| iv.end)
            .ok_or_else(|| {
                SurfaceDrainageError::input(format!(
                    "surface forcing schedule does not cover time_s={}",
                    self.time_s
                ))
            })?;
        let mut end_s = requested_end_s
            .min(self.time_s + cfg.max_step_s)
            .min(forcing_end);
        let start_s = self.time_s;
        let drainage_backup = self.drainage.snapshot_data();
        let start_surface_stages = Self::inlet_stages(&self.inlets, &self.surface)?;
        let start_drainage_heads: Vec<f64> = self
            .inlets
            .iter()
            .map(|i| self.drainage.state().nodes[i.node_index].head_m)
            .collect();

        for retry in 0..=cfg.max_retries {
            let dt_s = end_s - start_s;
            if retry > 0 && dt_s < cfg.min_step_s {
                return Err(SurfaceDrainageError::unresolved(format!(
                    "coupled step shortened to {dt_s} s below min_step_s ({} s)",
                    cfg.min_step_s
                )));
            }
            let prep = self
                .surface
                .coupling_prepare(end_s, retry, Some(&self.links), None)
                .map_err(SurfaceDrainageError::surface)?;
            let stage = match prep {
                CouplingPreparation::Ready(stage) => stage,
                CouplingPreparation::SourceRetry(_) | CouplingPreparation::BoundaryRetry(_) => {
                    end_s = start_s + 0.5 * dt_s;
                    continue;
                }
            };
            if dt_s > stage.limit() {
                end_s = start_s + LIMIT_SAFETY * stage.limit();
                continue;
            }
            let avail_vol_m3: Vec<f64> = stage.available_volume_m3().to_vec();
            let device = stage.available_volume_m3().device().clone();

            let mut est_surface = start_surface_stages.clone();
            let mut est_drainage = start_drainage_heads.clone();
            let mut prev_transfers = vec![0.0_f64; self.inlets.len()];
            let mut retry_smaller = false;

            for iter in 0..cfg.max_iterations {
                let transfers_m3 = propose_transfers(
                    &self.inlets,
                    self.links.len(),
                    dt_s,
                    cfg.gravity,
                    &start_surface_stages,
                    &est_surface,
                    &start_drainage_heads,
                    &est_drainage,
                    &avail_vol_m3,
                );
                let node_inflows = aggregate_node_inflows(&self.inlets, &transfers_m3, dt_s);
                let drainage_trial = self.drainage.trial_to(end_s, &node_inflows)?;
                let exchange =
                    build_surface_exchange(&device, &self.links, &self.inlets, &transfers_m3)?;
                let completed = self
                    .surface
                    .coupling_complete(&stage, Some(&exchange))
                    .map_err(SurfaceDrainageError::surface)?;

                let next_surface = match completed {
                    CouplingCompletion::Accepted(solver) => *solver,
                    CouplingCompletion::Retry(_) => {
                        drop(drainage_trial);
                        retry_smaller = true;
                        break;
                    }
                };

                let new_surface = Self::inlet_stages(&self.inlets, &next_surface)?;
                let new_drainage: Vec<f64> = self
                    .inlets
                    .iter()
                    .map(|i| drainage_trial.state().nodes[i.node_index].head_m)
                    .collect();
                let max_head_res = max_abs_diff(&new_surface, &est_surface)
                    .max(max_abs_diff(&new_drainage, &est_drainage));
                let max_vol_res = max_abs_diff(&transfers_m3, &prev_transfers);

                if max_head_res <= cfg.head_tolerance_m && max_vol_res <= cfg.volume_tolerance_m3 {
                    let transfer_records = build_transfer_records(
                        &self.inlets,
                        &transfers_m3,
                        dt_s,
                        &new_surface,
                        &new_drainage,
                    );
                    let diagnostics = SurfaceDrainageStepDiagnostics {
                        start_s,
                        end_s,
                        dt_s,
                        retries: retry,
                        iterations: iter + 1,
                        max_head_residual_m: max_head_res,
                        max_volume_residual_m3: max_vol_res,
                        transfers: transfer_records,
                    };
                    let _ = drainage_trial.accept();
                    return Ok(SurfaceDrainageTrial {
                        batch: self,
                        drainage_backup,
                        next_surface: Some(next_surface),
                        step_transfers_m3: transfers_m3,
                        diagnostics,
                        accepted: false,
                    });
                }

                // Unconverged iteration: drop `drainage_trial` to roll back SWMM bit-for-bit.
                drop(drainage_trial);
                let omega = relaxation(iter);
                for k in 0..self.inlets.len() {
                    est_surface[k] += omega * (new_surface[k] - est_surface[k]);
                    est_drainage[k] += omega * (new_drainage[k] - est_drainage[k]);
                }
                prev_transfers = transfers_m3;
            }

            if retry_smaller || retry < cfg.max_retries {
                end_s = start_s + 0.5 * dt_s;
                continue;
            }
        }

        Err(SurfaceDrainageError::unresolved(format!(
            "surface-drainage coupling did not converge within {} retries",
            cfg.max_retries
        )))
    }

    /// Exports a portable complete dual-component checkpoint.
    pub fn snapshot_data(&self) -> Result<SurfaceDrainageSnapshotData, SurfaceDrainageError> {
        let surface = self
            .surface
            .snapshot()
            .data()
            .map_err(SurfaceDrainageError::surface)?;
        let drainage = self.drainage.snapshot_data();
        Ok(SurfaceDrainageSnapshotData {
            schema: SURFACE_DRAINAGE_SNAPSHOT_SCHEMA.to_string(),
            contract: SURFACE_DRAINAGE_CONTRACT_VERSION.to_string(),
            vertical_datum: self.setup.vertical_datum.clone(),
            time_s: self.time_s,
            accepted_steps: self.accepted_steps,
            cumulative_inlet_transfers_m3: self.cumulative_inlet_transfers_m3.clone(),
            surface,
            drainage,
        })
    }

    /// Atomically imports a previously exported dual-component checkpoint.
    pub fn import_snapshot(
        &mut self,
        data: &SurfaceDrainageSnapshotData,
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
        if data.vertical_datum != self.setup.vertical_datum {
            return Err(SurfaceDrainageError::input(format!(
                "snapshot vertical_datum `{}` does not match `{}`",
                data.vertical_datum, self.setup.vertical_datum
            )));
        }
        if data.cumulative_inlet_transfers_m3.len() != self.inlets.len()
            || !data
                .cumulative_inlet_transfers_m3
                .iter()
                .all(|v| v.is_finite())
        {
            return Err(SurfaceDrainageError::input(
                "snapshot cumulative_inlet_transfers_m3 length or finiteness invalid",
            ));
        }
        if (data.drainage.time_s - data.time_s).abs() > 1e-6 {
            return Err(SurfaceDrainageError::input(format!(
                "snapshot drainage time_s ({}) disagrees with coordinator time_s ({})",
                data.drainage.time_s, data.time_s
            )));
        }

        let backup_surface = self.surface.snapshot();
        let backup_drainage = self.drainage.snapshot_data();

        if let Err(err) = self.surface.coupling_restore_data(&data.surface, None) {
            let _ = self.surface.coupling_restore(&backup_surface);
            return Err(SurfaceDrainageError::surface(err));
        }
        if (self.surface.time() - data.time_s).abs() > 1e-6 {
            let _ = self.surface.coupling_restore(&backup_surface);
            return Err(SurfaceDrainageError::input(format!(
                "restored surface time ({}) disagrees with coordinator time_s ({})",
                self.surface.time(),
                data.time_s
            )));
        }
        if let Err(err) = self.drainage.import_snapshot(&data.drainage) {
            let _ = self.surface.coupling_restore(&backup_surface);
            let _ = self.drainage.import_snapshot(&backup_drainage);
            return Err(SurfaceDrainageError::Drainage(err));
        }

        self.time_s = self.surface.time();
        self.accepted_steps = data.accepted_steps;
        self.cumulative_inlet_transfers_m3 = data.cumulative_inlet_transfers_m3.clone();
        Ok(())
    }

    fn inlet_stages(
        inlets: &[ResolvedInlet],
        solver: &Solver<R>,
    ) -> Result<Vec<f64>, SurfaceDrainageError> {
        let state = solver.state().map_err(SurfaceDrainageError::surface)?;
        let cells = state.cells();
        Ok(inlets
            .iter()
            .map(|inlet| inlet.surface_bed_m + cells[inlet.spec.surface_cell].h)
            .collect())
    }
}

impl<R: Runtime<DType = DType>> SurfaceDrainageBatch<R> {
    pub(super) fn commit_trial(
        &mut self,
        next_surface: Solver<R>,
        end_s: f64,
        step_transfers_m3: &[f64],
    ) {
        self.surface = next_surface;
        self.time_s = end_s;
        self.accepted_steps = self.accepted_steps.saturating_add(1);
        for (acc, &dv) in self
            .cumulative_inlet_transfers_m3
            .iter_mut()
            .zip(step_transfers_m3)
        {
            *acc += dv;
        }
    }

    pub(super) fn rollback_drainage(&mut self, backup: &crate::drainage::SwmmDrainageSnapshotData) {
        let _ = self.drainage.import_snapshot(backup);
    }
}
