//! Simultaneous 3-way conservative coordinator for 2D surface + 1D channels + SWMM drainage.

use super::error::SurfaceChannelDrainageError;
use super::forcing_splice::{covering_forcing_end, splice_drainage_forcing};
use super::state::{
    SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION, SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA,
    SurfaceChannelDrainageSetup, SurfaceChannelDrainageSnapshotData,
    SurfaceChannelDrainageStepDiagnostics,
};
use super::trial::{SurfaceChannelDrainageTrial, validate_snapshot_header};
use crate::coupling::surface_drainage::exchange::{
    ResolvedInlet, aggregate_node_inflows, build_transfer_records, max_abs_diff, propose_transfers,
    relaxation, resolve_link_commands,
};
use crate::coupling::{
    DrainageControlAuthorityTable, SurfaceChannelBatch, SurfaceChannelForcingInterval,
};
use crate::drainage::{SwmmDrainage, SwmmDrainageSnapshotData};
use crate::solver::ForcingSchedule;
use crate::structures::CulvertClient;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TensorOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::interpolate::{Interp1dAlgorithms, LinearInterp1dDiagnostics};
use solvr::optimize::CertifiedMappedBisectionAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
    CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
};
use std::collections::BTreeMap;

const DONOR_SAFETY: f64 = 0.95;

/// Simultaneous 3-way coordinator coupling 2D shallow-water surface flow, 1D river channels
/// (plus level pools, structures, controls, and runoff), and EPA SWMM v5.2.4 piped drainage.
pub struct SurfaceChannelDrainageBatch<R: Runtime<DType = DType>> {
    setup: SurfaceChannelDrainageSetup,
    authority: DrainageControlAuthorityTable,
    surface_channels: SurfaceChannelBatch<R>,
    drainage: SwmmDrainage,
    base_surface_forcing: ForcingSchedule,
    inlets: Vec<ResolvedInlet>,
    unique_cell_count: usize,
    time_s: f64,
    accepted_steps: u64,
    cumulative_inlet_transfers_m3: Vec<f64>,
}

impl<R: Runtime<DType = DType>> SurfaceChannelDrainageBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>
        + LinearInterp1dDiagnostics<R>
        + AndersonAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + CulvertClient<R>,
{
    /// Constructs a validated 3-way coordinator over `surface_channels` and `drainage`.
    pub fn new(
        surface_channels: SurfaceChannelBatch<R>,
        drainage: SwmmDrainage,
        setup: SurfaceChannelDrainageSetup,
    ) -> Result<Self, SurfaceChannelDrainageError> {
        if setup.vertical_datum.trim().is_empty() {
            return Err(SurfaceChannelDrainageError::input(
                "vertical_datum must not be empty",
            ));
        }
        if setup.vertical_datum != drainage.definition().vertical_datum() {
            return Err(SurfaceChannelDrainageError::input(format!(
                "vertical datum mismatch: setup declares `{}` but SWMM drainage declares `{}`",
                setup.vertical_datum,
                drainage.definition().vertical_datum()
            )));
        }
        setup.config.validate()?;
        if setup.inlets.is_empty() {
            return Err(SurfaceChannelDrainageError::input(
                "surface-channel-drainage setup requires at least one inlet",
            ));
        }
        if (surface_channels.time() - drainage.time_s()).abs() > 1e-9 {
            return Err(SurfaceChannelDrainageError::input(format!(
                "component start times disagree: surface_channels={} s, drainage={} s",
                surface_channels.time(),
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

        let domain = surface_channels.surface().domain();
        let bed = domain.bed();
        let active = domain.active();
        let cell_count = domain.grid().cell_count();
        let mut unique_cells = Vec::new();
        let mut cell_to_lane = BTreeMap::new();
        let mut resolved = Vec::with_capacity(setup.inlets.len());
        for inlet in &setup.inlets {
            if inlet.surface_cell >= cell_count || !active[inlet.surface_cell] {
                return Err(SurfaceChannelDrainageError::input(format!(
                    "inlet `{}` references inactive or out-of-range surface cell {}",
                    inlet.id, inlet.surface_cell
                )));
            }
            let Some(node_index) = drainage.node_index(&inlet.drainage_node_id) else {
                return Err(SurfaceChannelDrainageError::input(format!(
                    "inlet `{}` references unknown SWMM drainage node `{}`",
                    inlet.id, inlet.drainage_node_id
                )));
            };
            let node = &drainage.state().nodes[node_index];
            if inlet.rim_elevation_m < node.invert_m {
                return Err(SurfaceChannelDrainageError::input(format!(
                    "inlet `{}` rim_elevation_m ({}) lies below SWMM node `{}` invert ({})",
                    inlet.id, inlet.rim_elevation_m, node.id, node.invert_m
                )));
            }
            let surface_bed_m = bed[inlet.surface_cell];
            if inlet.rim_elevation_m < surface_bed_m - 1e-6 {
                return Err(SurfaceChannelDrainageError::input(format!(
                    "inlet `{}` rim_elevation_m ({}) lies below surface cell {} bed ({surface_bed_m})",
                    inlet.id, inlet.rim_elevation_m, inlet.surface_cell
                )));
            }
            let link_lane = *cell_to_lane.entry(inlet.surface_cell).or_insert_with(|| {
                let lane = unique_cells.len();
                unique_cells.push(inlet.surface_cell);
                lane
            });
            resolved.push(ResolvedInlet {
                spec: inlet.clone(),
                link_lane,
                node_index,
                surface_bed_m,
            });
        }

        let base_surface_forcing = surface_channels.setup().surface.forcing.clone();
        let time_s = surface_channels.time();
        let cumulative_inlet_transfers_m3 = vec![0.0; resolved.len()];
        Ok(Self {
            setup,
            authority,
            surface_channels,
            drainage,
            base_surface_forcing,
            inlets: resolved,
            unique_cell_count: unique_cells.len(),
            time_s,
            accepted_steps: 0,
            cumulative_inlet_transfers_m3,
        })
    }

    /// Declared static 3-way setup.
    pub fn setup(&self) -> &SurfaceChannelDrainageSetup {
        &self.setup
    }

    /// Checked exclusive control and inlet authority table.
    pub fn authority(&self) -> &DrainageControlAuthorityTable {
        &self.authority
    }

    /// Current accepted 2D surface + 1D channel/pool/structure coordinator.
    pub fn surface_channels(&self) -> &SurfaceChannelBatch<R> {
        &self.surface_channels
    }

    /// Current accepted pinned EPA SWMM v5.2.4 piped drainage engine.
    pub fn drainage(&self) -> &SwmmDrainage {
        &self.drainage
    }

    /// Current accepted common simulation time in seconds.
    pub fn time_s(&self) -> f64 {
        self.time_s
    }

    /// Number of accepted 3-way coupled steps.
    pub fn accepted_steps(&self) -> u64 {
        self.accepted_steps
    }

    /// Cumulative signed transfer volume (`m³`) per inlet in setup order.
    pub fn cumulative_inlet_transfers_m3(&self) -> &[f64] {
        &self.cumulative_inlet_transfers_m3
    }

    /// Executes one conservative 3-way coupled trial step toward `requested_end_s`.
    pub fn trial_to<'a>(
        &'a mut self,
        requested_end_s: f64,
        channel_forcing: &SurfaceChannelForcingInterval,
    ) -> Result<SurfaceChannelDrainageTrial<'a, R>, SurfaceChannelDrainageError> {
        self.trial_to_with_commands(requested_end_s, channel_forcing, &[])
    }

    /// Executes one conservative 3-way coupled trial step with external `hydr` SWMM link commands.
    pub fn trial_to_with_commands<'a>(
        &'a mut self,
        requested_end_s: f64,
        channel_forcing: &SurfaceChannelForcingInterval,
        link_commands: &[(String, f64)],
    ) -> Result<SurfaceChannelDrainageTrial<'a, R>, SurfaceChannelDrainageError> {
        if !requested_end_s.is_finite() || requested_end_s <= self.time_s {
            return Err(SurfaceChannelDrainageError::input(format!(
                "requested_end_s ({requested_end_s}) must be finite and greater than time_s ({})",
                self.time_s
            )));
        }
        let swmm_commands = resolve_link_commands(&self.authority, &self.drainage, link_commands)?;
        let cfg = self.setup.config;
        let forcing_end = covering_forcing_end(&self.base_surface_forcing, self.time_s)?;
        let mut end_s = requested_end_s
            .min(self.time_s + cfg.max_step_s)
            .min(forcing_end)
            .min(channel_forcing.end_s);
        let start_s = self.time_s;
        let drainage_backup = self.drainage.snapshot_data();
        let (start_surface_stages, avail_vol_m3) = self.inlet_stages_and_avail()?;
        let start_drainage_heads: Vec<f64> = self
            .inlets
            .iter()
            .map(|i| self.drainage.state().nodes[i.node_index].head_m)
            .collect();

        for retry in 0..=cfg.max_retries {
            let dt_s = end_s - start_s;
            if retry > 0 && dt_s < cfg.min_step_s {
                return Err(SurfaceChannelDrainageError::unresolved(format!(
                    "3-way coupled step shortened to {dt_s} s below min_step_s ({} s)",
                    cfg.min_step_s
                )));
            }

            let mut est_surface = start_surface_stages.clone();
            let mut est_drainage = start_drainage_heads.clone();
            let mut prev_transfers = vec![0.0_f64; self.inlets.len()];
            let mut shortened_end: Option<f64> = None;

            for iter in 0..cfg.max_iterations {
                let transfers_m3 = propose_transfers(
                    &self.inlets,
                    self.unique_cell_count,
                    dt_s,
                    cfg.gravity,
                    &start_surface_stages,
                    &est_surface,
                    &start_drainage_heads,
                    &est_drainage,
                    &avail_vol_m3,
                );
                let node_inflows = aggregate_node_inflows(&self.inlets, &transfers_m3, dt_s);
                let drainage_trial =
                    self.drainage
                        .trial_to_with_commands(end_s, &node_inflows, &swmm_commands)?;

                let spliced = splice_drainage_forcing(
                    &self.base_surface_forcing,
                    self.surface_channels.surface().domain(),
                    &self.surface_channels.setup().surface.boundaries,
                    start_s,
                    end_s,
                    &self.inlets,
                    &transfers_m3,
                )?;
                let sc_spliced = self.surface_channels.continue_surface_forcing(spliced)?;
                let sc_trial = match sc_spliced.trial_to(end_s, channel_forcing) {
                    Ok(trial) => trial,
                    Err(err) => {
                        drop(drainage_trial);
                        return Err(SurfaceChannelDrainageError::SurfaceChannel(err));
                    }
                };

                if sc_trial.time() < end_s - 1e-12 {
                    shortened_end = Some(sc_trial.time());
                    drop(drainage_trial);
                    break;
                }

                let sc_diag = sc_trial.diagnostics()?;
                let next_sc = sc_trial.into_next();
                let new_surface = Self::inlet_stages(&self.inlets, &next_sc)?;
                let new_drainage: Vec<f64> = self
                    .inlets
                    .iter()
                    .map(|i| drainage_trial.state().nodes[i.node_index].head_m)
                    .collect();
                let max_head_res = max_abs_diff(&new_surface, &est_surface)
                    .max(max_abs_diff(&new_drainage, &est_drainage));
                let max_vol_res = max_abs_diff(&transfers_m3, &prev_transfers);

                if max_head_res <= cfg.head_tolerance_m && max_vol_res <= cfg.volume_tolerance_m3 {
                    let inlet_transfers = build_transfer_records(
                        &self.inlets,
                        &transfers_m3,
                        dt_s,
                        &new_surface,
                        &new_drainage,
                    );
                    let diagnostics = SurfaceChannelDrainageStepDiagnostics {
                        start_s,
                        end_s,
                        dt_s,
                        retries: retry,
                        iterations: iter + 1,
                        max_head_residual_m: max_head_res,
                        max_volume_residual_m3: max_vol_res,
                        surface_channel: sc_diag,
                        inlet_transfers,
                    };
                    let _ = drainage_trial.accept();
                    return Ok(SurfaceChannelDrainageTrial {
                        batch: self,
                        drainage_backup,
                        next_sc: Some(next_sc),
                        step_transfers_m3: transfers_m3,
                        diagnostics,
                        accepted: false,
                    });
                }

                drop(drainage_trial);
                let omega = relaxation(iter);
                for k in 0..self.inlets.len() {
                    est_surface[k] += omega * (new_surface[k] - est_surface[k]);
                    est_drainage[k] += omega * (new_drainage[k] - est_drainage[k]);
                }
                prev_transfers = transfers_m3;
            }

            end_s = shortened_end.unwrap_or(start_s + 0.5 * dt_s);
        }

        Err(SurfaceChannelDrainageError::unresolved(format!(
            "3-way surface-channel-drainage coupling did not converge within {} retries",
            cfg.max_retries
        )))
    }

    /// Exports a portable complete 3-way checkpoint.
    pub fn snapshot_data(
        &self,
    ) -> Result<SurfaceChannelDrainageSnapshotData, SurfaceChannelDrainageError> {
        let surface_channel = self.surface_channels.snapshot_data()?;
        let drainage = self.drainage.snapshot_data();
        Ok(SurfaceChannelDrainageSnapshotData {
            schema: SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA.to_string(),
            contract: SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION.to_string(),
            vertical_datum: self.setup.vertical_datum.clone(),
            time_s: self.time_s,
            accepted_steps: self.accepted_steps,
            cumulative_inlet_transfers_m3: self.cumulative_inlet_transfers_m3.clone(),
            surface_channel,
            drainage,
        })
    }

    /// Atomically imports a previously exported 3-way checkpoint.
    pub fn import_snapshot(
        &mut self,
        client: R::Client,
        data: &SurfaceChannelDrainageSnapshotData,
    ) -> Result<(), SurfaceChannelDrainageError> {
        validate_snapshot_header(data, &self.setup.vertical_datum, self.inlets.len())?;

        let restored_sc = SurfaceChannelBatch::<R>::resume(
            client,
            &data.surface_channel,
            data.surface_channel.surface.setup.forcing.clone(),
        )?;
        if (restored_sc.time() - data.time_s).abs() > 1e-6 {
            return Err(SurfaceChannelDrainageError::input(format!(
                "restored surface_channel time ({}) disagrees with coordinator time_s ({})",
                restored_sc.time(),
                data.time_s
            )));
        }
        let backup_drainage = self.drainage.snapshot_data();
        if let Err(err) = self.drainage.import_snapshot(&data.drainage) {
            let _ = self.drainage.import_snapshot(&backup_drainage);
            return Err(SurfaceChannelDrainageError::Drainage(err));
        }

        self.surface_channels = restored_sc;
        self.time_s = self.surface_channels.time();
        self.accepted_steps = data.accepted_steps;
        self.cumulative_inlet_transfers_m3 = data.cumulative_inlet_transfers_m3.clone();
        Ok(())
    }

    fn inlet_stages_and_avail(&self) -> Result<(Vec<f64>, Vec<f64>), SurfaceChannelDrainageError> {
        let state = self
            .surface_channels
            .surface()
            .state()
            .map_err(|e| SurfaceChannelDrainageError::Numerical(e.to_string()))?;
        let cells = state.cells();
        let area = self.surface_channels.surface().domain().grid().cell_area();
        let mut stages = Vec::with_capacity(self.inlets.len());
        let mut avail = vec![0.0_f64; self.unique_cell_count];
        for inlet in &self.inlets {
            let h = cells[inlet.spec.surface_cell].h;
            stages.push(inlet.surface_bed_m + h);
            avail[inlet.link_lane] = (h * area * DONOR_SAFETY).max(0.0);
        }
        Ok((stages, avail))
    }

    fn inlet_stages(
        inlets: &[ResolvedInlet],
        sc: &SurfaceChannelBatch<R>,
    ) -> Result<Vec<f64>, SurfaceChannelDrainageError> {
        let state = sc
            .surface()
            .state()
            .map_err(|e| SurfaceChannelDrainageError::Numerical(e.to_string()))?;
        let cells = state.cells();
        Ok(inlets
            .iter()
            .map(|inlet| inlet.surface_bed_m + cells[inlet.spec.surface_cell].h)
            .collect())
    }
}

impl<R: Runtime<DType = DType>> SurfaceChannelDrainageBatch<R> {
    pub(super) fn commit_trial(
        &mut self,
        next_sc: SurfaceChannelBatch<R>,
        end_s: f64,
        step_transfers_m3: &[f64],
    ) {
        self.surface_channels = next_sc;
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

    pub(super) fn rollback_drainage(&mut self, backup: &SwmmDrainageSnapshotData) {
        let _ = self.drainage.import_snapshot(backup);
    }
}
