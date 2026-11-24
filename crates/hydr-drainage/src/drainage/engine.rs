//! Transactional SWMM v5.2.4 drainage engine with atomic trial rollback and restart.

use super::definition::SwmmDrainageDefinition;
use super::error::DrainageError;
use super::protocol::{SwmmWorker, sha256_hex};
use super::state::{
    FT3_TO_M3, SWMM_DRAINAGE_CONTRACT_VERSION, SWMM_DRAINAGE_SNAPSHOT_SCHEMA,
    SwmmDrainageSnapshotData, SwmmDrainageState, SwmmLinkCommand, SwmmNetworkSummary,
};
use std::collections::BTreeSet;

/// Running instance of a pinned EPA SWMM v5.2.4 piped drainage network.
///
/// Advances only through [`Self::trial_to`] and [`SwmmDrainageTrial::accept`]. Dropping a
/// trial without accepting restores the worker's complete hidden state to the last accepted
/// checkpoint.
pub struct SwmmDrainage {
    definition: SwmmDrainageDefinition,
    summary: SwmmNetworkSummary,
    worker: SwmmWorker,
    time_s: f64,
    accepted_steps: u64,
    accepted_snapshot_hex: String,
    state: SwmmDrainageState,
    nonzero_inflow_nodes: Vec<usize>,
}

impl SwmmDrainage {
    /// Opens a fresh pinned SWMM v5.2.4 worker for `definition`.
    ///
    /// Rejects if the worker binary is missing or if the `.inp` file's surcharge method
    /// disagrees with `definition.surcharge_method()`.
    pub fn open(definition: SwmmDrainageDefinition) -> Result<Self, DrainageError> {
        let (mut worker, summary) =
            SwmmWorker::open(definition.worker_binary(), definition.inp_content())?;
        if summary.surcharge_method != definition.surcharge_method() {
            return Err(DrainageError::input(format!(
                "declared surcharge_method `{}` disagrees with SWMM .inp surcharge method `{}`; align the declaration and .inp SURCHARGE_METHOD",
                definition.surcharge_method().id(),
                summary.surcharge_method.id()
            )));
        }
        let state = worker.state()?;
        let accepted_snapshot_hex = worker.snapshot()?;
        let time_s = state.time_s;
        Ok(Self {
            definition,
            summary,
            worker,
            time_s,
            accepted_steps: 0,
            accepted_snapshot_hex,
            state,
            nonzero_inflow_nodes: Vec::new(),
        })
    }

    /// Opens a fresh worker and restores it from a previously exported checkpoint.
    pub fn resume(
        definition: SwmmDrainageDefinition,
        snapshot: &SwmmDrainageSnapshotData,
    ) -> Result<Self, DrainageError> {
        let mut engine = Self::open(definition)?;
        engine.import_snapshot(snapshot)?;
        Ok(engine)
    }

    /// Validated drainage definition.
    pub fn definition(&self) -> &SwmmDrainageDefinition {
        &self.definition
    }

    /// Network object counts and options reported at open.
    pub fn summary(&self) -> SwmmNetworkSummary {
        self.summary
    }

    /// Current accepted simulation time in seconds.
    pub fn time_s(&self) -> f64 {
        self.time_s
    }

    /// Number of accepted steps.
    pub fn accepted_steps(&self) -> u64 {
        self.accepted_steps
    }

    /// Current accepted canonical SI state.
    pub fn state(&self) -> &SwmmDrainageState {
        &self.state
    }

    /// Looks up a node's zero-based SWMM index by its `.inp` identifier.
    pub fn node_index(&self, id: &str) -> Option<usize> {
        self.state
            .nodes
            .iter()
            .find(|node| node.id == id)
            .map(|node| node.index)
    }

    /// Looks up a link's zero-based SWMM index by its `.inp` identifier.
    pub fn link_index(&self, id: &str) -> Option<usize> {
        self.state
            .links
            .iter()
            .find(|link| link.id == id)
            .map(|link| link.index)
    }

    /// Executes a transactional trial step from `self.time_s()` to `end_s`.
    ///
    /// `node_inflows_m3_per_s` supplies external nodal inflow rates in `m³/s` for this step;
    /// any node omitted from the slice receives zero external API inflow during the step.
    /// Dropping the returned [`SwmmDrainageTrial`] without calling [`SwmmDrainageTrial::accept`]
    /// restores the worker to the last accepted state bit-for-bit.
    pub fn trial_to<'a>(
        &'a mut self,
        end_s: f64,
        node_inflows_m3_per_s: &[(usize, f64)],
    ) -> Result<SwmmDrainageTrial<'a>, DrainageError> {
        self.trial_to_with_commands(end_s, node_inflows_m3_per_s, &[])
    }

    /// Executes a transactional trial step with both nodal inflows and external link commands.
    pub fn trial_to_with_commands<'a>(
        &'a mut self,
        end_s: f64,
        node_inflows_m3_per_s: &[(usize, f64)],
        link_commands: &[SwmmLinkCommand],
    ) -> Result<SwmmDrainageTrial<'a>, DrainageError> {
        if !end_s.is_finite() || end_s <= self.time_s {
            return Err(DrainageError::input(format!(
                "trial end_s ({end_s}) must be finite and greater than current time_s ({})",
                self.time_s
            )));
        }
        let mut seen = BTreeSet::new();
        for &(node_idx, rate_m3_per_s) in node_inflows_m3_per_s {
            if node_idx >= self.summary.node_count {
                return Err(DrainageError::input(format!(
                    "node index {node_idx} out of range (node_count = {})",
                    self.summary.node_count
                )));
            }
            if !rate_m3_per_s.is_finite() {
                return Err(DrainageError::input(format!(
                    "node {node_idx} inflow rate must be finite, got {rate_m3_per_s}"
                )));
            }
            if !seen.insert(node_idx) {
                return Err(DrainageError::input(format!(
                    "duplicate inflow entry for node index {node_idx}"
                )));
            }
        }
        let mut seen_links = BTreeSet::new();
        for cmd in link_commands {
            if cmd.link_index >= self.summary.link_count {
                return Err(DrainageError::input(format!(
                    "link index {} out of range (link_count = {})",
                    cmd.link_index, self.summary.link_count
                )));
            }
            if !cmd.target_setting.is_finite() || !(0.0..=1.0).contains(&cmd.target_setting) {
                return Err(DrainageError::input(format!(
                    "link {} target_setting must be finite in [0.0, 1.0], got {}",
                    cmd.link_index, cmd.target_setting
                )));
            }
            if !seen_links.insert(cmd.link_index) {
                return Err(DrainageError::input(format!(
                    "duplicate link command for link index {}",
                    cmd.link_index
                )));
            }
        }

        let dt_s = end_s - self.time_s;
        let run_result = (|| -> Result<(SwmmDrainageState, String, Vec<usize>), DrainageError> {
            for &prev_node in &self.nonzero_inflow_nodes {
                self.worker.set_inflow(prev_node, 0.0)?;
            }
            let mut active = Vec::with_capacity(node_inflows_m3_per_s.len());
            for &(node_idx, rate_m3_per_s) in node_inflows_m3_per_s {
                let rate_cfs = rate_m3_per_s / FT3_TO_M3;
                self.worker.set_inflow(node_idx, rate_cfs)?;
                if rate_m3_per_s != 0.0 {
                    active.push(node_idx);
                }
            }
            for cmd in link_commands {
                self.worker.set_link_setting(
                    cmd.link_index,
                    cmd.target_setting,
                    cmd.impose_setting,
                )?;
            }
            let _reported_time_s = self.worker.step(dt_s)?;
            let next_state = self.worker.state()?;
            let next_snapshot_hex = self.worker.snapshot()?;
            Ok((next_state, next_snapshot_hex, active))
        })();

        match run_result {
            Ok((next_state, next_snapshot_hex, active_inflows)) => Ok(SwmmDrainageTrial {
                engine: self,
                next_state,
                next_snapshot_hex,
                active_inflows,
                accepted: false,
            }),
            Err(err) => {
                let _ = self.worker.restore(&self.accepted_snapshot_hex);
                Err(err)
            }
        }
    }

    /// Exports the current accepted complete hidden-state checkpoint.
    pub fn snapshot_data(&self) -> SwmmDrainageSnapshotData {
        SwmmDrainageSnapshotData {
            schema: SWMM_DRAINAGE_SNAPSHOT_SCHEMA.to_string(),
            contract: SWMM_DRAINAGE_CONTRACT_VERSION.to_string(),
            inp_sha256: self.definition.inp_sha256().to_string(),
            vertical_datum: self.definition.vertical_datum().to_string(),
            surcharge_method: self.definition.surcharge_method(),
            time_s: self.time_s,
            accepted_steps: self.accepted_steps,
            state_digest_sha256: sha256_hex(self.accepted_snapshot_hex.as_bytes()),
            raw_state_hex: self.accepted_snapshot_hex.clone(),
        }
    }

    /// Atomically imports a previously exported checkpoint into this running worker.
    ///
    /// Checks schema, contract, `.inp` SHA-256, vertical datum, surcharge method, and
    /// payload SHA-256 digest before touching the worker; rolls back on restore failure.
    pub fn import_snapshot(
        &mut self,
        data: &SwmmDrainageSnapshotData,
    ) -> Result<(), DrainageError> {
        if data.schema != SWMM_DRAINAGE_SNAPSHOT_SCHEMA {
            return Err(DrainageError::input(format!(
                "snapshot schema `{}` does not match `{SWMM_DRAINAGE_SNAPSHOT_SCHEMA}`",
                data.schema
            )));
        }
        if data.contract != SWMM_DRAINAGE_CONTRACT_VERSION {
            return Err(DrainageError::input(format!(
                "snapshot contract `{}` does not match `{SWMM_DRAINAGE_CONTRACT_VERSION}`",
                data.contract
            )));
        }
        if data.inp_sha256 != self.definition.inp_sha256() {
            return Err(DrainageError::input(format!(
                "snapshot inp_sha256 `{}` does not match open model `{}`",
                data.inp_sha256,
                self.definition.inp_sha256()
            )));
        }
        if data.vertical_datum != self.definition.vertical_datum() {
            return Err(DrainageError::input(format!(
                "snapshot vertical_datum `{}` does not match open model `{}`",
                data.vertical_datum,
                self.definition.vertical_datum()
            )));
        }
        if data.surcharge_method != self.definition.surcharge_method() {
            return Err(DrainageError::input(format!(
                "snapshot surcharge_method `{}` does not match open model `{}`",
                data.surcharge_method.id(),
                self.definition.surcharge_method().id()
            )));
        }
        if !data.time_s.is_finite() || data.time_s < 0.0 {
            return Err(DrainageError::input(format!(
                "snapshot time_s must be finite and non-negative, got {}",
                data.time_s
            )));
        }
        let expected_digest = sha256_hex(data.raw_state_hex.as_bytes());
        if expected_digest != data.state_digest_sha256 {
            return Err(DrainageError::input(
                "snapshot state_digest_sha256 does not match raw_state_hex payload",
            ));
        }

        if let Err(err) = self.worker.restore(&data.raw_state_hex) {
            let _ = self.worker.restore(&self.accepted_snapshot_hex);
            return Err(err);
        }
        let restored_state = match self.worker.state() {
            Ok(state) => state,
            Err(err) => {
                let _ = self.worker.restore(&self.accepted_snapshot_hex);
                return Err(err);
            }
        };
        if (restored_state.time_s - data.time_s).abs() > 1e-6 {
            let _ = self.worker.restore(&self.accepted_snapshot_hex);
            return Err(DrainageError::input(format!(
                "restored worker time_s ({}) disagrees with snapshot header time_s ({})",
                restored_state.time_s, data.time_s
            )));
        }

        self.time_s = restored_state.time_s;
        self.accepted_steps = data.accepted_steps;
        self.accepted_snapshot_hex = data.raw_state_hex.clone();
        self.state = restored_state;
        // Mark all nodes as potentially having restored apiExtInflow so the next trial clears them.
        self.nonzero_inflow_nodes = (0..self.summary.node_count).collect();
        Ok(())
    }
}

/// Uncommitted trial step of a [`SwmmDrainage`] worker.
///
/// Call [`Self::accept`] to commit the step. Dropping this value without accepting
/// restores the worker's complete hidden state to the previous accepted checkpoint.
pub struct SwmmDrainageTrial<'a> {
    engine: &'a mut SwmmDrainage,
    next_state: SwmmDrainageState,
    next_snapshot_hex: String,
    active_inflows: Vec<usize>,
    accepted: bool,
}

impl<'a> SwmmDrainageTrial<'a> {
    /// Proposed canonical SI state at the end of the trial step.
    pub fn state(&self) -> &SwmmDrainageState {
        &self.next_state
    }

    /// Proposed simulation time in seconds.
    pub fn end_s(&self) -> f64 {
        self.next_state.time_s
    }

    /// Commits the trial step as the engine's accepted state and returns it.
    pub fn accept(mut self) -> SwmmDrainageState {
        self.accepted = true;
        self.engine.time_s = self.next_state.time_s;
        self.engine.accepted_steps = self.engine.accepted_steps.saturating_add(1);
        self.engine.accepted_snapshot_hex = std::mem::take(&mut self.next_snapshot_hex);
        self.engine.state = self.next_state.clone();
        self.engine.nonzero_inflow_nodes = std::mem::take(&mut self.active_inflows);
        self.next_state.clone()
    }
}

impl Drop for SwmmDrainageTrial<'_> {
    fn drop(&mut self) {
        if !self.accepted {
            let _ = self
                .engine
                .worker
                .restore(&self.engine.accepted_snapshot_hex);
        }
    }
}
