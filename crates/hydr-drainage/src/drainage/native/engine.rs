//! Link-node 1D dynamic-wave Preissmann-slot pressurized-pipe solver engine.

use super::super::DrainageError;
use super::network::{
    NATIVE_PIPE_CONTRACT_VERSION, NATIVE_PIPE_SNAPSHOT_SCHEMA, NativePipeConduitState,
    NativePipeLedger, NativePipeNetworkSpec, NativePipeNodeState, NativePipeSnapshotData,
    NativePipeState, PipeNodeKind, PipeOutfallBoundary,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct ResolvedConduit {
    from: usize,
    to: usize,
}

/// Pure-Rust 1D dynamic-wave Preissmann-slot pressurized-pipe drainage network solver.
#[derive(Clone, Debug)]
pub struct NativePipeNetwork {
    spec: NativePipeNetworkSpec,
    node_index: BTreeMap<String, usize>,
    conduit_index: BTreeMap<String, usize>,
    resolved: Vec<ResolvedConduit>,
    node_crown_m: Vec<f64>,
    time_s: f64,
    node_depths_m: Vec<f64>,
    conduit_flows: Vec<f64>,
    conduit_settings: Vec<f64>,
    node_flooding_rate: Vec<f64>,
    ledger: NativePipeLedger,
}

impl NativePipeNetwork {
    /// Constructs and validates a pure-Rust pressurized-pipe drainage network.
    pub fn new(spec: NativePipeNetworkSpec) -> Result<Self, DrainageError> {
        let (node_index, conduit_index, links, node_crown_m) = spec.validate_topology()?;
        let node_depths_m: Vec<f64> = spec.nodes.iter().map(|n| n.initial_depth_m).collect();
        let resolved: Vec<ResolvedConduit> = links
            .into_iter()
            .map(|(from, to)| ResolvedConduit { from, to })
            .collect();
        let conduit_flows: Vec<f64> = spec
            .conduits
            .iter()
            .map(|c| c.initial_flow_m3_per_s)
            .collect();
        let conduit_settings = vec![1.0; spec.conduits.len()];

        let n_nodes = spec.nodes.len();
        let mut net = Self {
            spec,
            node_index,
            conduit_index,
            resolved,
            node_crown_m,
            time_s: 0.0,
            node_depths_m,
            conduit_flows,
            conduit_settings,
            node_flooding_rate: vec![0.0; n_nodes],
            ledger: NativePipeLedger {
                initial_storage_m3: 0.0,
                final_storage_m3: 0.0,
                lateral_inflow_m3: 0.0,
                outfall_outflow_m3: 0.0,
                flooding_loss_m3: 0.0,
                residual_m3: 0.0,
            },
        };
        let init_vol = net.total_storage_m3();
        net.ledger.initial_storage_m3 = init_vol;
        net.ledger.final_storage_m3 = init_vol;
        Ok(net)
    }

    /// Network specification.
    pub fn spec(&self) -> &NativePipeNetworkSpec {
        &self.spec
    }

    /// Current accepted simulation time (`s`).
    pub fn time_s(&self) -> f64 {
        self.time_s
    }

    /// Looks up a node index by identifier.
    pub fn node_index(&self, id: &str) -> Option<usize> {
        self.node_index.get(id).copied()
    }

    /// Looks up a conduit index by identifier.
    pub fn conduit_index(&self, id: &str) -> Option<usize> {
        self.conduit_index.get(id).copied()
    }

    /// Projects the current live hydraulic state in SI units.
    pub fn state(&self) -> NativePipeState {
        let cfg = self.spec.config;
        let nodes = self
            .spec
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let d = self.node_depths_m[i];
                let head = n.invert_m + d;
                NativePipeNodeState {
                    id: n.id.clone(),
                    invert_m: n.invert_m,
                    depth_m: d,
                    head_m: head,
                    surcharged: head > self.node_crown_m[i] + 1e-6,
                    flooding_m3_per_s: self.node_flooding_rate[i],
                }
            })
            .collect();
        let conduits = self
            .spec
            .conduits
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let r = &self.resolved[k];
                let y_mid = self.conduit_mid_depth(k, r);
                let hyd = c
                    .section
                    .hydraulics(y_mid, cfg.gravity, cfg.slot_celerity_m_per_s);
                NativePipeConduitState {
                    id: c.id.clone(),
                    flow_m3_per_s: self.conduit_flows[k],
                    depth_m: y_mid,
                    setting: self.conduit_settings[k],
                    pressurized: hyd.pressurized,
                }
            })
            .collect();
        NativePipeState {
            time_s: self.time_s,
            nodes,
            conduits,
            ledger: self.ledger,
        }
    }

    /// Runs a transactional trial step to `target_s` under per-node lateral inflows (`(node_idx, m³/s)`)
    /// and optional conduit control settings (`(conduit_idx, setting in [0, 1])`).
    ///
    /// Call [`NativePipeTrial::accept`] to commit. Dropping without accepting leaves `self` unchanged.
    pub fn trial_to<'a>(
        &'a mut self,
        target_s: f64,
        lateral_inflows: &[(usize, f64)],
        conduit_commands: &[(usize, f64)],
    ) -> Result<NativePipeTrial<'a>, DrainageError> {
        if !target_s.is_finite() || target_s <= self.time_s {
            return Err(DrainageError::input(format!(
                "target_s ({target_s}) must be finite and > time_s ({})",
                self.time_s
            )));
        }
        let n_nodes = self.spec.nodes.len();
        let mut q_lat = vec![0.0_f64; n_nodes];
        for &(idx, rate) in lateral_inflows {
            if idx >= n_nodes || !rate.is_finite() {
                return Err(DrainageError::input(format!(
                    "invalid lateral inflow (node={idx}, rate={rate})"
                )));
            }
            q_lat[idx] += rate;
        }

        let mut candidate = self.clone();
        for &(idx, setting) in conduit_commands {
            if idx >= candidate.conduit_settings.len()
                || !setting.is_finite()
                || !(0.0..=1.0).contains(&setting)
            {
                return Err(DrainageError::input(format!(
                    "invalid conduit command (conduit={idx}, setting={setting})"
                )));
            }
            candidate.conduit_settings[idx] = setting;
        }

        let total_dt = target_s - self.time_s;
        let substeps = (total_dt / self.spec.config.max_substep_s).ceil().max(1.0) as usize;
        let dt = total_dt / (substeps as f64);
        for _ in 0..substeps {
            candidate.step_substep(dt, &q_lat);
        }
        candidate.time_s = target_s;
        let final_vol = candidate.total_storage_m3();
        candidate.ledger.final_storage_m3 = final_vol;
        let net = candidate.ledger.lateral_inflow_m3
            - candidate.ledger.outfall_outflow_m3
            - candidate.ledger.flooding_loss_m3;
        candidate.ledger.residual_m3 = (final_vol - candidate.ledger.initial_storage_m3) - net;

        Ok(NativePipeTrial {
            parent: self,
            candidate,
        })
    }

    /// Exports a portable complete checkpoint.
    pub fn snapshot_data(&self) -> NativePipeSnapshotData {
        NativePipeSnapshotData {
            schema: NATIVE_PIPE_SNAPSHOT_SCHEMA.to_string(),
            contract: NATIVE_PIPE_CONTRACT_VERSION.to_string(),
            vertical_datum: self.spec.vertical_datum.clone(),
            time_s: self.time_s,
            node_depths_m: self.node_depths_m.clone(),
            conduit_flows_m3_per_s: self.conduit_flows.clone(),
            conduit_settings: self.conduit_settings.clone(),
            ledger: self.ledger,
        }
    }

    /// Atomically imports a previously exported checkpoint.
    pub fn import_snapshot(&mut self, data: &NativePipeSnapshotData) -> Result<(), DrainageError> {
        if data.schema != NATIVE_PIPE_SNAPSHOT_SCHEMA {
            return Err(DrainageError::input(format!(
                "snapshot schema `{}` does not match `{NATIVE_PIPE_SNAPSHOT_SCHEMA}`",
                data.schema
            )));
        }
        if data.contract != NATIVE_PIPE_CONTRACT_VERSION {
            return Err(DrainageError::input(format!(
                "snapshot contract `{}` does not match `{NATIVE_PIPE_CONTRACT_VERSION}`",
                data.contract
            )));
        }
        if data.vertical_datum != self.spec.vertical_datum {
            return Err(DrainageError::input(format!(
                "snapshot vertical_datum `{}` disagrees with `{}`",
                data.vertical_datum, self.spec.vertical_datum
            )));
        }
        if data.node_depths_m.len() != self.node_depths_m.len()
            || data.conduit_flows_m3_per_s.len() != self.conduit_flows.len()
            || data.conduit_settings.len() != self.conduit_settings.len()
            || !data.time_s.is_finite()
            || data
                .node_depths_m
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            || data.conduit_flows_m3_per_s.iter().any(|v| !v.is_finite())
            || data
                .conduit_settings
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(DrainageError::input(
                "snapshot lengths or values are invalid",
            ));
        }
        self.time_s = data.time_s;
        self.node_depths_m = data.node_depths_m.clone();
        self.conduit_flows = data.conduit_flows_m3_per_s.clone();
        self.conduit_settings = data.conduit_settings.clone();
        self.ledger = data.ledger;
        self.node_flooding_rate.fill(0.0);
        Ok(())
    }

    fn conduit_mid_depth(&self, k: usize, r: &ResolvedConduit) -> f64 {
        let c = &self.spec.conduits[k];
        let h_from = self.spec.nodes[r.from].invert_m + self.node_depths_m[r.from];
        let h_to = self.spec.nodes[r.to].invert_m + self.node_depths_m[r.to];
        let y_from = (h_from - (self.spec.nodes[r.from].invert_m + c.in_offset_m)).max(0.0);
        let y_to = (h_to - (self.spec.nodes[r.to].invert_m + c.out_offset_m)).max(0.0);
        0.5 * (y_from + y_to)
    }

    fn node_storage_m3(&self, i: usize, depth_m: f64) -> f64 {
        let d = depth_m.max(0.0);
        match self.spec.nodes[i].kind {
            PipeNodeKind::Junction {
                max_depth_m,
                surcharge_depth_m,
                shaft_area_m2,
                ponding_area_m2,
            } => {
                let cap = max_depth_m + surcharge_depth_m;
                if d <= cap {
                    shaft_area_m2 * d
                } else {
                    shaft_area_m2 * cap + ponding_area_m2 * (d - cap)
                }
            }
            PipeNodeKind::Storage { area_m2, .. } => area_m2 * d,
            PipeNodeKind::Outfall { .. } => 0.0,
        }
    }

    fn total_storage_m3(&self) -> f64 {
        let cfg = self.spec.config;
        let mut vol = 0.0_f64;
        for i in 0..self.spec.nodes.len() {
            vol += self.node_storage_m3(i, self.node_depths_m[i]);
        }
        for (k, c) in self.spec.conduits.iter().enumerate() {
            let y_mid = self.conduit_mid_depth(k, &self.resolved[k]);
            let hyd = c
                .section
                .hydraulics(y_mid, cfg.gravity, cfg.slot_celerity_m_per_s);
            vol += hyd.area_m2 * c.length_m;
        }
        vol
    }

    fn step_substep(&mut self, dt: f64, q_lat: &[f64]) {
        let cfg = self.spec.config;
        let g = cfg.gravity;
        let n_nodes = self.spec.nodes.len();
        let mut net_inflow = vec![0.0_f64; n_nodes];
        let mut surf_area = vec![0.0_f64; n_nodes];

        // 1. Update conduit discharges via semi-implicit dynamic-wave momentum equation.
        for (k, c) in self.spec.conduits.iter().enumerate() {
            let r = &self.resolved[k];
            let setting = self.conduit_settings[k];
            let h_from = self.spec.nodes[r.from].invert_m + self.node_depths_m[r.from];
            let h_to = self.spec.nodes[r.to].invert_m + self.node_depths_m[r.to];
            let z_from = self.spec.nodes[r.from].invert_m + c.in_offset_m;
            let z_to = self.spec.nodes[r.to].invert_m + c.out_offset_m;
            let y_from = (h_from - z_from).max(0.0);
            let y_to = (h_to - z_to).max(0.0);
            let y_mid = 0.5 * (y_from + y_to);
            let hyd = c.section.hydraulics(y_mid, g, cfg.slot_celerity_m_per_s);
            let half_pipe_area = 0.5 * hyd.top_width_m * c.length_m;
            surf_area[r.from] += half_pipe_area;
            surf_area[r.to] += half_pipe_area;

            let q_next = if setting <= 0.0 || (y_from < cfg.dry_depth_m && y_to < cfg.dry_depth_m) {
                0.0
            } else {
                let eff_h_from = h_from.max(z_from);
                let eff_h_to = h_to.max(z_to);
                let area = hyd.area_m2.max(1e-6);
                let rh = hyd.hydraulic_radius_m.max(1e-4);
                let q_prev = self.conduit_flows[k];
                let num = q_prev - g * area * (dt / c.length_m) * (eff_h_to - eff_h_from);
                let drag =
                    g * c.manning_n * c.manning_n * q_prev.abs() * dt / (area * rh.powf(4.0 / 3.0));
                let mut q = setting * num / (1.0 + drag);
                // Prevent outflow from a dry upstream node or reverse flow from a dry downstream node.
                if (q > 0.0 && y_from < cfg.dry_depth_m) || (q < 0.0 && y_to < cfg.dry_depth_m) {
                    q = 0.0;
                }
                q
            };
            self.conduit_flows[k] = q_next;
            net_inflow[r.from] -= q_next;
            net_inflow[r.to] += q_next;
        }

        // 2. Update node depths via continuity equation & track exact outfall/flooding volumes.
        for (i, node) in self.spec.nodes.iter().enumerate() {
            self.node_flooding_rate[i] = 0.0;
            let q_total = net_inflow[i] + q_lat[i];
            match node.kind {
                PipeNodeKind::Outfall { boundary } => {
                    self.ledger.lateral_inflow_m3 += q_lat[i] * dt;
                    self.ledger.outfall_outflow_m3 += q_total * dt;
                    self.node_depths_m[i] = match boundary {
                        PipeOutfallBoundary::Free => 0.0,
                        PipeOutfallBoundary::FixedStage(stage) => (stage - node.invert_m).max(0.0),
                    };
                }
                PipeNodeKind::Junction {
                    max_depth_m,
                    surcharge_depth_m,
                    shaft_area_m2,
                    ponding_area_m2,
                } => {
                    self.ledger.lateral_inflow_m3 += q_lat[i] * dt;
                    let cap = max_depth_m + surcharge_depth_m;
                    let d_old = self.node_depths_m[i];
                    let area = if d_old > cap && ponding_area_m2 > 0.0 {
                        ponding_area_m2 + surf_area[i]
                    } else {
                        shaft_area_m2 + surf_area[i]
                    };
                    let mut d_new = (d_old + (q_total * dt) / area.max(1e-4)).max(0.0);
                    if d_new > cap && ponding_area_m2 == 0.0 {
                        let excess_vol = (d_new - cap) * area;
                        self.ledger.flooding_loss_m3 += excess_vol;
                        self.node_flooding_rate[i] = excess_vol / dt;
                        d_new = cap;
                    }
                    self.node_depths_m[i] = d_new;
                }
                PipeNodeKind::Storage { area_m2, .. } => {
                    self.ledger.lateral_inflow_m3 += q_lat[i] * dt;
                    let area = area_m2 + surf_area[i];
                    self.node_depths_m[i] =
                        (self.node_depths_m[i] + (q_total * dt) / area.max(1e-4)).max(0.0);
                }
            }
        }
    }
}

/// Uncommitted transactional step of a [`NativePipeNetwork`].
///
/// Call [`Self::accept`] to commit. Dropping without accepting leaves the parent network unchanged.
pub struct NativePipeTrial<'a> {
    parent: &'a mut NativePipeNetwork,
    candidate: NativePipeNetwork,
}

impl NativePipeTrial<'_> {
    /// Proposed end-of-step network state.
    pub fn state(&self) -> NativePipeState {
        self.candidate.state()
    }

    /// Commits the trial state to the parent network and returns the committed state.
    pub fn accept(self) -> NativePipeState {
        *self.parent = self.candidate;
        self.parent.state()
    }
}
