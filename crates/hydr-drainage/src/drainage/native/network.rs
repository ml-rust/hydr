//! Network specification, state, water-balance ledger, and snapshot types for `NativePipeNetwork`.

use super::super::DrainageError;
use super::geometry::PipeCrossSection;

/// Stable numerical contract version for `NativePipeNetwork`.
pub const NATIVE_PIPE_CONTRACT_VERSION: &str = "hydr.native-pipe-drainage/1";

/// Stable checkpoint schema version for `NativePipeNetwork`.
pub const NATIVE_PIPE_SNAPSHOT_SCHEMA: &str = "hydr.native-pipe-drainage-snapshot/1";

/// Outfall hydraulic boundary condition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PipeOutfallBoundary {
    /// Free outfall at invert (unsubmerged outflow).
    Free,
    /// Fixed tailwater stage elevation in meters.
    FixedStage(f64),
}

/// Kind of node in a native pressurized-pipe drainage network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PipeNodeKind {
    /// Manhole / junction with plan area, rim depth, optional surcharge allowance, and ponding area.
    Junction {
        /// Distance from node invert to street/rim elevation (`m`, `> 0`).
        max_depth_m: f64,
        /// Additional pressurized head permitted above `max_depth_m` before surface flooding (`m`, `>= 0`).
        surcharge_depth_m: f64,
        /// Minimum manhole shaft plan area (`m²`, `> 0`, e.g. `1.13` for a 1.2 m manhole).
        shaft_area_m2: f64,
        /// Surface ponding area if head exceeds `max_depth_m + surcharge_depth_m` (`0.0` = lost flooding).
        ponding_area_m2: f64,
    },
    /// Constant-area detention / storage node.
    Storage {
        /// Distance from invert to rim (`m`, `> 0`).
        max_depth_m: f64,
        /// Horizontal plan area (`m²`, `> 0`).
        area_m2: f64,
    },
    /// Network outfall node.
    Outfall {
        /// Outfall hydraulic boundary condition.
        boundary: PipeOutfallBoundary,
    },
}

/// Static specification of one network node.
#[derive(Clone, Debug, PartialEq)]
pub struct PipeNodeSpec {
    /// Unique node identifier.
    pub id: String,
    /// Node invert elevation in meters above the network vertical datum.
    pub invert_m: f64,
    /// Initial water depth above invert (`m`, `>= 0`).
    pub initial_depth_m: f64,
    /// Node type and geometry.
    pub kind: PipeNodeKind,
}

/// Static specification of one closed conduit connecting two nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct PipeConduitSpec {
    /// Unique conduit identifier.
    pub id: String,
    /// Upstream node identifier.
    pub from_node: String,
    /// Downstream node identifier.
    pub to_node: String,
    /// Conduit length (`m`, `> 0`).
    pub length_m: f64,
    /// Interior cross-section shape and dimensions.
    pub section: PipeCrossSection,
    /// Manning roughness coefficient `n` (`s/m^(1/3)`, `> 0`).
    pub manning_n: f64,
    /// Upstream invert offset above `from_node` invert (`m`, `>= 0`).
    pub in_offset_m: f64,
    /// Downstream invert offset above `to_node` invert (`m`, `>= 0`).
    pub out_offset_m: f64,
    /// Initial discharge (`m³/s`, positive `from_node -> to_node`).
    pub initial_flow_m3_per_s: f64,
}

/// Numerical controls for `NativePipeNetwork`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePipeConfig {
    /// Gravitational acceleration (`m/s²`, default `9.81`).
    pub gravity: f64,
    /// Preissmann slot acoustic wave speed (`m/s`, default `10.0`).
    pub slot_celerity_m_per_s: f64,
    /// Maximum internal dynamic-wave substep (`s`, default `0.5`).
    pub max_substep_s: f64,
    /// Minimum dry depth threshold (`m`, default `1e-4`).
    pub dry_depth_m: f64,
}

impl Default for NativePipeConfig {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            slot_celerity_m_per_s: 10.0,
            max_substep_s: 0.5,
            dry_depth_m: 1e-4,
        }
    }
}

impl NativePipeConfig {
    /// Validates numerical controls.
    pub fn validate(self) -> Result<(), DrainageError> {
        if !self.gravity.is_finite() || self.gravity <= 0.0 {
            return Err(DrainageError::input("gravity must be finite and > 0"));
        }
        if !self.slot_celerity_m_per_s.is_finite() || self.slot_celerity_m_per_s <= 0.0 {
            return Err(DrainageError::input(
                "slot_celerity_m_per_s must be finite and > 0",
            ));
        }
        if !self.max_substep_s.is_finite() || self.max_substep_s <= 0.0 {
            return Err(DrainageError::input("max_substep_s must be finite and > 0"));
        }
        if !self.dry_depth_m.is_finite() || self.dry_depth_m <= 0.0 {
            return Err(DrainageError::input("dry_depth_m must be finite and > 0"));
        }
        Ok(())
    }
}

/// Complete static specification of a native pressurized-pipe network.
#[derive(Clone, Debug, PartialEq)]
pub struct NativePipeNetworkSpec {
    /// Vertical datum identifier (e.g. `"NAVD88"`).
    pub vertical_datum: String,
    /// Network nodes (junctions, storage units, outfalls).
    pub nodes: Vec<PipeNodeSpec>,
    /// Closed conduits linking nodes.
    pub conduits: Vec<PipeConduitSpec>,
    /// Numerical solver controls.
    pub config: NativePipeConfig,
}

/// Live hydraulic state of one node in SI units.
#[derive(Clone, Debug, PartialEq)]
pub struct NativePipeNodeState {
    /// Node identifier.
    pub id: String,
    /// Invert elevation (`m`).
    pub invert_m: f64,
    /// Water depth above invert (`m`).
    pub depth_m: f64,
    /// Hydraulic head `invert_m + depth_m` (`m`).
    pub head_m: f64,
    /// True when node head exceeds the maximum crown of connected conduits.
    pub surcharged: bool,
    /// Instantaneous surface flooding rate (`m³/s`, `>= 0`).
    pub flooding_m3_per_s: f64,
}

/// Live hydraulic state of one conduit in SI units.
#[derive(Clone, Debug, PartialEq)]
pub struct NativePipeConduitState {
    /// Conduit identifier.
    pub id: String,
    /// Discharge (`m³/s`, positive `from_node -> to_node`).
    pub flow_m3_per_s: f64,
    /// Midpoint flow depth (`m`).
    pub depth_m: f64,
    /// Current control setting in `[0.0, 1.0]`.
    pub setting: f64,
    /// True when midpoint depth exceeds conduit crown (Preissmann slot active).
    pub pressurized: bool,
}

/// Exact SI water-balance continuity ledger for `NativePipeNetwork`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePipeLedger {
    /// Initial stored network water volume (`m³`).
    pub initial_storage_m3: f64,
    /// Current stored network water volume (`m³`).
    pub final_storage_m3: f64,
    /// Cumulative external lateral inflow (`m³`, net signed).
    pub lateral_inflow_m3: f64,
    /// Cumulative outflow leaving through outfall nodes (`m³`).
    pub outfall_outflow_m3: f64,
    /// Cumulative unponded surface flooding loss (`m³`).
    pub flooding_loss_m3: f64,
    /// Mass conservation residual `(final - initial) - (lateral - outfall - flooding)` (`m³`).
    pub residual_m3: f64,
}

/// Complete live state of `NativePipeNetwork`.
#[derive(Clone, Debug, PartialEq)]
pub struct NativePipeState {
    /// Current simulation time (`s`).
    pub time_s: f64,
    /// Node states in setup order.
    pub nodes: Vec<NativePipeNodeState>,
    /// Conduit states in setup order.
    pub conduits: Vec<NativePipeConduitState>,
    /// Cumulative SI water-balance ledger.
    pub ledger: NativePipeLedger,
}

/// Portable complete checkpoint for `NativePipeNetwork`.
#[derive(Clone, Debug, PartialEq)]
pub struct NativePipeSnapshotData {
    /// Schema identifier (`NATIVE_PIPE_SNAPSHOT_SCHEMA`).
    pub schema: String,
    /// Contract identifier (`NATIVE_PIPE_CONTRACT_VERSION`).
    pub contract: String,
    /// Vertical datum identifier.
    pub vertical_datum: String,
    /// Simulation time (`s`).
    pub time_s: f64,
    /// Node water depths (`m`).
    pub node_depths_m: Vec<f64>,
    /// Conduit discharges (`m³/s`).
    pub conduit_flows_m3_per_s: Vec<f64>,
    /// Conduit control settings in `[0.0, 1.0]`.
    pub conduit_settings: Vec<f64>,
    /// Cumulative SI continuity ledger.
    pub ledger: NativePipeLedger,
}

impl NativePipeNetworkSpec {
    #[allow(clippy::type_complexity)]
    pub(super) fn validate_topology(
        &self,
    ) -> Result<
        (
            std::collections::BTreeMap<String, usize>,
            std::collections::BTreeMap<String, usize>,
            Vec<(usize, usize)>,
            Vec<f64>,
        ),
        DrainageError,
    > {
        if self.vertical_datum.trim().is_empty() {
            return Err(DrainageError::input("vertical_datum must not be empty"));
        }
        self.config.validate()?;
        if self.nodes.is_empty() {
            return Err(DrainageError::input(
                "native pipe network requires at least one node",
            ));
        }
        let mut node_index = std::collections::BTreeMap::new();
        let mut node_crown_m = Vec::with_capacity(self.nodes.len());
        for (idx, node) in self.nodes.iter().enumerate() {
            if node.id.trim().is_empty() {
                return Err(DrainageError::input("node id must not be empty"));
            }
            if node_index.insert(node.id.clone(), idx).is_some() {
                return Err(DrainageError::input(format!(
                    "duplicate node id `{}`",
                    node.id
                )));
            }
            if !node.invert_m.is_finite()
                || !node.initial_depth_m.is_finite()
                || node.initial_depth_m < 0.0
            {
                return Err(DrainageError::input(format!(
                    "node `{}` has invalid invert or initial_depth_m",
                    node.id
                )));
            }
            match node.kind {
                PipeNodeKind::Junction {
                    max_depth_m,
                    surcharge_depth_m,
                    shaft_area_m2,
                    ponding_area_m2,
                } if !max_depth_m.is_finite()
                    || max_depth_m <= 0.0
                    || !surcharge_depth_m.is_finite()
                    || surcharge_depth_m < 0.0
                    || !shaft_area_m2.is_finite()
                    || shaft_area_m2 <= 0.0
                    || !ponding_area_m2.is_finite()
                    || ponding_area_m2 < 0.0 =>
                {
                    return Err(DrainageError::input(format!(
                        "junction `{}` has invalid geometry parameters",
                        node.id
                    )));
                }
                PipeNodeKind::Storage {
                    max_depth_m,
                    area_m2,
                } if !max_depth_m.is_finite()
                    || max_depth_m <= 0.0
                    || !area_m2.is_finite()
                    || area_m2 <= 0.0 =>
                {
                    return Err(DrainageError::input(format!(
                        "storage node `{}` has invalid geometry parameters",
                        node.id
                    )));
                }
                PipeNodeKind::Outfall {
                    boundary: PipeOutfallBoundary::FixedStage(stage),
                } if !stage.is_finite() => {
                    return Err(DrainageError::input(format!(
                        "outfall `{}` has non-finite FixedStage",
                        node.id
                    )));
                }
                _ => {}
            }
            node_crown_m.push(node.invert_m);
        }

        let mut conduit_index = std::collections::BTreeMap::new();
        let mut links = Vec::with_capacity(self.conduits.len());
        for (idx, c) in self.conduits.iter().enumerate() {
            if c.id.trim().is_empty() {
                return Err(DrainageError::input("conduit id must not be empty"));
            }
            if conduit_index.insert(c.id.clone(), idx).is_some() {
                return Err(DrainageError::input(format!(
                    "duplicate conduit id `{}`",
                    c.id
                )));
            }
            c.section.validate()?;
            if !c.length_m.is_finite()
                || c.length_m <= 0.0
                || !c.manning_n.is_finite()
                || c.manning_n <= 0.0
                || !c.in_offset_m.is_finite()
                || c.in_offset_m < 0.0
                || !c.out_offset_m.is_finite()
                || c.out_offset_m < 0.0
                || !c.initial_flow_m3_per_s.is_finite()
            {
                return Err(DrainageError::input(format!(
                    "conduit `{}` has invalid length, roughness, offset, or initial flow",
                    c.id
                )));
            }
            let Some(&from) = node_index.get(&c.from_node) else {
                return Err(DrainageError::input(format!(
                    "conduit `{}` references unknown from_node `{}`",
                    c.id, c.from_node
                )));
            };
            let Some(&to) = node_index.get(&c.to_node) else {
                return Err(DrainageError::input(format!(
                    "conduit `{}` references unknown to_node `{}`",
                    c.id, c.to_node
                )));
            };
            if from == to {
                return Err(DrainageError::input(format!(
                    "conduit `{}` cannot connect node `{}` to itself",
                    c.id, c.from_node
                )));
            }
            let d_full = c.section.full_depth_m();
            node_crown_m[from] =
                node_crown_m[from].max(self.nodes[from].invert_m + c.in_offset_m + d_full);
            node_crown_m[to] =
                node_crown_m[to].max(self.nodes[to].invert_m + c.out_offset_m + d_full);
            links.push((from, to));
        }
        Ok((node_index, conduit_index, links, node_crown_m))
    }
}
