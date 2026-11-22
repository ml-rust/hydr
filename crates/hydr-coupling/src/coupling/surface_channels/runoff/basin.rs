//! Runoff delivery cases: a 2×1 surface, one walled 3-cell reach, and one pond.
//!
//! One impervious subcatchment without depression storage owns surface cell 0, so that cell's
//! rain is suppressed. Surface cell 1 takes direct rain. The pond joins surface cell 1, or reach
//! cell 2 in a linked case, through a weir whose crest the pond never reaches. The outlet
//! drains into the pond or into cells of the reach.

use super::scenario::{OUTLET, POND, SOURCE, definition, outlet, record};
use super::{
    SurfaceChannelRunoff, SurfaceChannelRunoffDestination, SurfaceChannelRunoffOutlet,
    SurfaceChannelRunoffShare,
};
use crate::channel::{
    ChannelBoundary, ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
    RectangularChannelCell,
};
use crate::coupling::surface_channels::pools::scenario::{declaration, link, nodes_setup, pond};
use crate::coupling::surface_channels::structures::scenario::{
    Batch, DATUM, HEAD_ALLOWANCE, TestResult, client, weir_law,
};
use crate::coupling::surface_channels::{
    SurfaceChannelBatch, SurfaceChannelConfig, SurfaceChannelError, SurfaceChannelForcingInterval,
    SurfaceChannelInitialState, SurfaceChannelNode, SurfaceChannelReach,
    SurfaceChannelReachForcing, SurfaceChannelSetup,
};
use crate::runoff::{
    CellRainAssignment, Contribution, ContributionKind, PartitionAssertion, PoolRainSupport,
    SubcatchmentConfig, SubcatchmentInput, SubcatchmentState, SupportAtom, SupportPlanInput,
    SurfaceCellSupport,
};
use crate::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, SolverConfig, SolverSetup,
};
use crate::storage::{LevelPoolConfig, LevelPoolState};
use crate::structures::{DirectionPolicy, StructureOperatingPoint};
use hydr_core::grid::CartesianGrid;
use hydr_core::model::{CellState, Domain, FlowState, SpatialReference};
use hydr_soil::soil::{LandModel, SoilState};

/// Identity of the reach.
pub const REACH: &str = "reach";
/// Plan area of every surface cell, every reach cell, the pond, and the subcatchment, in m².
pub const CELL_AREA: f64 = 4.0;

/// Receiving node of the outlet.
#[derive(Clone, Debug)]
pub enum Receiver {
    /// The pond.
    Pool,
    /// Cells of the reach with their weights.
    Reach(Vec<SurfaceChannelRunoffShare>),
}

/// One basin case.
#[derive(Clone, Debug)]
pub struct Basin {
    pub receiver: Receiver,
    /// Initial runoff depth on the subcatchment, in m.
    pub depth_m: f64,
    /// Initial stage of every reach cell above its bed at 0, in m.
    pub reach_stage_m: f64,
    /// Initial longitudinal momentum of every reach cell, in m⁴/s.
    pub reach_momentum: f64,
    /// Initial pond volume, in m³. The pond's curve origin lies at 0.
    pub pond_m3: f64,
    /// Free-outlet invert, in m.
    pub invert_m: f64,
    /// The pond weir opens onto reach cell 2 instead of surface cell 1.
    pub linked: bool,
    /// Longest common interval, in s.
    pub maximum_step_s: f64,
    /// Coordinator retry limit.
    pub retry_limit: usize,
}

impl Basin {
    /// The default case draining into `receiver`: 1 cm of runoff, a still reach at 5 cm, an
    /// empty pond, and an invert at 3 m. The reach CFL limit lies above 1 s.
    pub fn into(receiver: Receiver) -> Self {
        Self {
            receiver,
            depth_m: 0.01,
            reach_stage_m: 0.05,
            reach_momentum: 0.0,
            pond_m3: 0.0,
            invert_m: 3.0,
            linked: false,
            maximum_step_s: 1.0,
            retry_limit: 16,
        }
    }

    /// The outlet of this case.
    fn outlet(&self) -> SurfaceChannelRunoffOutlet {
        let mut declared = outlet(match &self.receiver {
            Receiver::Pool => SurfaceChannelRunoffDestination::Pool {
                pool_id: POND.into(),
            },
            Receiver::Reach(cells) => SurfaceChannelRunoffDestination::ChannelCells {
                reach_id: REACH.into(),
                cells: cells.clone(),
            },
        });
        declared.free_outlet.invert_elevation_m = self.invert_m;
        declared
    }

    /// Runoff owning surface cell 0, with direct rain on cell 1 and the pond.
    fn runoff(&self) -> TestResult<SurfaceChannelRunoff> {
        let atom = |id: &str| SupportAtom {
            id: id.into(),
            area_m2: CELL_AREA,
        };
        Ok(SurfaceChannelRunoff {
            support: SupportPlanInput {
                partition: PartitionAssertion {
                    identity: "basin-partition".into(),
                    covered_support: "both surface cells and the pond".into(),
                    provenance: "Explicit complete disjoint synthetic partition".into(),
                    complete_disjoint: true,
                },
                surface_input_identity: "surface-input".into(),
                atoms: vec![atom("upstream"), atom("open"), atom("pond-surface")],
                contributions: vec![Contribution {
                    id: SOURCE.into(),
                    input_identity: "upstream-input".into(),
                    provenance: "Explicit synthetic rainfall support".into(),
                    atoms: vec!["upstream".into()],
                    includes: Vec::new(),
                    kind: ContributionKind::ModelledRunoff {
                        definition_id: "upstream-model".into(),
                        outlet_id: OUTLET.into(),
                    },
                }],
                replacements: Vec::new(),
                surface_cells: vec![
                    SurfaceCellSupport {
                        cell: 0,
                        atoms: vec!["upstream".into()],
                        rain: CellRainAssignment::Suppressed {
                            selected_contribution_id: SOURCE.into(),
                        },
                    },
                    SurfaceCellSupport {
                        cell: 1,
                        atoms: vec!["open".into()],
                        rain: CellRainAssignment::Direct,
                    },
                ],
                boundaries: Vec::new(),
                pool_rain: vec![PoolRainSupport {
                    pool_id: POND.into(),
                    atoms: vec!["pond-surface".into()],
                    rain: CellRainAssignment::Direct,
                }],
            },
            definitions: vec![definition(CELL_AREA)?],
            config: SubcatchmentConfig::default(),
            outlets: vec![self.outlet()],
        })
    }

    /// Setup and initial state of this case.
    pub fn parts(&self) -> TestResult<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
        let grid = CartesianGrid::new(2, 1, 2.0, 2.0)?;
        let domain = Domain::new(
            grid,
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![true, true],
            SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1000.0,
                rain: vec![0.0, 0.0],
                potential_evaporation: vec![0.0, 0.0],
                withdrawal: vec![0.0, 0.0],
                boundaries: Vec::new(),
            }],
        )?;
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let still = CellState {
            h: 0.0,
            qx: 0.0,
            qy: 0.0,
        };
        let flow = FlowState::new(&domain, vec![still, still])?;
        let cell = |index: usize| RectangularChannelCell {
            id: format!("cell-{index}"),
            length_m: 2.0,
            width_m: 2.0,
            bed_elevation_m: 0.0,
            maximum_depth_m: 4.0,
            manning_n: 0.0,
        };
        let reach = ChannelReachDefinition {
            vertical_datum: DATUM.into(),
            provenance: "Three walled rectangular cells".into(),
            geometry: ChannelGeometry::Rectangular {
                cells: (0..3).map(cell).collect(),
            },
            lateral_bindings: Vec::new(),
            config: ChannelConfig::default(),
        };
        let setup = SurfaceChannelSetup {
            surface: SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            reaches: vec![SurfaceChannelReach {
                id: REACH.into(),
                definition: reach,
            }],
            links: Vec::new(),
            ports: Vec::new(),
            pools: Vec::new(),
            pool_support: Vec::new(),
            pool_sources: None,
            pool_inflows: Vec::new(),
            storage: LevelPoolConfig::default(),
            structures: Vec::new(),
            pool_ends: Vec::new(),
            operating: None,
            runoff: Some(self.runoff()?),
            storage_exclusion_evidence: "The reach and the pond lie outside the surface".into(),
            config: SurfaceChannelConfig {
                head_absolute_m: HEAD_ALLOWANCE,
                head_relative: HEAD_ALLOWANCE,
                transfer_absolute_m3: 1e-12,
                transfer_relative: 1e-12,
                mass_absolute_m3: 1e-9,
                mass_relative: 1e-12,
                minimum_step_s: 1e-10,
                maximum_step_s: self.maximum_step_s,
                iteration_limit: 64,
                retry_limit: self.retry_limit,
                contraction_target: 0.5,
            },
        };
        let receiving = if self.linked {
            SurfaceChannelNode::ChannelCell {
                reach_id: REACH.into(),
                cell: 2,
            }
        } else {
            SurfaceChannelNode::SurfaceCell(1)
        };
        let spill = link(
            "spill",
            weir_law(1.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            (SurfaceChannelNode::Pool(POND.into()), 0.0),
            (receiving, 0.0),
        );
        let pools = declaration(vec![pond(POND, 0.0, CELL_AREA, 2.0)?], false, Vec::new());
        let setup = nodes_setup(setup, pools, vec![spill]);
        let mut runoff = SubcatchmentState::initialized(&definition(CELL_AREA)?.parameters)?;
        runoff.depths_m[0] = self.depth_m;
        let initial = SurfaceChannelInitialState {
            time_s: 0.0,
            surface: InitialState {
                time: 0.0,
                flow,
                soil,
            },
            reaches: vec![(
                REACH.into(),
                ChannelInitialState {
                    time_s: 0.0,
                    volume_m3: vec![self.reach_stage_m * CELL_AREA; 3],
                    longitudinal_momentum_m4_per_s: vec![self.reach_momentum; 3],
                },
            )],
            pools: vec![LevelPoolState {
                volume_m3: self.pond_m3,
            }],
            runoff: vec![runoff],
        };
        Ok((setup, initial))
    }

    /// The batch of this case.
    pub fn build(&self) -> TestResult<Batch> {
        let (setup, initial) = self.parts()?;
        SurfaceChannelBatch::new(client(), setup, initial).map_err(Into::into)
    }
}

/// Forcing of `batch` over `[t, t + dt]` with `rain` m/s on the subcatchment.
pub fn forcing(batch: &Batch, dt: f64, rain: f64) -> SurfaceChannelForcingInterval {
    let start = batch.time();
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: start + dt,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: Some(ChannelBoundary::Wall),
            right: Some(ChannelBoundary::Wall),
            lateral_inputs: Vec::new(),
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff: vec![record(start, start + dt, rain)],
    }
}

/// One accepted interval of `batch` toward `t + dt` with `rain` m/s on the subcatchment.
pub fn advance(batch: &Batch, dt: f64, rain: f64) -> Result<Batch, SurfaceChannelError> {
    let forcing = forcing(batch, dt, rain);
    Ok(batch.trial_to(forcing.end_s, &forcing)?.into_next())
}

/// Accepted pond volume, in m³.
pub fn pond_volume(batch: &Batch) -> TestResult<f64> {
    let pools = batch.pools().ok_or("the pond")?;
    Ok(pools.states()?.first().ok_or("a pond state")?.volume_m3)
}

/// Accepted reach water, in m³.
pub fn reach_volume(batch: &Batch) -> TestResult<f64> {
    let reach = batch.reach(REACH).ok_or("the reach")?;
    Ok(reach.state()?.iter().map(|cell| cell.volume_m3).sum())
}

/// Export of every subarea when the accepted runoff child of `batch` trials alone to `end`
/// without rain.
pub fn isolated(batch: &Batch, end: f64) -> TestResult<Vec<f64>> {
    let child = batch.runoff().ok_or("the runoff child")?;
    let input = SubcatchmentInput {
        rain_rate_m_per_s: 0.0,
        potential_evaporation_m_per_s: 0.0,
        external_runon_volumes_m3: [0.0; 3],
    };
    let trial = child.trial_to(end, &[input])?;
    let exports = trial.original_outlet()?.try_to_vec::<f64>()?;
    Ok(exports)
}

/// Coefficient `α` of `q = α d^p` on the subcatchment, in SI units.
pub fn alpha() -> TestResult<f64> {
    let parameters = definition(CELL_AREA)?.parameters;
    let closure = parameters
        .closure(crate::runoff::Subarea::ImperviousZeroStorage)
        .ok_or("the impervious closure")?;
    closure
        .coefficient_si()
        .ok_or_else(|| "a finite coefficient".into())
}

/// Fails unless the lifetime composite balance of `batch` holds.
pub fn balanced(batch: &Batch) -> TestResult {
    let balance = batch.ledger()?;
    assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
    Ok(())
}
