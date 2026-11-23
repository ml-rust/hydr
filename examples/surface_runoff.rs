//! Native runoff attached to the surface/channel coordinator.
//!
//! Three impervious subcatchments own the three surface cells, so the plan suppresses their
//! rain. The north outlet drains into the middle cell of a walled reach. The south outlet drains
//! into a pond. The east outlet drains into surface cell 0, which no link or structure touches.
//! A weir joins the pond to surface cell 1, but the pond never reaches its crest. The run
//! prints each interval's runoff transfers and the composite balance, then resumes from a
//! checkpoint package.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
    RectangularChannelCell,
};
use hydr::coupling::{
    CoupledPoolSupport, StructureScenario, SurfaceChannelAttachment, SurfaceChannelBatch,
    SurfaceChannelConfig, SurfaceChannelForcingInterval, SurfaceChannelFreeOutlet,
    SurfaceChannelInitialState, SurfaceChannelNode, SurfaceChannelOperationOwner,
    SurfaceChannelReach, SurfaceChannelReachForcing, SurfaceChannelRunoff,
    SurfaceChannelRunoffDestination, SurfaceChannelRunoffOutlet, SurfaceChannelRunoffShare,
    SurfaceChannelSetup, SurfaceChannelStructure,
};
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::runoff::{
    CellRainAssignment, Contribution, ContributionKind, PartitionAssertion, PoolRainSupport,
    SubcatchmentConfig, SubcatchmentContributionInput, SubcatchmentDefinition,
    SubcatchmentGeometry, SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting,
    SubcatchmentState, SupportAtom, SupportPlanInput, SurfaceCellSupport,
};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, SolverConfig, SolverSetup,
};
use hydr::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolState,
};
use hydr::structures::{
    DirectionPolicy, StructureCondition, StructureElement, StructureLaw, StructureOperatingPoint,
    SuppressedRectangularWeirParameters, WeirApplicability,
};

type Batch = SurfaceChannelBatch<crate::runtime::Runtime>;
type Error = Box<dyn std::error::Error>;

const DATUM: &str = "Example datum";
/// Plan area of every surface cell, reach cell, subcatchment, and the pond, in m².
const AREA: f64 = 4.0;
/// Rain on every subcatchment, in m/s.
const RAIN: f64 = 2e-5;

/// One impervious subcatchment without depression storage.
fn definition(id: &str) -> Result<SubcatchmentDefinition, Error> {
    Ok(SubcatchmentDefinition {
        id: id.into(),
        parameters: SubcatchmentParameters::new(
            SubcatchmentGeometry {
                total_area_m2: AREA,
                width_m: 2.0,
                slope: 1.0 / 64.0,
                impervious_fraction: 1.0,
                zero_storage_impervious_fraction: 1.0,
                impervious_manning: 1.0 / 32.0,
                pervious_manning: 1.0 / 32.0,
                impervious_depression_m: 0.0,
                pervious_depression_m: 0.0,
            },
            None,
            SubcatchmentRouting::Direct,
        )?,
    })
}

/// The modelled source `name` over its own atom.
fn source(name: &str) -> Contribution {
    Contribution {
        id: format!("{name}-source"),
        input_identity: format!("{name}-input"),
        provenance: "Synthetic rainfall over one subcatchment".into(),
        atoms: vec![name.into()],
        includes: Vec::new(),
        kind: ContributionKind::ModelledRunoff {
            definition_id: format!("{name}-model"),
            outlet_id: format!("{name}-outlet"),
        },
    }
}

/// Surface cell `cell` owned by the source `name`.
fn owned_cell(cell: usize, name: &str) -> SurfaceCellSupport {
    SurfaceCellSupport {
        cell,
        atoms: vec![name.into()],
        rain: CellRainAssignment::Suppressed {
            selected_contribution_id: format!("{name}-source"),
        },
    }
}

/// The outlet of the source `name` into `destination`, free below 3 m.
fn outlet(name: &str, destination: SurfaceChannelRunoffDestination) -> SurfaceChannelRunoffOutlet {
    SurfaceChannelRunoffOutlet {
        outlet_id: format!("{name}-outlet"),
        destination,
        free_outlet: SurfaceChannelFreeOutlet {
            invert_elevation_m: 3.0,
            evidence: "The outfall sits above the highest receiving water".into(),
        },
    }
}

/// Runoff of every subcatchment: north into reach cell 1, south into the pond, east into
/// surface cell 0.
fn runoff() -> Result<SurfaceChannelRunoff, Error> {
    let atom = |id: &str| SupportAtom {
        id: id.into(),
        area_m2: AREA,
    };
    Ok(SurfaceChannelRunoff {
        support: SupportPlanInput {
            partition: PartitionAssertion {
                identity: "example-partition".into(),
                covered_support: "every surface cell and the pond".into(),
                provenance: "Complete disjoint synthetic partition".into(),
                complete_disjoint: true,
            },
            surface_input_identity: "surface-input".into(),
            atoms: vec![
                atom("north"),
                atom("south"),
                atom("east"),
                atom("pond-surface"),
            ],
            contributions: vec![source("north"), source("south"), source("east")],
            replacements: Vec::new(),
            surface_cells: vec![
                owned_cell(0, "north"),
                owned_cell(1, "south"),
                owned_cell(2, "east"),
            ],
            boundaries: Vec::new(),
            pool_rain: vec![PoolRainSupport {
                pool_id: "pond".into(),
                atoms: vec!["pond-surface".into()],
                rain: CellRainAssignment::Direct,
            }],
        },
        definitions: vec![
            definition("east-model")?,
            definition("north-model")?,
            definition("south-model")?,
        ],
        config: SubcatchmentConfig::default(),
        outlets: vec![
            outlet(
                "north",
                SurfaceChannelRunoffDestination::ChannelCells {
                    reach_id: "reach".into(),
                    cells: vec![SurfaceChannelRunoffShare {
                        index: 1,
                        weight: 1.0,
                    }],
                },
            ),
            outlet(
                "south",
                SurfaceChannelRunoffDestination::Pool {
                    pool_id: "pond".into(),
                },
            ),
            outlet(
                "east",
                SurfaceChannelRunoffDestination::SurfaceCells {
                    cells: vec![SurfaceChannelRunoffShare {
                        index: 0,
                        weight: 1.0,
                    }],
                },
            ),
        ],
    })
}

/// A weir from the pond to surface cell 1 with its crest at 1 m.
fn spill() -> Result<SurfaceChannelStructure, Error> {
    let pond = SurfaceChannelNode::Pool("pond".into());
    let cell = SurfaceChannelNode::SurfaceCell(1);
    let attachment = |node: SurfaceChannelNode| SurfaceChannelAttachment {
        node,
        bed_elevation_m: 0.0,
        physical_connection_assertion: "The weir opens onto this node".into(),
    };
    Ok(SurfaceChannelStructure {
        element: StructureElement {
            id: "spill".into(),
            endpoint_a: pond.identity(),
            endpoint_b: cell.identity(),
            datum: DATUM.into(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
                1.0,
                1.0,
                1.7,
                WeirApplicability {
                    coefficient_provenance: "Declared weir coefficient".into(),
                    free_nappe_and_approach_assertion: "Suppressed crest, quiescent approach"
                        .into(),
                    submergence_correction_assertion: "Villemonte correction adopted".into(),
                    symmetric_reversal: true,
                },
            )?),
            condition: StructureCondition::AsBuilt,
        },
        endpoint_a: attachment(pond),
        endpoint_b: attachment(cell),
        scenario: StructureScenario {
            id: "spill-as-built".into(),
            evidence: "As-built geometry, no blockage".into(),
        },
        operation: SurfaceChannelOperationOwner::Manual(StructureOperatingPoint::Passive),
    })
}

/// Setup and original state: a dry 3×1 surface, a still reach at 5 cm, an empty pond, and
/// 1 cm of runoff on each subcatchment.
fn parts() -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState), Error> {
    let domain = Domain::new(
        CartesianGrid::new(3, 1, 2.0, 2.0)?,
        vec![0.0; 3],
        vec![0.0; 3],
        vec![true; 3],
        SpatialReference::new("Example projected CRS".into(), DATUM.into(), [0.0; 2])?,
    )?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 1000.0,
            rain: vec![0.0; 3],
            potential_evaporation: vec![0.0; 3],
            withdrawal: vec![0.0; 3],
            boundaries: Vec::new(),
        }],
    )?;
    let land = LandModel::impervious(&domain);
    let soil = SoilState::initialized(&land)?;
    let dry = CellState {
        h: 0.0,
        qx: 0.0,
        qy: 0.0,
    };
    let flow = FlowState::new(&domain, vec![dry; 3])?;
    let cell = |index: usize| RectangularChannelCell {
        id: format!("cell-{index}"),
        length_m: 2.0,
        width_m: 2.0,
        bed_elevation_m: 0.0,
        maximum_depth_m: 4.0,
        manning_n: 0.0,
    };
    let knot = |volume_m3, stage_m| LevelPoolKnot { volume_m3, stage_m };
    let pond = LevelPoolElement {
        id: "pond".into(),
        curve: LevelPoolCurve::new(
            vec![knot(0.0, 0.0), knot(2.0 * AREA, 2.0)],
            DATUM.into(),
            "Constant-area pond".into(),
        )?,
        connected_pool_assertion: "One level water body".into(),
        storage_support_identity: "pond-basin".into(),
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
            id: "reach".into(),
            definition: ChannelReachDefinition {
                vertical_datum: DATUM.into(),
                provenance: "Three walled rectangular cells".into(),
                geometry: ChannelGeometry::Rectangular {
                    cells: (0..3).map(cell).collect(),
                },
                lateral_bindings: Vec::new(),
                config: ChannelConfig::default(),
            },
        }],
        links: Vec::new(),
        ports: Vec::new(),
        pools: vec![pond],
        pool_support: vec![CoupledPoolSupport {
            pool_id: "pond".into(),
            zero_volume_stage_m: 0.0,
            zero_volume_limit_assertion: "The pond empties at its curve origin".into(),
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
        storage: LevelPoolConfig::default(),
        structures: vec![spill()?],
        pool_ends: Vec::new(),
        operating: None,
        geometry_schedules: Vec::new(),
        pool_curve_schedules: Vec::new(),
        surface_bed_schedules: Vec::new(),
        channel_section_schedules: Vec::new(),
        runoff: Some(runoff()?),
        storage_exclusion_evidence: "The reach and the pond lie outside the surface".into(),
        config: SurfaceChannelConfig {
            head_absolute_m: 1e-12,
            head_relative: 1e-12,
            transfer_absolute_m3: 1e-12,
            transfer_relative: 1e-12,
            mass_absolute_m3: 1e-9,
            mass_relative: 1e-12,
            minimum_step_s: 1e-10,
            maximum_step_s: 1.0,
            iteration_limit: 64,
            retry_limit: 16,
            contraction_target: 0.5,
        },
    };
    let mut wet = SubcatchmentState::initialized(&definition("north-model")?.parameters)?;
    wet.depths_m[0] = 0.01;
    let initial = SurfaceChannelInitialState {
        time_s: 0.0,
        surface: InitialState {
            time: 0.0,
            flow,
            soil,
        },
        reaches: vec![(
            "reach".into(),
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![0.05 * AREA; 3],
                longitudinal_momentum_m4_per_s: vec![0.0; 3],
            },
        )],
        pools: vec![LevelPoolState { volume_m3: 0.0 }],
        runoff: vec![wet; 3],
    };
    Ok((setup, initial))
}

/// Forcing over `[start, end]` with walls at both reach ends and rain on every subcatchment.
fn forcing(start: f64, end: f64) -> SurfaceChannelForcingInterval {
    let record = |name: &str| SubcatchmentContributionInput {
        contribution_id: format!("{name}-source"),
        input_identity: format!("{name}-input"),
        start,
        end,
        input: SubcatchmentInput {
            rain_rate_m_per_s: RAIN,
            potential_evaporation_m_per_s: 0.0,
            external_runon_volumes_m3: [0.0; 3],
        },
    };
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: "reach".into(),
            left: Some(ChannelBoundary::Wall),
            right: Some(ChannelBoundary::Wall),
            lateral_inputs: Vec::new(),
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff: vec![record("north"), record("south"), record("east")],
    }
}

/// One accepted interval of 0.25 s.
fn advance(batch: &Batch) -> Result<Batch, Error> {
    let end = batch.time() + 0.25;
    Ok(batch
        .trial_to(end, &forcing(batch.time(), end))?
        .into_next())
}

fn main() -> Result<(), Error> {
    let (setup, initial) = parts()?;
    let mut batch = Batch::new(crate::runtime::client(), setup, initial)?;
    for _ in 0..4 {
        batch = advance(&batch)?;
        let step = batch.last_step()?.ok_or("an accepted interval")?;
        for transfer in &step.runoff {
            println!(
                "t={} outlet={} target={} volume_m3={}",
                step.end_s,
                transfer.outlet_id,
                transfer.target.identity(),
                transfer.volume_m3
            );
        }
        let balance = batch.ledger()?;
        println!(
            "t={} runoff_storage_m3={} runoff_delivered_m3={} residual_m3={} bound_m3={}",
            batch.time(),
            balance.runoff_storage_m3,
            balance.runoff_delivered_m3,
            balance.residual_m3,
            balance.bound_m3
        );
    }
    let data = batch.snapshot_data()?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &data,
        data.surface.setup.forcing.clone(),
    )?;
    let next = advance(&batch)?;
    let repeated = advance(&resumed)?;
    if next.snapshot_data()? != repeated.snapshot_data()? {
        return Err("the resumed run differs from the uninterrupted run".into());
    }
    println!("resumed run matches at t={}", repeated.time());
    Ok(())
}
