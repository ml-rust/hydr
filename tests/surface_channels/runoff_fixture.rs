//! Native runoff fixtures through the public coordinator API.
//!
//! The tanks of `pools_fixture` gain one pond and one impervious subcatchment without
//! depression storage. The pond joins the surface cell through a weir whose crest its water
//! never reaches. The subcatchment owns the surface cell's support, so that cell's rain is
//! suppressed. The pond takes direct rain over its own atom. The outlet drains into the pond,
//! the reach cell, or the surface cell. A surface cell case without the pond and its spill
//! leaves the coordinator coupled by runoff alone.

use super::pools_fixture::{
    AREA, Case, INFLOW, REACH, Result, WITHDRAWAL, manual, pond, pool, structure, surface, weir_law,
};
use hydr::channel::{ChannelBoundary, ChannelLateralInput};
use hydr::coupling::{
    SurfaceChannelForcingInterval, SurfaceChannelFreeOutlet, SurfaceChannelInitialState,
    SurfaceChannelReachForcing, SurfaceChannelRunoff, SurfaceChannelRunoffDestination,
    SurfaceChannelRunoffOutlet, SurfaceChannelRunoffShare, SurfaceChannelSetup,
};
use hydr::forcing::WaterAmount;
use hydr::runoff::{
    CellRainAssignment, Contribution, ContributionKind, PartitionAssertion, PoolRainSupport,
    SubcatchmentConfig, SubcatchmentContributionInput, SubcatchmentDefinition,
    SubcatchmentGeometry, SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting,
    SubcatchmentState, SupportAtom, SupportPlanInput, SurfaceCellSupport,
};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint};

/// Identity of the one modelled source.
pub const SOURCE: &str = "upstream-source";
/// Outlet identity of the one modelled source.
pub const OUTLET: &str = "upstream-outlet";
/// Identity of the one subcatchment definition.
pub const MODEL: &str = "upstream-model";
/// Identity of the one pond.
pub const POND: &str = "pond";
/// Initial runoff depth on the subcatchment, in m.
pub const DEPTH: f64 = 0.01;

/// Receiving node of the outlet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Receiver {
    /// The pond.
    Pool,
    /// The reach cell.
    Reach,
    /// The surface cell, which the spill links to the pond.
    Surface,
    /// The surface cell of the tanks without the pond and its spill. The cell is a runoff-only
    /// lane, and runoff alone couples the coordinator.
    UnlinkedSurface,
}

/// Every receiving node: the pond, the reach cell, and the surface cell with and without a
/// link.
pub const RECEIVERS: [Receiver; 4] = [
    Receiver::Pool,
    Receiver::Reach,
    Receiver::Surface,
    Receiver::UnlinkedSurface,
];

/// The impervious subcatchment over `AREA` m².
pub fn definition() -> Result<SubcatchmentDefinition> {
    Ok(SubcatchmentDefinition {
        id: MODEL.into(),
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

/// The outlet into `receiver` with its invert at `invert` m.
pub fn outlet(receiver: Receiver, invert: f64) -> SurfaceChannelRunoffOutlet {
    SurfaceChannelRunoffOutlet {
        outlet_id: OUTLET.into(),
        destination: match receiver {
            Receiver::Pool => SurfaceChannelRunoffDestination::Pool {
                pool_id: POND.into(),
            },
            Receiver::Reach => SurfaceChannelRunoffDestination::ChannelCells {
                reach_id: REACH.into(),
                cells: vec![SurfaceChannelRunoffShare {
                    index: 0,
                    weight: 1.0,
                }],
            },
            Receiver::Surface | Receiver::UnlinkedSurface => {
                SurfaceChannelRunoffDestination::SurfaceCells {
                    cells: vec![SurfaceChannelRunoffShare {
                        index: 0,
                        weight: 1.0,
                    }],
                }
            }
        },
        free_outlet: SurfaceChannelFreeOutlet {
            invert_elevation_m: invert,
            evidence: "The outfall pipe sits above the receiving water".into(),
        },
    }
}

/// Runoff whose source owns the surface cell, with direct rain on the pond.
///
/// Into `UnlinkedSurface` the plan holds no pond atom and no pool rain.
pub fn runoff(receiver: Receiver) -> Result<SurfaceChannelRunoff> {
    let atom = |id: &str, area_m2| SupportAtom {
        id: id.into(),
        area_m2,
    };
    let mut runoff = SurfaceChannelRunoff {
        support: SupportPlanInput {
            partition: PartitionAssertion {
                identity: "tank-partition".into(),
                covered_support: "the surface cell and the pond".into(),
                provenance: "Explicit complete disjoint synthetic partition".into(),
                complete_disjoint: true,
            },
            surface_input_identity: "surface-input".into(),
            atoms: vec![atom("upstream", AREA), atom("pond-surface", AREA)],
            contributions: vec![Contribution {
                id: SOURCE.into(),
                input_identity: "upstream-input".into(),
                provenance: "Explicit synthetic rainfall support".into(),
                atoms: vec!["upstream".into()],
                includes: Vec::new(),
                kind: ContributionKind::ModelledRunoff {
                    definition_id: MODEL.into(),
                    outlet_id: OUTLET.into(),
                },
            }],
            replacements: Vec::new(),
            surface_cells: vec![SurfaceCellSupport {
                cell: 0,
                atoms: vec!["upstream".into()],
                rain: CellRainAssignment::Suppressed {
                    selected_contribution_id: SOURCE.into(),
                },
            }],
            boundaries: Vec::new(),
            pool_rain: vec![PoolRainSupport {
                pool_id: POND.into(),
                atoms: vec!["pond-surface".into()],
                rain: CellRainAssignment::Direct,
            }],
        },
        definitions: vec![definition()?],
        config: SubcatchmentConfig::default(),
        outlets: vec![outlet(receiver, 3.0)],
    };
    if receiver == Receiver::UnlinkedSurface {
        let support = &mut runoff.support;
        support.partition.covered_support = "the surface cell".into();
        support.atoms.retain(|atom| atom.id != "pond-surface");
        support.pool_rain.clear();
    }
    Ok(runoff)
}

/// The tanks with a dry surface cell, the reach at 0.5 m, an empty pond, and runoff into
/// `receiver` from `DEPTH` of initial runoff.
///
/// Into `UnlinkedSurface` the tanks hold no pond and no spill.
pub fn parts(receiver: Receiver) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let mut case = Case::tanks(0.0, 0.5);
    if receiver != Receiver::UnlinkedSurface {
        case.ponds = vec![(pond(POND, 0.0, AREA, 2.0)?, 0.0)];
        case.structures = vec![structure(
            "spill",
            weir_law(1.0, 1.0)?,
            DirectionPolicy::Bidirectional,
            manual(StructureOperatingPoint::Passive),
            (pool(POND), 0.0),
            (surface(), 0.0),
        )];
    }
    let (mut setup, mut initial) = case.parts()?;
    setup.runoff = Some(runoff(receiver)?);
    let mut state = SubcatchmentState::initialized(&definition()?.parameters)?;
    state.depths_m[0] = DEPTH;
    initial.runoff = vec![state];
    Ok((setup, initial))
}

/// One rate record of the source over `[start, end]` with `rain` m/s.
pub fn record(start: f64, end: f64, rain: f64) -> SubcatchmentContributionInput {
    SubcatchmentContributionInput {
        contribution_id: SOURCE.into(),
        input_identity: "upstream-input".into(),
        start,
        end,
        input: SubcatchmentInput {
            rain_rate_m_per_s: rain,
            potential_evaporation_m_per_s: 0.0,
            external_runon_volumes_m3: [0.0; 3],
        },
    }
}

/// Forcing over `[start, end]` with zero caller rates, walls, and `rain` m/s on the
/// subcatchment.
pub fn forcing(start: f64, end: f64, rain: f64) -> SurfaceChannelForcingInterval {
    let rate = WaterAmount::Rate { rate_m3_per_s: 0.0 };
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: Some(ChannelBoundary::Wall),
            right: Some(ChannelBoundary::Wall),
            lateral_inputs: vec![
                ChannelLateralInput::Inflow {
                    boundary_id: INFLOW.into(),
                    amount: rate,
                    longitudinal_velocity_m_per_s: 0.0,
                },
                ChannelLateralInput::Withdrawal {
                    boundary_id: WITHDRAWAL.into(),
                    amount: rate,
                },
            ],
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff: vec![record(start, end, rain)],
    }
}
