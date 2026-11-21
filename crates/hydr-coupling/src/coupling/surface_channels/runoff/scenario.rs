//! Runoff test cases: the structure tanks with one pond and one upstream subcatchment.
//!
//! The subcatchment owns the surface cell's support, so the cell's rain is suppressed. The pond
//! joins the surface cell through a weir and takes direct rain over its own atom. The outlet
//! drains into the reach cell unless a case changes its destination.

use super::{
    SurfaceChannelFreeOutlet, SurfaceChannelRunoff, SurfaceChannelRunoffDestination,
    SurfaceChannelRunoffOutlet, SurfaceChannelRunoffShare,
};
use crate::channel::ChannelBoundary;
use crate::coupling::surface_channels::pools::scenario::{
    declaration, link, nodes_setup, pond, pool, surface,
};
use crate::coupling::surface_channels::structures::scenario::{
    AREA, Batch, REACH, Tanks, TestResult, client, inflow, weir_law, withdrawal,
};
use crate::coupling::surface_channels::{
    SurfaceChannelBatch, SurfaceChannelDefinition, SurfaceChannelError,
    SurfaceChannelForcingInterval, SurfaceChannelInitialState, SurfaceChannelReachForcing,
    SurfaceChannelSetup,
};
use crate::runoff::{
    CellRainAssignment, Contribution, ContributionKind, PartitionAssertion, PoolRainSupport,
    SubcatchmentConfig, SubcatchmentContributionInput, SubcatchmentDefinition,
    SubcatchmentGeometry, SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting,
    SubcatchmentState, SupportAtom, SupportPlanInput, SurfaceCellSupport,
};
use crate::structures::{DirectionPolicy, StructureOperatingPoint};

/// Identity of the one modelled source.
pub const SOURCE: &str = "upstream-source";
/// Outlet identity of the one modelled source.
pub const OUTLET: &str = "upstream-outlet";
/// Identity of the one pond.
pub const POND: &str = "pond";

/// One impervious subcatchment of `area` m² without depression storage.
pub fn definition(area: f64) -> TestResult<SubcatchmentDefinition> {
    Ok(SubcatchmentDefinition {
        id: "upstream-model".into(),
        parameters: SubcatchmentParameters::new(
            SubcatchmentGeometry {
                total_area_m2: area,
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

fn atom(id: &str, area_m2: f64) -> SupportAtom {
    SupportAtom {
        id: id.into(),
        area_m2,
    }
}

/// The outlet into reach cell 0.
pub fn reach_outlet() -> SurfaceChannelRunoffOutlet {
    outlet(SurfaceChannelRunoffDestination::ChannelCells {
        reach_id: REACH.into(),
        cells: vec![share(0, 1.0)],
    })
}

/// The outlet into `destination` with a valid free-outlet attestation.
pub fn outlet(destination: SurfaceChannelRunoffDestination) -> SurfaceChannelRunoffOutlet {
    SurfaceChannelRunoffOutlet {
        outlet_id: OUTLET.into(),
        destination,
        free_outlet: SurfaceChannelFreeOutlet {
            invert_elevation_m: 3.0,
            evidence: "The culvert outfall sits above the receiving water".into(),
        },
    }
}

/// One target cell with `weight`.
pub fn share(index: usize, weight: f64) -> SurfaceChannelRunoffShare {
    SurfaceChannelRunoffShare { index, weight }
}

/// Runoff whose source owns the surface cell, with direct rain on the pond.
pub fn runoff() -> TestResult<SurfaceChannelRunoff> {
    Ok(SurfaceChannelRunoff {
        support: SupportPlanInput {
            partition: PartitionAssertion {
                identity: "tank-partition".into(),
                covered_support: "the surface cell and the pond".into(),
                provenance: "Explicit complete disjoint synthetic partition".into(),
                complete_disjoint: true,
            },
            surface_input_identity: "surface-input".into(),
            atoms: vec![atom("upstream", AREA), atom("pond-surface", 1.0)],
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
        definitions: vec![definition(AREA)?],
        config: SubcatchmentConfig::default(),
        outlets: vec![reach_outlet()],
    })
}

/// The tanks with the pond joined to the surface cell, and `pond_sources` source policies.
pub fn pond_parts(
    pond_sources: bool,
) -> TestResult<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let (setup, mut initial) = Tanks::at(0.0, 0.5).parts()?;
    let spill = link(
        "spill",
        weir_law(1.0)?,
        DirectionPolicy::Bidirectional,
        StructureOperatingPoint::Passive,
        (pool(POND), 0.0),
        (surface(), 0.0),
    );
    let pools = declaration(vec![pond(POND, 0.0, AREA, 2.0)?], pond_sources, Vec::new());
    initial.pools = vec![crate::storage::LevelPoolState { volume_m3: 0.0 }];
    Ok((nodes_setup(setup, pools, vec![spill]), initial))
}

/// The pond case with the default runoff and its initialized runoff state.
pub fn parts() -> TestResult<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let (mut setup, mut initial) = pond_parts(false)?;
    let runoff = runoff()?;
    initial.runoff = runoff
        .definitions
        .iter()
        .map(|definition| SubcatchmentState::initialized(&definition.parameters))
        .collect::<Result<_, _>>()?;
    setup.runoff = Some(runoff);
    Ok((setup, initial))
}

/// The batch of `setup` at `initial`.
pub fn build(
    (setup, initial): (SurfaceChannelSetup, SurfaceChannelInitialState),
) -> Result<Batch, SurfaceChannelError> {
    SurfaceChannelBatch::new(client(), setup, initial)
}

/// The input message of a rejected definition of `setup`.
pub fn rejection(setup: SurfaceChannelSetup) -> TestResult<String> {
    match SurfaceChannelDefinition::new(setup) {
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
        Ok(_) => Err("the definition admitted invalid runoff".into()),
    }
}

/// The input message of a rejected `result`.
pub fn input_message<T>(result: Result<T, SurfaceChannelError>) -> TestResult<String> {
    match result {
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
        Ok(_) => Err("the call admitted invalid runoff".into()),
    }
}

/// Fails unless `message` starts with the rule `entry`.
pub fn starts(message: &str, entry: &str) {
    assert!(message.starts_with(entry), "{entry}: {message}");
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

/// Forcing of `batch` toward `start + dt` with zero caller inputs and `runoff` records.
pub fn forcing(
    batch: &Batch,
    dt: f64,
    runoff: Vec<SubcatchmentContributionInput>,
) -> SurfaceChannelForcingInterval {
    let start = batch.time();
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: start + dt,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: Some(ChannelBoundary::Wall),
            right: Some(ChannelBoundary::Wall),
            lateral_inputs: vec![inflow(0.0), withdrawal(0.0)],
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff,
    }
}
