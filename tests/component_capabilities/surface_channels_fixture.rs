//! Surface/channel cases and rejection outcomes for the capability checks.

use crate::surface_channel_support::fixture::{self, Result};
use crate::surface_channel_support::scenario::{Batch, CELL_LENGTH, ReachSpec, Spec, weir};
use crate::surface_channels_variants::{Variant, variant};
use hydr::channel::{ChannelGeometry, TriangularChannelCell};
use hydr::coupling::{
    SurfaceChannelForcingInterval, SurfaceChannelInitialState, SurfaceChannelSetup,
    SurfaceChannelStepDiagnostics,
};
use hydr::soil::{LandCell, LandModel, SoilState};
use hydr::solver::{ForcingInterval, ForcingSchedule};

/// Rejection variant and message, or `None` when the case is accepted.
pub type Outcome = Result<Option<(Variant, String)>>;

/// Requested end of every one-interval case, in seconds.
pub const STEP: f64 = 0.1;

/// Linked reach of every paired case.
pub const REACH: &str = "river";

/// Crest elevation of every paired case, in metres.
pub const CREST: f64 = 1.2;

/// Surface bed elevation of every paired case, in metres.
const SURFACE_BED: f64 = 1.0;

/// One surface cell at `surface_stage` beside a two-cell reach at `channel_stage`.
///
/// The bank weir joins the surface cell to the first channel cell. The channel bed is zero.
pub fn pair(surface_stage: f64, channel_stage: f64) -> Spec {
    Spec::new(1, 1, SURFACE_BED, surface_stage - SURFACE_BED)
        .reach(ReachSpec::new(REACH, 0.0, &[channel_stage; 2]))
        .link(weir("bank", 0, REACH, 0, CREST, 2.0))
}

/// Builds the coordinator and returns its rejection, or `None` when it admits the case.
pub fn construction_outcome(
    setup: SurfaceChannelSetup,
    initial: SurfaceChannelInitialState,
) -> Outcome {
    Ok(match Batch::new(crate::runtime::client(), setup, initial) {
        Ok(_) => None,
        Err(error) => Some(variant(&error)),
    })
}

/// The fixture case after `edit`, built once.
pub fn edited(edit: impl FnOnce(&mut SurfaceChannelSetup)) -> Outcome {
    let (mut setup, initial) = fixture::case()?;
    edit(&mut setup);
    construction_outcome(setup, initial)
}

/// Runs one trial of `batch` and fails when a rejection changes the batch.
pub fn trial_outcome(batch: &Batch, end: f64, forcing: &SurfaceChannelForcingInterval) -> Outcome {
    let before = batch.snapshot_data()?;
    let outcome = match batch.trial_to(end, forcing) {
        Ok(_) => None,
        Err(error) => Some(variant(&error)),
    };
    assert_eq!(batch.snapshot_data()?, before, "a trial changed the batch");
    Ok(outcome)
}

/// Accepts one interval of `batch` and returns the next batch with its diagnostics.
pub fn accept(
    batch: &Batch,
    end: f64,
    forcing: &SurfaceChannelForcingInterval,
) -> Result<(Batch, SurfaceChannelStepDiagnostics)> {
    let next = batch.trial_to(end, forcing)?.into_next();
    let step = next
        .last_step()?
        .ok_or("the batch has no accepted interval")?;
    Ok((next, step))
}

/// `spec` with uniform surface `rain` and potential `evaporation`, both in m/s.
pub fn with_rates(
    spec: &Spec,
    rain: f64,
    evaporation: f64,
) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let (mut setup, initial) = spec.parts()?;
    let surface = &mut setup.surface;
    let count = surface.domain.grid().cell_count();
    surface.forcing = ForcingSchedule::new(
        &surface.domain,
        &surface.boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 100.0,
            rain: vec![rain; count],
            potential_evaporation: vec![evaporation; count],
            withdrawal: spec.withdrawal.clone(),
            boundaries: Vec::new(),
        }],
    )?;
    Ok((setup, initial))
}

/// `spec` with pinned Modified Green-Ampt soil on every surface cell.
///
/// The soil parameters are the ones the soil and land-source tests use.
pub fn green_ampt(spec: &Spec) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let (mut setup, mut initial) = spec.parts()?;
    let surface = &mut setup.surface;
    let count = surface.domain.grid().cell_count();
    let soil = LandCell::ModifiedGreenAmpt(crate::runoff_fixture::soil_parameters()?);
    surface.land = LandModel::new(&surface.domain, vec![soil; count])?;
    initial.surface.soil = SoilState::initialized(&surface.land)?;
    Ok((setup, initial))
}

/// `spec` with every reach replaced by a 1:1 triangular reach at the same depths.
pub fn triangular(spec: &Spec) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let (mut setup, mut initial) = spec.parts()?;
    for (reach, given) in setup.reaches.iter_mut().zip(&spec.reaches) {
        reach.definition.geometry = ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 1.0,
            right_bank_slope_h_per_v: 1.0,
            bed_elevation_m: given.bed,
            cells: (0..given.depths.len())
                .map(|i| TriangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: CELL_LENGTH,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                })
                .collect(),
        };
    }
    for (id, state) in &mut initial.reaches {
        let given = spec
            .reaches
            .iter()
            .find(|reach| reach.id == *id)
            .ok_or("initial state names no reach")?;
        // A 1:1 triangle holds `depth²` of area per metre of length.
        state.volume_m3 = given.depths.iter().map(|d| d * d * CELL_LENGTH).collect();
    }
    Ok((setup, initial))
}
