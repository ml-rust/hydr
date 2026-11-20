//! Surveyed reaches for the channel/pool attachment tests.

use super::fixture::{DATUM, Result, config, connection, forcing, pool};
use super::reference::{CASES, Case};
use super::surveyed_profiles::{Profile, RECTANGLE_1, RECTANGLE_3, THREE_BANDS, TRIANGLE_1};
use super::surveyed_reference::SurveyedCase;
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
    ChannelSurveyedConfig, CrossSectionDefinition, CrossSectionPoint, SurveyedChannelCell,
};
use hydr::coupling::{
    ChannelPoolEndpoint, ChannelPoolForcingInterval, ChannelPoolInitialState, ChannelPoolReach,
    ChannelPoolSetup,
};
use hydr::storage::LevelPoolState;

/// The survey section of a frozen profile.
pub fn section(profile: Profile) -> CrossSectionDefinition {
    CrossSectionDefinition {
        id: profile.name.into(),
        vertical_datum: DATUM.into(),
        points: profile
            .points
            .iter()
            .map(|point| CrossSectionPoint {
                station_m: f64::from_bits(point[0]),
                elevation_m: f64::from_bits(point[1]),
            })
            .collect(),
        maximum_stage_m: f64::from_bits(profile.cap),
        profile_provenance: "Frozen surveyed channel/pool reference profile".into(),
    }
}

/// `count` surveyed cells of `length` metres on one frozen profile, bed at elevation 0.
pub fn definition(
    profile: Profile,
    gravity: f64,
    count: usize,
    length: f64,
) -> ChannelReachDefinition {
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "Explicit surveyed finite-pool reach".into(),
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile: section(profile),
            cells: (0..count)
                .map(|i| SurveyedChannelCell {
                    id: format!("cell-{i}"),
                    length_m: length,
                    manning_n: 0.0,
                })
                .collect(),
            numerical: ChannelSurveyedConfig::default(),
        },
        lateral_bindings: Vec::new(),
        config: ChannelConfig {
            gravity,
            max_step: 1.0,
            ..ChannelConfig::default()
        },
    }
}

/// The surveyed profile that reproduces a rectangular or triangular frozen case.
///
/// The rectangle of width `coefficient` and the triangle of side slope `coefficient`.
pub fn case_profile(case: &Case) -> Profile {
    if case.triangle {
        TRIANGLE_1
    } else if case.coefficient == 3.0 {
        RECTANGLE_3
    } else {
        RECTANGLE_1
    }
}

/// A frozen rectangular or triangular case as the same case on its surveyed profile.
pub fn surveyed_case(case: &Case) -> SurveyedCase {
    SurveyedCase {
        name: case.name,
        profile: case_profile(case),
        gravity: case.gravity,
        interior_area_m2: case.coefficient,
        cell_length_m: 128.0,
        elevation: case.elevation,
        loss: case.loss,
        pool_area: case.pool_area,
        dt: 0.125,
        expected: case.expected,
    }
}

/// Every frozen rectangular and triangular case on its surveyed profile.
pub fn rectangle_and_triangle_cases() -> Vec<SurveyedCase> {
    CASES.iter().map(surveyed_case).collect()
}

/// Depth at which the frozen case starts its reach cell at rest.
pub fn rest_depth(case: &SurveyedCase) -> f64 {
    if case.profile.name == THREE_BANDS.name {
        1.5
    } else {
        1.0
    }
}

/// The interval over `[start_s, end_s]`. The unconnected end holds water at rest at `depth_m`.
pub fn held(
    setup: &ChannelPoolSetup,
    depth_m: f64,
    start_s: f64,
    end_s: f64,
) -> ChannelPoolForcingInterval {
    let mut interval = forcing(setup, start_s, end_s);
    let reach = &mut interval.reaches[0];
    for side in [&mut reach.left, &mut reach.right] {
        if side.is_some() {
            *side = Some(ChannelBoundary::Exterior {
                depth_m,
                velocity_m_per_s: 0.0,
            });
        }
    }
    interval
}

/// One surveyed cell and one constant-area pool joined at `endpoint`.
pub fn single(
    case: &SurveyedCase,
    endpoint: ChannelPoolEndpoint,
) -> Result<(ChannelPoolSetup, ChannelPoolInitialState)> {
    Ok((
        ChannelPoolSetup {
            reaches: vec![ChannelPoolReach {
                id: "reach".into(),
                definition: definition(case.profile, case.gravity, 1, case.cell_length_m),
            }],
            pools: vec![pool("pool", case.pool_area, 0.0)?],
            connections: vec![connection(
                "connection",
                "reach",
                endpoint,
                "pool",
                case.loss,
            )],
            vertical_datum: DATUM.into(),
            nonoverlap_assertion: "Basin and reach occupy separate physical storage".into(),
            config: config(),
        },
        ChannelPoolInitialState {
            time_s: 0.0,
            reaches: vec![ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![case.cell_length_m * case.interior_area_m2],
                longitudinal_momentum_m4_per_s: vec![0.0],
            }],
            pools: vec![LevelPoolState {
                volume_m3: case.pool_area * case.elevation,
            }],
        },
    ))
}
