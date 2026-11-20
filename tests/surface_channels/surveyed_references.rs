//! Frozen channel/pool references on surveyed profiles, for the surface/channel tests.
//!
//! The reference tables live under `tests/channel_pools/` and `tests/support/`. This module
//! loads them beside the surface fixtures and builds surveyed reaches from them.

pub(crate) use crate::channel_pool_reference as reference;
#[path = "../support/surveyed_profiles.rs"]
pub mod surveyed_profiles;
#[path = "../channel_pools/surveyed_reference.rs"]
pub mod surveyed_reference;

use crate::pools_fixture::DATUM;
use hydr::channel::{
    ChannelConfig, ChannelGeometry, ChannelReachDefinition, ChannelSurveyedConfig,
    CrossSectionDefinition, CrossSectionPoint, SurveyedChannelCell,
};
use surveyed_profiles::{Profile, RECTANGLE_1, RECTANGLE_3, THREE_BANDS, TRIANGLE_1};
pub use surveyed_reference::{SurveyedCase, THREE_BANDS_CASES};

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

/// A frozen rectangular or triangular case as the same case on its surveyed profile.
///
/// The rectangle has width `coefficient` and the triangle has side slope `coefficient`.
pub fn surveyed_case(case: &reference::Case) -> SurveyedCase {
    let profile = if case.triangle {
        TRIANGLE_1
    } else if case.coefficient == 3.0 {
        RECTANGLE_3
    } else {
        RECTANGLE_1
    };
    SurveyedCase {
        name: case.name,
        profile,
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

/// Depth at which a frozen case starts its reach cell at rest.
pub fn rest_depth(case: &SurveyedCase) -> f64 {
    if case.profile.name == THREE_BANDS.name {
        1.5
    } else {
        1.0
    }
}
