use super::fixture_model::{Config, Execution};
pub(super) const CONFIGS: &[Config] = &[
    Config {
        floats: [
            0x3ff0000000000000,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3e112e0be826d695,
            0x3e45798ee2308c3a,
            0x3ddb7cdfd9d7bdbb,
            0x3d719799812dea11,
            0x3d719799812dea11,
        ],
        counts: [256, 1048576, 4096, 4, 4096, 10, 16777216],
    },
    Config {
        floats: [
            0x40239eb851eb851f,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3e112e0be826d695,
            0x3e45798ee2308c3a,
            0x3ddb7cdfd9d7bdbb,
            0x3d719799812dea11,
            0x3d719799812dea11,
        ],
        counts: [256, 1048576, 4096, 4, 4096, 10, 16777216],
    },
    Config {
        floats: [
            0x4008000000000000,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3e112e0be826d695,
            0x3e45798ee2308c3a,
            0x3ddb7cdfd9d7bdbb,
            0x3d719799812dea11,
            0x3d719799812dea11,
        ],
        counts: [256, 1048576, 4096, 4, 4096, 10, 16777216],
    },
    Config {
        floats: [
            0x4047000000000000,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3e112e0be826d695,
            0x3e45798ee2308c3a,
            0x3ddb7cdfd9d7bdbb,
            0x3d719799812dea11,
            0x3d719799812dea11,
        ],
        counts: [256, 1048576, 4096, 4, 4096, 10, 16777216],
    },
    Config {
        floats: [
            0x3ff0000000000000,
            0x3ff0000000000000,
            0x3ff0000000000000,
            0x3e112e0be826d695,
            0x3e45798ee2308c3a,
            0x3ddb7cdfd9d7bdbb,
            0x3d719799812dea11,
            0x3d719799812dea11,
        ],
        counts: [256, 1048576, 4096, 4, 4096, 10, 16777216],
    },
    Config {
        floats: [
            0x3ff0000000000000,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3e112e0be826d695,
            0x3e45798ee2308c3a,
            0x3ddb7cdfd9d7bdbb,
            0x3d719799812dea11,
            0x3d719799812dea11,
        ],
        counts: [256, 1048576, 4096, 4, 4096, 10, 0],
    },
    Config {
        floats: [
            0x3ff0000000000000,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3cd203af9ee75616,
            0x3cd203af9ee75616,
        ],
        counts: [256, 1048576, 4096, 4, 4096, 10, 16777216],
    },
    Config {
        floats: [
            0x40239eb851eb851f,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3d719799812dea11,
            0x3cd203af9ee75616,
            0x3cd203af9ee75616,
        ],
        counts: [256, 1048576, 4096, 4, 4096, 10, 16777216],
    },
];
#[path = "point_candidates_fixture_admission_and_capacity.rs"]
mod admission_and_capacity;
#[path = "point_candidates_fixture_depth_transitions.rs"]
mod depth_transitions;
#[path = "point_candidates_fixture_knot_neighbors.rs"]
mod knot_neighbors;
#[path = "point_candidates_fixture_minimum_uniform_states.rs"]
mod minimum_uniform_states;
#[path = "point_candidates_fixture_multiband_dry_states.rs"]
mod multiband_dry_states;
#[path = "point_candidates_fixture_native_continuation.rs"]
mod native_continuation;
#[path = "point_candidates_fixture_native_origin_lower_faces.rs"]
mod native_origin_lower_faces;
#[path = "point_candidates_fixture_native_origin_upper_faces.rs"]
mod native_origin_upper_faces;
#[path = "point_candidates_fixture_rectangle_dry_fans.rs"]
mod rectangle_dry_fans;
#[path = "point_candidates_fixture_retained_capacity.rs"]
mod retained_capacity;
#[path = "point_candidates_fixture_retained_continuation.rs"]
mod retained_continuation;
#[path = "point_candidates_fixture_retained_knots.rs"]
mod retained_knots;
#[path = "point_candidates_fixture_retained_origin_lower_faces.rs"]
mod retained_origin_lower_faces;
#[path = "point_candidates_fixture_retained_origin_upper_faces.rs"]
mod retained_origin_upper_faces;
#[path = "point_candidates_fixture_shock_and_weak_transitions.rs"]
mod shock_and_weak_transitions;
#[path = "point_candidates_fixture_stationary_shocks.rs"]
mod stationary_shocks;
#[path = "point_candidates_fixture_tiny_collision.rs"]
mod tiny_collision;
#[path = "point_candidates_fixture_tiny_fan_and_film.rs"]
mod tiny_fan_and_film;
#[path = "point_candidates_fixture_tiny_upwind.rs"]
mod tiny_upwind;
#[path = "point_candidates_fixture_triangle_dry_fans.rs"]
mod triangle_dry_fans;
#[path = "point_candidates_fixture_uniform_sections.rs"]
mod uniform_sections;
#[path = "point_candidates_fixture_vacuum_limits.rs"]
mod vacuum_limits;
#[path = "point_candidates_fixture_wet_fans.rs"]
mod wet_fans;
pub(super) fn executions() -> Vec<Execution> {
    let mut rows = Vec::with_capacity(70);
    rows.extend_from_slice(uniform_sections::ROWS);
    rows.extend_from_slice(depth_transitions::ROWS);
    rows.extend_from_slice(shock_and_weak_transitions::ROWS);
    rows.extend_from_slice(rectangle_dry_fans::ROWS);
    rows.extend_from_slice(triangle_dry_fans::ROWS);
    rows.extend_from_slice(multiband_dry_states::ROWS);
    rows.extend_from_slice(vacuum_limits::ROWS);
    rows.extend_from_slice(stationary_shocks::ROWS);
    rows.extend_from_slice(knot_neighbors::ROWS);
    rows.extend_from_slice(admission_and_capacity::ROWS);
    rows.extend_from_slice(wet_fans::ROWS);
    rows.extend_from_slice(native_origin_lower_faces::ROWS);
    rows.extend_from_slice(native_origin_upper_faces::ROWS);
    rows.extend_from_slice(native_continuation::ROWS);
    rows.extend_from_slice(retained_origin_lower_faces::ROWS);
    rows.extend_from_slice(retained_origin_upper_faces::ROWS);
    rows.extend_from_slice(retained_continuation::ROWS);
    rows.extend_from_slice(retained_knots::ROWS);
    rows.extend_from_slice(retained_capacity::ROWS);
    rows.extend_from_slice(tiny_collision::ROWS);
    rows.extend_from_slice(minimum_uniform_states::ROWS);
    rows.extend_from_slice(tiny_fan_and_film::ROWS);
    rows.extend_from_slice(tiny_upwind::ROWS);
    rows
}
