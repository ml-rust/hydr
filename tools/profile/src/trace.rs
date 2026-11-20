//! Certified surveyed scalar endpoint traces: the frozen three-band reference rows.
//!
//! Each interval takes the next three-band row of `MULTIBAND`: a one-cell reach at the row's
//! interior state, the row's policy at its end, and a wall at the other end, stepped once over
//! a span far below the end-face CFL limit. These are the rows of `endpoints_traces.rs`.

use crate::alloc::Totals;
use crate::endpoints_reference::{End, MULTIBAND, Trace};
use crate::runtime::{Runtime, client};
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelGeometry, ChannelInitialState,
    ChannelReach, ChannelReachDefinition, ChannelSurveyedConfig, CrossSectionDefinition,
    CrossSectionPoint, SurveyedChannelCell,
};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Profile name of the reference rows.
const PROFILE: &str = "three-bands";

/// Span of one interval, in seconds: one step far below the end-face CFL limit.
const SPAN_S: f64 = 1.0 / 1048576.0;

/// The one-cell reach of `row` at its interior state.
fn reach(row: &Trace, probe: solvr::optimize::CertifiedRootProbe) -> Result<ChannelReach<Runtime>> {
    let mut numerical = ChannelSurveyedConfig::default();
    numerical.normalized_root.probe = probe;
    let profile = CrossSectionDefinition {
        id: row.profile.name.into(),
        vertical_datum: "Synthetic SI datum".into(),
        points: row
            .profile
            .points
            .iter()
            .map(|point| CrossSectionPoint {
                station_m: f64::from_bits(point[0]),
                elevation_m: f64::from_bits(point[1]),
            })
            .collect(),
        maximum_stage_m: f64::from_bits(row.profile.cap),
        profile_provenance: "Frozen original-coordinate acceptance survey".into(),
    };
    let definition = ChannelReachDefinition {
        vertical_datum: "Synthetic SI datum".into(),
        provenance: "Surveyed trace profile fixture".into(),
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile,
            cells: vec![SurveyedChannelCell {
                id: "cell-0".into(),
                length_m: 1.0,
                manning_n: 0.0,
            }],
            numerical,
        },
        lateral_bindings: Vec::new(),
        config: ChannelConfig {
            gravity: row.gravity,
            ..ChannelConfig::default()
        },
    };
    Ok(ChannelReach::new(
        client(),
        definition,
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![row.interior_area_m2],
            longitudinal_momentum_m4_per_s: vec![row.interior_discharge_m3_per_s],
        },
    )?)
}

/// Steps `intervals` reference rows and prints one receipt row each.
pub fn run(intervals: usize, probe: solvr::optimize::CertifiedRootProbe) -> Result {
    let rows: Vec<&Trace> = MULTIBAND
        .iter()
        .filter(|row| row.profile.name == PROFILE)
        .collect();
    println!(
        "interval row alloc_calls alloc_bytes root_solves root_iterations residual_points simpson_points refinements panels digest"
    );
    for index in 0..intervals {
        let position = index % rows.len();
        let row = rows[position];
        let origin = reach(row, probe)?;
        let policy = row.policy.boundary();
        let (left, right) = match row.end {
            End::Left => (policy, ChannelBoundary::Wall),
            End::Right => (ChannelBoundary::Wall, policy),
        };
        let forcing = ChannelForcingInterval {
            start_s: 0.0,
            end_s: SPAN_S,
            left,
            right,
            lateral_inputs: Vec::new(),
        };
        let before = Totals::now();
        let next = origin.step(SPAN_S, &forcing)?.next;
        let allocations = Totals::now().since(before);
        let receipt = next.last_step()?.ok_or("missing trace receipt")?;
        let digest = crate::digest(&format!("{receipt:?} {:?}", next.snapshot_data()?));
        let work = receipt
            .riemann
            .and_then(|riemann| riemann.surveyed_work)
            .ok_or("missing surveyed work")?;
        println!(
            "{index} {position} {} {} {} {} {} {} {} {} {digest:016x}",
            allocations.calls,
            allocations.bytes,
            work.root_solves,
            work.root_iterations,
            work.residual_points,
            work.simpson_value_points,
            work.refinements,
            work.maximum_simpson_panels
        );
    }
    Ok(())
}
