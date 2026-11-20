//! Interior identity of a surveyed triangle profile with a triangular reach.
//!
//! Frozen criterion, "Interior identity, triangle": the triangle profile against a triangular reach with `first-order-reference`,
//! same state, `Wall`/`Exterior` ends. At every interior face of the first interval, the
//! surveyed flux enclosure and the triangular certified bracket intersect.
//!
//! The surveyed enclosure is the certified point Riemann bound of the surveyed section, which
//! the surveyed reach solves on its first interval. The triangular bracket holds the face rate
//! of the receipt, and its receipt width bounds the bracket width. So the bracket lies inside
//! `rate ± width`. A dyadic duration makes each face volume the exact rate times duration.

use super::support::{TestResult, forcing, profile, reach};
use super::surveyed_profiles::TRIANGLE_1;
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReach,
    ChannelReachDefinition, ChannelStepDiagnostics, ChannelSurveyedConfig,
    ChannelTriangularReconstruction, CrossSectionBatch, PrismaticRiemannConfig,
    TriangularChannelCell,
};
use numr::dtype::DType;
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, with_interval_arithmetic};

const GRAVITY: f64 = 9.81;
const DURATION: f64 = 1.0 / 4096.0;
/// Areas `h²` and global discharges of the shared start state, one per unit-length cell.
const AREAS: [f64; 6] = [1.0, 0.64, 1.44, 2.25, 0.5, 1.21];
const DISCHARGES: [f64; 6] = [0.3, -0.2, 0.5, 0.0, 0.125, -0.4];

type Runtime = crate::runtime::Runtime;

fn boundaries() -> [ChannelBoundary; 2] {
    [
        ChannelBoundary::Wall,
        ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: 0.0,
        },
    ]
}

fn config() -> ChannelConfig {
    ChannelConfig {
        gravity: GRAVITY,
        triangular_reconstruction: ChannelTriangularReconstruction::FirstOrderReference,
        ..ChannelConfig::default()
    }
}

/// The triangular reach of unit bank slopes, so `A = h²`, with cap depth 4 m.
fn triangular() -> TestResult<ChannelReach<Runtime>> {
    let cells = (0..AREAS.len())
        .map(|index| TriangularChannelCell {
            id: format!("cell-{index}"),
            length_m: 1.0,
            maximum_depth_m: 4.0,
            manning_n: 0.0,
        })
        .collect();
    Ok(ChannelReach::new(
        crate::runtime::client(),
        ChannelReachDefinition {
            vertical_datum: super::support::DATUM.into(),
            provenance: "Triangular identity reference".into(),
            geometry: ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v: 1.0,
                right_bank_slope_h_per_v: 1.0,
                bed_elevation_m: 0.0,
                cells,
            },
            lateral_bindings: Vec::new(),
            config: config(),
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: AREAS.to_vec(),
            longitudinal_momentum_m4_per_s: DISCHARGES.to_vec(),
        },
    )?)
}

/// Exact zero-radius points.
fn points(values: &[f64]) -> TestResult<CertifiedExpansion<Runtime>> {
    let client = crate::runtime::client();
    let n = values.len();
    let high = Tensor::<Runtime>::from_slice(values, &[n], client.device())?;
    let zero = Tensor::<Runtime>::zeros(&[n], DType::F64, client.device())?;
    Ok(with_interval_arithmetic::<Runtime, _, _, _>(
        &client,
        DType::F64,
        "identity cell points",
        |a| a.expansion_bounds(&high, &zero, &zero),
    )?)
}

/// Surveyed water and momentum enclosures `[lower, upper]` at every interior face.
fn surveyed_enclosures() -> TestResult<[Vec<[f64; 2]>; 2]> {
    let client = crate::runtime::client();
    let section = profile(TRIANGLE_1.name, TRIANGLE_1.points, TRIANGLE_1.cap);
    let batch =
        CrossSectionBatch::<Runtime>::new(client.clone(), vec![section])?.certify_prismatic()?;
    let numerical = ChannelSurveyedConfig::default();
    let channel = config();
    // The configuration a surveyed reach derives from its channel and surveyed settings.
    let riemann = PrismaticRiemannConfig {
        gravity_m_per_s2: channel.gravity,
        recovery: numerical.recovery,
        water_flux_atol_m3_per_s: channel.riemann_water_flux_absolute_m3_per_s,
        momentum_flux_atol_m4_per_s2: channel.riemann_momentum_flux_absolute_m4_per_s2,
        flux_rtol: channel.riemann_flux_relative,
        root: numerical.normalized_root.clone(),
        max_bracket_expansions: numerical.max_bracket_expansions,
        initial_simpson_panels: numerical.initial_simpson_panels,
        max_simpson_panels: numerical.max_simpson_panels,
        max_refinements: numerical.max_refinements,
        max_native_query_points: numerical.max_native_query_points,
    };
    let faces = AREAS.len() - 1;
    let owners = Tensor::<Runtime>::from_slice(&vec![0_i64; faces], &[faces], client.device())?;
    let result = batch.riemann_flux_points_tensor(
        &owners,
        &points(&AREAS[..faces])?,
        &points(&DISCHARGES[..faces])?,
        &points(&AREAS[1..])?,
        &points(&DISCHARGES[1..])?,
        &riemann,
    )?;
    let read = |lower: &Tensor<Runtime>, upper: &Tensor<Runtime>| -> TestResult<Vec<[f64; 2]>> {
        let lower = lower.try_to_vec::<f64>()?;
        let upper = upper.try_to_vec::<f64>()?;
        Ok(lower.into_iter().zip(upper).map(|(l, u)| [l, u]).collect())
    };
    let water = result.water_flux_bound_m3_per_s();
    let momentum = result.momentum_flux_bound_m4_per_s2();
    Ok([
        read(water.lower(), water.upper())?,
        read(momentum.lower(), momentum.upper())?,
    ])
}

fn first_interval(reach: ChannelReach<Runtime>) -> TestResult<ChannelStepDiagnostics> {
    let next = reach
        .step(DURATION, &forcing(0.0, DURATION, boundaries()))?
        .next;
    assert_eq!(next.time().to_bits(), DURATION.to_bits(), "one interval");
    Ok(next.last_step()?.ok_or("missing first-interval receipt")?)
}

#[test]
fn triangle_profile_interior_fluxes_intersect_the_triangular_brackets() -> TestResult {
    let [water, momentum] = surveyed_enclosures()?;
    let receipt = first_interval(triangular()?)?;
    let widths = receipt
        .riemann
        .as_ref()
        .ok_or("missing triangular widths")?;
    let surveyed = first_interval(reach::<Runtime>(
        crate::runtime::client(),
        profile(TRIANGLE_1.name, TRIANGLE_1.points, TRIANGLE_1.cap),
        &[1.0; AREAS.len()],
        &[0.0; AREAS.len()],
        &AREAS,
        &DISCHARGES,
        config(),
        ChannelSurveyedConfig::default(),
    )?)?;
    let mut worst: f64 = 0.0;
    for face in 1..AREAS.len() {
        let enclosure = face - 1;
        for (label, bounds, transfers, width, reach_transfers) in [
            (
                "water",
                water[enclosure],
                &receipt.face_volume_m3,
                widths.water_flux_width_m3_per_s[face],
                &surveyed.face_volume_m3,
            ),
            (
                "momentum",
                momentum[enclosure],
                &receipt.face_momentum_m4_per_s,
                widths.momentum_flux_width_m4_per_s2[face],
                &surveyed.face_momentum_m4_per_s,
            ),
        ] {
            let rate = transfers[face] / DURATION;
            let [lower, upper] = bounds;
            assert!(lower <= upper && width >= 0.0);
            assert!(
                lower <= rate + width && rate - width <= upper,
                "face {face} {label}: surveyed [{lower:e}, {upper:e}] misses triangular {rate:e} ± {width:e}"
            );
            worst = worst.max((rate - 0.5 * (lower + upper)).abs());
            // The surveyed reach selects its first-interval rate inside the same enclosure.
            let selected = reach_transfers[face] / DURATION;
            assert!(
                lower <= selected && selected <= upper,
                "face {face} {label}: surveyed reach rate {selected:e} leaves [{lower:e}, {upper:e}]"
            );
        }
    }
    println!(
        "interior identity: faces={}, worst center gap={worst:e}",
        AREAS.len() - 1
    );
    Ok(())
}
