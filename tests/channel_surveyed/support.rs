use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelGeometry, ChannelInitialState,
    ChannelReach, ChannelReachDefinition, ChannelSurveyedConfig, CrossSectionDefinition,
    CrossSectionPoint, SurveyedChannelCell,
};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps,
    TensorOps, TwoSumOps,
};
use numr::runtime::Runtime;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedMappedBisectionAlgorithms;
use solvr::optimize::roots::{
    BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms,
};

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;
pub type Reach = ChannelReach<crate::runtime::Runtime>;
pub const DATUM: &str = "Synthetic SI datum";

pub trait Client<R: Runtime<DType = DType>>:
    TensorOps<R>
    + CompareOps<R>
    + CorrectlyRoundedOps<R>
    + HypotOps<R>
    + MidpointOps<R>
    + NextRepresentableOps<R>
    + ScalarOps<R>
    + TwoSumOps<R>
    + CertifiedSimpsonAlgorithms<R>
    + CertifiedBisectionAlgorithms<R>
    + CertifiedMappedBisectionAlgorithms<R>
    + CertifiedUpperBracketAlgorithms<R>
    + BatchedBisectionAlgorithms<R>
{
}

impl<R, C> Client<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + CorrectlyRoundedOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
}

pub fn profile(id: &str, points: &[[u64; 2]], cap: u64) -> CrossSectionDefinition {
    CrossSectionDefinition {
        id: id.into(),
        vertical_datum: DATUM.into(),
        points: points
            .iter()
            .map(|point| CrossSectionPoint {
                station_m: f64::from_bits(point[0]),
                elevation_m: f64::from_bits(point[1]),
            })
            .collect(),
        maximum_stage_m: f64::from_bits(cap),
        profile_provenance: "Frozen original-coordinate acceptance survey".into(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn reach<R>(
    client: R::Client,
    profile: CrossSectionDefinition,
    lengths: &[f64],
    roughness: &[f64],
    areas: &[f64],
    discharges: &[f64],
    config: ChannelConfig,
    numerical: ChannelSurveyedConfig,
) -> TestResult<ChannelReach<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let count = lengths.len();
    if [roughness.len(), areas.len(), discharges.len()]
        .iter()
        .any(|&length| length != count)
    {
        return Err("Surveyed fixture lengths differ".into());
    }
    let definition = ChannelReachDefinition {
        vertical_datum: profile.vertical_datum.clone(),
        provenance: "Independent surveyed-reach acceptance".into(),
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile,
            cells: lengths
                .iter()
                .zip(roughness)
                .enumerate()
                .map(|(index, (&length_m, &manning_n))| SurveyedChannelCell {
                    id: format!("cell-{index}"),
                    length_m,
                    manning_n,
                })
                .collect(),
            numerical,
        },
        lateral_bindings: Vec::new(),
        config,
    };
    Ok(ChannelReach::new(
        client,
        definition,
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: areas.iter().zip(lengths).map(|(a, dx)| a * dx).collect(),
            longitudinal_momentum_m4_per_s: discharges
                .iter()
                .zip(lengths)
                .map(|(q, dx)| q * dx)
                .collect(),
        },
    )?)
}

pub fn forcing(
    start_s: f64,
    end_s: f64,
    boundaries: [ChannelBoundary; 2],
) -> ChannelForcingInterval {
    ChannelForcingInterval {
        start_s,
        end_s,
        left: boundaries[0].clone(),
        right: boundaries[1].clone(),
        lateral_inputs: Vec::new(),
    }
}

pub fn walls() -> [ChannelBoundary; 2] {
    [ChannelBoundary::Wall, ChannelBoundary::Wall]
}

pub fn advance<R>(
    mut current: ChannelReach<R>,
    target: f64,
    boundaries: [ChannelBoundary; 2],
) -> TestResult<ChannelReach<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let mut steps = 0;
    let started = std::time::Instant::now();
    while current.time() < target {
        let start = current.time();
        current = current
            .step(target, &forcing(start, target, boundaries.clone()))?
            .next;
        assert!(current.time() > start && current.time() <= target);
        check_water(&current)?;
        steps += 1;
        if steps <= 3 || steps % 25 == 0 {
            println!(
                "surveyed advancement: steps={steps}, time_s={}, elapsed_s={:.3}",
                current.time(),
                started.elapsed().as_secs_f64()
            );
        }
        assert!(
            steps < 1_000_000,
            "Surveyed advancement exhausted its step ceiling"
        );
    }
    assert_eq!(current.time().to_bits(), target.to_bits());
    Ok(current)
}

pub fn check_water<R>(reach: &ChannelReach<R>) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let ledger = reach.ledger();
    let scale = ledger
        .initial_storage_m3
        .max(ledger.storage_m3)
        .max(ledger.boundary_in_m3)
        .max(ledger.boundary_out_m3)
        .max(ledger.lateral_in_m3)
        .max(ledger.lateral_out_m3)
        .max(1.0);
    let tolerance = reach.config().volume_absolute + reach.config().volume_relative * scale;
    for charge in [
        ledger.numerical_transfer_underflow_bound_m3,
        ledger.lateral_counter_abs_bound_m3,
        ledger.lateral_assembly_abs_bound_m3,
        ledger.storage_selection_abs_bound_m3,
    ] {
        assert!(charge.is_finite() && charge >= 0.0);
    }
    assert!(
        ledger.residual_m3.abs()
            + ledger.numerical_transfer_underflow_bound_m3
            + ledger.lateral_counter_abs_bound_m3
            + ledger.lateral_assembly_abs_bound_m3
            + ledger.storage_selection_abs_bound_m3
            <= tolerance
    );
    let state = reach.state()?;
    for cell in &state {
        assert!(cell.volume_m3.is_finite() && cell.volume_m3 >= 0.0);
        assert!(cell.longitudinal_momentum_m4_per_s.is_finite());
        if cell.dry {
            assert_eq!(cell.volume_m3, 0.0);
            assert_eq!(cell.longitudinal_momentum_m4_per_s, 0.0);
        }
    }
    if let Some(step) = reach.last_step()? {
        for impulses in [
            &step.left_pressure_impulse_m4_per_s,
            &step.right_pressure_impulse_m4_per_s,
        ] {
            assert_eq!(impulses.len(), state.len() + 1);
            assert!(impulses.iter().all(|&impulse| impulse == 0.0));
        }
        let dt = step.end_s - step.start_s;
        assert!(dt > 0.0 && dt <= step.cfl_limit_s);
        assert!(dt <= step.accepted_start_cfl_limit_s);
        assert!(dt <= step.depletion_limit_s);
        if let Some(candidate) = step.candidate_cfl_limit_s {
            assert!(dt <= candidate);
        }
        let diagnostics = step
            .riemann
            .as_ref()
            .ok_or("Missing surveyed Riemann receipt")?;
        assert!(diagnostics.surveyed_work.is_some());
        for (widths, transfers, absolute) in [
            (
                &diagnostics.water_flux_width_m3_per_s,
                &step.face_volume_m3,
                reach.config().riemann_water_flux_absolute_m3_per_s,
            ),
            (
                &diagnostics.momentum_flux_width_m4_per_s2,
                &step.face_momentum_m4_per_s,
                reach.config().riemann_momentum_flux_absolute_m4_per_s2,
            ),
        ] {
            assert_eq!(widths.len(), transfers.len());
            for (&width, &transfer) in widths.iter().zip(transfers) {
                assert!(width.is_finite() && width >= 0.0);
                let allowance = absolute
                    + reach.config().riemann_flux_relative * ((transfer / dt).abs() + width);
                assert!(
                    width <= allowance,
                    "Surveyed flux width={width:e}, allowance={allowance:e}"
                );
            }
        }
    }
    Ok(())
}

pub fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}
