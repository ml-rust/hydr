use super::friction_test_data::{ODE, Ode, SELECTION};
use crate::channel::{
    ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReach, ChannelReachDefinition,
    ChannelSurveyedConfig, CrossSectionDefinition, CrossSectionPoint, SurveyedChannelCell,
};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use solvr::optimize::roots::BatchedBisectionAlgorithms;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn reach(row: &Ode, roughness: f64) -> TestResult<ChannelReach<CpuRuntime>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    reach_with_client::<CpuRuntime>(client, row, roughness)
}

fn reach_with_client<R>(client: R::Client, row: &Ode, roughness: f64) -> TestResult<ChannelReach<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>,
{
    let operands = row.operands.map(f64::from_bits);
    Ok(ChannelReach::new(
        client,
        ChannelReachDefinition {
            vertical_datum: "Synthetic SI datum".into(),
            provenance: "Independent original-coordinate fixed-area Manning references".into(),
            geometry: ChannelGeometry::PrismaticSurveyed {
                profile: CrossSectionDefinition {
                    id: row.id.into(),
                    vertical_datum: "Synthetic SI datum".into(),
                    points: row
                        .points
                        .iter()
                        .map(|point| CrossSectionPoint {
                            station_m: f64::from_bits(point[0]),
                            elevation_m: f64::from_bits(point[1]),
                        })
                        .collect(),
                    maximum_stage_m: f64::from_bits(row.cap),
                    profile_provenance: "Frozen original survey bits".into(),
                },
                cells: vec![SurveyedChannelCell {
                    id: "cell".into(),
                    length_m: operands[0],
                    manning_n: roughness,
                }],
                numerical: ChannelSurveyedConfig::default(),
            },
            lateral_bindings: Vec::new(),
            config: ChannelConfig {
                gravity: operands[4],
                ..ChannelConfig::default()
            },
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![operands[2]],
            longitudinal_momentum_m4_per_s: vec![operands[3]],
        },
    )?)
}

#[test]
fn original_fixed_area_manning_matches_fourteen_exact_power_ode_references() -> TestResult {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    check_ode_references::<CpuRuntime>(client)
}

fn check_ode_references<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    assert_eq!(ODE.len(), 14);
    for row in ODE {
        let operands = row.operands.map(f64::from_bits);
        assert_eq!((operands[2] / operands[0]).to_bits(), row.operands[1]);
        let reach = reach_with_client::<R>(client.clone(), row, operands[5])?;
        let before = reach.snapshot_data()?;
        let context = &reach.context;
        let trial = context.friction_trial(
            &reach.accepted.volume,
            &reach.accepted.momentum,
            operands[6],
        )?;
        let actual = trial.momentum.item::<f64>()?;
        let geometry = trial.geometry.ok_or("Missing Manning geometry charge")?;
        let charge = geometry.momentum_abs_bound.item::<f64>()?;
        assert!(
            charge.is_finite() && charge >= 0.0,
            "{}: charge={charge}",
            row.id
        );
        assert!(
            charge <= 1e-9 + 1e-12 * operands[3].abs(),
            "{}: charge={charge}",
            row.id
        );
        let reference = row.reference.map(f64::from_bits);
        if row.active {
            let error = (actual - reference[0])
                .abs()
                .max((actual - reference[1]).abs());
            let magnitude = reference[0].abs().min(reference[1].abs());
            let allowance = 1e-10 + 1e-11 * magnitude;
            assert!(
                error <= allowance,
                "{}: native={actual:e}, reference={reference:?}, error={error:e}, allowance={allowance:e}",
                row.id
            );
            assert_eq!(
                actual.is_sign_negative(),
                operands[3].is_sign_negative(),
                "{}",
                row.id
            );
            assert!(
                actual.abs() > 0.0 && actual.abs() <= operands[3].abs(),
                "{}",
                row.id
            );
        } else {
            assert_eq!(actual.to_bits(), row.operands[3], "{}", row.id);
            assert_eq!(charge.to_bits(), 0.0_f64.to_bits(), "{}", row.id);
        }
        assert_eq!(reach.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn six_frozen_radius_selections_check_the_production_geometry_gate() -> TestResult {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    check_radius_selections::<CpuRuntime>(client)
}

fn check_radius_selections<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>,
{
    assert_eq!(SELECTION.len(), 6);
    let ordinary = ODE.first().ok_or("Missing ordinary Manning fixture")?;
    let reach = reach_with_client::<R>(client, ordinary, f64::from_bits(ordinary.operands[5]))?;
    let context = &reach.context;
    let active = context.tensor(&[1.0])?;
    for row in SELECTION {
        let [lower, upper, selected, momentum] = row.radii_momentum.map(f64::from_bits);
        let result = context.surveyed_radius_selection_bound(
            &context.tensor(&[lower])?,
            &context.tensor(&[upper])?,
            &context.tensor(&[selected])?,
            &context.tensor(&[momentum.abs()])?,
            &active,
        );
        if row.expected == "accept" {
            let actual = result?.item::<f64>()?;
            let exact = row.bound.map(f64::from_bits);
            assert!(
                actual >= exact[1],
                "{}: bound={actual:e}, exact={exact:?}",
                row.id
            );
            assert!(actual <= 1e-9 + 1e-12 * momentum.abs(), "{}", row.id);
            if exact[1] == 0.0 {
                assert_eq!(actual, 0.0, "{}", row.id);
            }
        } else {
            let error = result.err().ok_or("Expected radius-selection rejection")?;
            let text = error.to_string();
            let expected = if row.expected == "reject_selection" {
                "enclosure"
            } else {
                "geometry allowance"
            };
            assert!(text.contains(expected), "{}: {text}", row.id);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn configured_cuda_manning_matches_fourteen_exact_power_ode_references() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    check_ode_references::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[cfg(feature = "cuda")]
#[test]
fn configured_cuda_checks_six_frozen_radius_selections() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    check_radius_selections::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[test]
fn inactive_radius_selection_masks_invalid_operands_before_arithmetic() -> TestResult {
    let ordinary = ODE.first().ok_or("Missing ordinary Manning fixture")?;
    let reach = reach(ordinary, 0.03)?;
    let context = &reach.context;
    let result = context.surveyed_radius_selection_bound(
        &context.tensor(&[0.0])?,
        &context.tensor(&[f64::INFINITY])?,
        &context.tensor(&[f64::NAN])?,
        &context.tensor(&[f64::MAX])?,
        &context.tensor(&[0.0])?,
    )?;
    assert_eq!(result.item::<f64>()?.to_bits(), 0.0_f64.to_bits());
    Ok(())
}

#[test]
fn active_manning_retains_duration_and_arithmetic_rejections() -> TestResult {
    let ordinary = ODE.first().ok_or("Missing ordinary Manning fixture")?;
    let normal = reach(ordinary, 0.03)?;
    for duration in [-1.0, f64::NAN, f64::INFINITY] {
        let result = normal.context.friction_trial(
            &normal.accepted.volume,
            &normal.accepted.momentum,
            duration,
        );
        let error = result
            .err()
            .ok_or("Invalid Manning duration requires rejection")?;
        assert!(error.to_string().contains("duration"));
    }
    for roughness in [f64::from_bits(1), f64::MAX] {
        let reach = reach(ordinary, roughness)?;
        let result =
            reach
                .context
                .friction_trial(&reach.accepted.volume, &reach.accepted.momentum, 0.1);
        let error = result
            .err()
            .ok_or("Unrepresented active Manning arithmetic requires rejection")?;
        assert!(error.to_string().contains("roughness"), "{error}");
    }
    let result = normal.context.friction_trial(
        &normal.accepted.volume,
        &normal.accepted.momentum,
        f64::from_bits(1),
    );
    let error = result
        .err()
        .ok_or("Unrepresented resistance requires rejection")?;
    assert!(error.to_string().contains("resistance"), "{error}");
    Ok(())
}

#[test]
fn normal_roughness_distinguishes_equal_area_bank_perimeters() -> TestResult {
    let mut outputs = Vec::new();
    for id in ["symmetric_equal_area", "asymmetric_equal_area"] {
        let row = ODE
            .iter()
            .find(|row| row.id == id)
            .ok_or("Missing equal-area Manning fixture")?;
        let reach = reach(row, 0.03)?;
        outputs.push(
            reach
                .context
                .friction_trial(&reach.accepted.volume, &reach.accepted.momentum, 0.1)?
                .momentum
                .item::<f64>()?,
        );
    }
    assert!((outputs[0] - outputs[1]).abs() > 1e-6);
    Ok(())
}
