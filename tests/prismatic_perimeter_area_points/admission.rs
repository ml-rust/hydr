use super::fixtures::CASES;
use super::point_support;
use super::support::{Client, Result};
use hydr::channel::{CrossSectionError, PrismaticRecoveryConfig};
use numr::dtype::DType;
use numr::runtime::Runtime;

fn rejected<T>(
    result: std::result::Result<T, CrossSectionError>,
    numerical: bool,
    section: Option<usize>,
) -> Result {
    let error = result
        .err()
        .ok_or("Perimeter recovery accepts an unsupported area point")?;
    assert_eq!(
        matches!(error, CrossSectionError::Numerical { .. }),
        numerical,
        "{error}"
    );
    if !numerical {
        assert!(
            matches!(error, CrossSectionError::InvalidInput { .. }),
            "{error}"
        );
    }
    assert_eq!(error.section(), section, "{error}");
    Ok(())
}

fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = point_support::batch::<R>(&client)?.certify_perimeter()?;
    let config = PrismaticRecoveryConfig::default();
    let controls: Vec<_> = CASES.iter().filter(|case| !case.resolved).collect();
    assert_eq!(controls.len(), 2);
    for case in controls {
        assert!(!case.dry);
        let (owner, area) = point_support::inputs::<R>(
            &client,
            &[i64::try_from(case.owner)?],
            &[f64::from_bits(case.input_high)],
            &[f64::from_bits(case.input_low)],
            false,
        )?;
        rejected(
            batch.recover_area_points_tensor(&owner, &area, &config),
            true,
            Some(case.owner),
        )?;
    }
    let (owner, negative) =
        point_support::inputs::<R>(&client, &[0], &[0.0], &[-f64::from_bits(1)], false)?;
    rejected(
        batch.recover_area_points_tensor(&owner, &negative, &config),
        false,
        Some(0),
    )?;
    let high = point_support::tensor::<R>(&client, &[1.0], false)?;
    let low = point_support::tensor::<R>(&client, &[0.0], false)?;
    let radius = point_support::tensor::<R>(&client, &[f64::from_bits(1)], false)?;
    let uncertain = point_support::expansion::<R>(&client, &high, &low, &radius)?;
    rejected(
        batch.recover_area_points_tensor(&owner, &uncertain, &config),
        false,
        Some(0),
    )?;
    let (empty_owner, empty_area) = point_support::inputs::<R>(&client, &[], &[], &[], false)?;
    let empty = batch.recover_area_points_tensor(&empty_owner, &empty_area, &config)?;
    assert!(empty.properties()?.is_empty());
    assert_eq!(empty.geometry().owners().shape(), &[0]);
    assert_eq!(empty.geometry().requested_areas_m2().high().shape(), &[0]);
    assert_eq!(empty.geometry().requested_areas_m2().low().shape(), &[0]);
    for bounds in [empty.wetted_perimeter_m(), empty.hydraulic_radius_m()] {
        point_support::interval(bounds, &[0]);
    }
    let invalid = PrismaticRecoveryConfig {
        depth_atol_m: -1.0,
        depth_rtol: 0.0,
    };
    rejected(
        batch.recover_area_points_tensor(&empty_owner, &empty_area, &invalid),
        false,
        None,
    )?;
    Ok(())
}

#[test]
fn cpu_perimeter_area_points_delegate_selected_point_admission() -> Result {
    run::<super::runtime::Runtime>(super::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_perimeter_area_points_delegate_selected_point_admission() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
