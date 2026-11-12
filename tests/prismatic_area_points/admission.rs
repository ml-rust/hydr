use super::fixtures::{CASES, PROFILES};
use super::support::{self, Client, Result};
use hydr::channel::{CrossSectionError, PRISMATIC_BAND_WORK_LIMIT, PrismaticRecoveryConfig};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn rejected<T>(
    result: std::result::Result<T, CrossSectionError>,
    numerical: bool,
    section: Option<usize>,
    label: &str,
) -> Result {
    let error = result.err().ok_or_else(|| format!("Accepted {label}"))?;
    assert_eq!(
        matches!(error, CrossSectionError::Numerical { .. }),
        numerical,
        "{label}: {error}"
    );
    if !numerical {
        assert!(
            matches!(error, CrossSectionError::InvalidInput { .. }),
            "{label}: {error}"
        );
    }
    assert_eq!(error.section(), section, "{label}: {error}");
    Ok(())
}

fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(&client)?;
    let config = PrismaticRecoveryConfig::default();
    let controls: Vec<_> = CASES.iter().filter(|case| !case.resolved).collect();
    assert_eq!(controls.len(), 2);
    for case in controls {
        assert!(!case.dry);
        for strided in [false, true] {
            let (owners, area) = support::inputs::<R>(
                &client,
                &[i64::try_from(case.owner)?],
                &[f64::from_bits(case.input_high)],
                &[f64::from_bits(case.input_low)],
                strided,
            )?;
            rejected(
                batch.recover_area_points_tensor(&owners, &area, &config),
                true,
                Some(case.owner),
                case.id,
            )?;
        }
    }
    for owner in [-1, i64::try_from(PROFILES.len())?] {
        let (owners, areas) = support::inputs::<R>(&client, &[owner], &[1.0], &[0.0], false)?;
        rejected(
            batch.recover_area_points_tensor(&owners, &areas, &config),
            false,
            None,
            "owner outside compiled sections",
        )?;
    }
    let (owner, negative) =
        support::inputs::<R>(&client, &[0], &[0.0], &[-f64::from_bits(1)], false)?;
    rejected(
        batch.recover_area_points_tensor(&owner, &negative, &config),
        false,
        Some(0),
        "negative area point",
    )?;
    let high = support::tensor::<R>(&client, &[1.0], false)?;
    let zero = support::tensor::<R>(&client, &[0.0], false)?;
    let radius = support::tensor::<R>(&client, &[f64::from_bits(1)], false)?;
    let uncertain = support::expansion::<R>(&client, &high, &zero, &radius)?;
    rejected(
        batch.recover_area_points_tensor(&owner, &uncertain, &config),
        false,
        Some(0),
        "nonzero point radius",
    )?;
    for cap in CASES.iter().filter(|case| case.id.ends_with("exact_cap")) {
        let deficit = CASES
            .iter()
            .find(|case| case.owner == cap.owner && case.id.ends_with("cap_retained_deficit"))
            .ok_or("Missing retained cap deficit")?;
        let excess = -f64::from_bits(deficit.input_low);
        assert!(excess > 0.0);
        let (cap_owner, cap_area) = support::inputs::<R>(
            &client,
            &[i64::try_from(cap.owner)?],
            &[f64::from_bits(cap.input_high)],
            &[excess],
            false,
        )?;
        assert_eq!(cap_area.high().item::<f64>()?.to_bits(), cap.input_high);
        assert!(cap_area.low().item::<f64>()? > 0.0);
        rejected(
            batch.recover_area_points_tensor(&cap_owner, &cap_area, &config),
            false,
            Some(cap.owner),
            "cap excess retained only in low component",
        )?;
    }
    let (_, area) = support::inputs::<R>(&client, &[0], &[1.0], &[0.0], false)?;
    let scalar_owner = Tensor::<R>::from_slice(&[0i64], &[], client.device())?;
    let floating_owner = Tensor::<R>::from_slice(&[0.0f64], &[1], client.device())?;
    let two_owners = Tensor::<R>::from_slice(&[0i64, 0], &[2], client.device())?;
    for owners in [&scalar_owner, &floating_owner, &two_owners] {
        rejected(
            batch.recover_area_points_tensor(owners, &area, &config),
            false,
            None,
            "owner metadata",
        )?;
    }
    let matrix = Tensor::<R>::from_slice(&[1.0f64], &[1, 1], client.device())?;
    let matrix_zero = Tensor::<R>::zeros(&[1, 1], DType::F64, client.device())?;
    let matrix_area = support::expansion::<R>(&client, &matrix, &matrix_zero, &matrix_zero)?;
    rejected(
        batch.recover_area_points_tensor(&owner, &matrix_area, &config),
        false,
        None,
        "area rank",
    )?;
    let f32_area = Tensor::<R>::from_slice(&[1.0f32], &[1], client.device())?;
    let f32_zero = Tensor::<R>::zeros(&[1], DType::F32, client.device())?;
    let f32_area = support::expansion::<R>(&client, &f32_area, &f32_zero, &f32_zero)?;
    rejected(
        batch.recover_area_points_tensor(&owner, &f32_area, &config),
        false,
        None,
        "area dtype",
    )?;
    let (empty_owner, empty_area) = support::inputs::<R>(&client, &[], &[], &[], false)?;
    let empty = batch.recover_area_points_tensor(&empty_owner, &empty_area, &config)?;
    assert!(empty.properties()?.is_empty());
    assert_eq!(empty.owners().shape(), &[0]);
    assert_eq!(empty.requested_areas_m2().high().shape(), &[0]);
    assert_eq!(empty.requested_areas_m2().low().shape(), &[0]);
    assert_eq!(empty.dry_mask().shape(), &[0]);
    assert_eq!(
        empty.band_coordinates().candidate_mask().shape(),
        &[0, batch.maximum_band_count()]
    );
    for bounds in [
        empty.depth_m(),
        empty.forward_area_m2(),
        empty.top_width_m(),
        empty.hydrostatic_integral_m3(),
        empty.hydraulic_depth_m(),
    ] {
        support::interval(bounds, &[0]);
    }
    for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for invalid_config in [
            PrismaticRecoveryConfig {
                depth_atol_m: invalid,
                depth_rtol: 0.0,
            },
            PrismaticRecoveryConfig {
                depth_atol_m: 0.0,
                depth_rtol: invalid,
            },
        ] {
            for (owners, areas) in [(&owner, &area), (&empty_owner, &empty_area)] {
                rejected(
                    batch.recover_area_points_tensor(owners, areas, &invalid_config),
                    false,
                    None,
                    "invalid depth allowance",
                )?;
            }
        }
    }
    let wet = CASES
        .iter()
        .find(|case| {
            case.resolved
                && !case.dry
                && case.geometry[0].is_some_and(|bounds| bounds[0] != bounds[1])
        })
        .ok_or("Missing positive coordinate reference")?;
    let (wet_owner, wet_area) = support::inputs::<R>(
        &client,
        &[i64::try_from(wet.owner)?],
        &[f64::from_bits(wet.input_high)],
        &[f64::from_bits(wet.input_low)],
        false,
    )?;
    let tight = PrismaticRecoveryConfig {
        depth_atol_m: 0.0,
        depth_rtol: 0.0,
    };
    rejected(
        batch.recover_area_points_tensor(&wet_owner, &wet_area, &tight),
        true,
        Some(wet.owner),
        "zero depth width allowance",
    )?;
    let rows = PRISMATIC_BAND_WORK_LIMIT / batch.maximum_band_count() + 1;
    let expanded_owner = owner.broadcast_to(&[rows])?;
    let broadcast = high.broadcast_to(&[rows])?;
    let broadcast_zero = zero.broadcast_to(&[rows])?;
    let expanded_area =
        support::expansion::<R>(&client, &broadcast, &broadcast_zero, &broadcast_zero)?;
    rejected(
        batch.recover_area_points_tensor(&expanded_owner, &expanded_area, &config),
        false,
        None,
        "padded band budget",
    )?;
    Ok(())
}

#[test]
fn cpu_area_points_admit_only_supported_exact_selected_points() -> Result {
    run::<super::runtime::Runtime>(super::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_area_points_admit_only_supported_exact_selected_points() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_area_points_reject_foreign_owner_device() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    let batch = support::batch::<CudaRuntime>(&client)?;
    let (_, area) = support::inputs::<CudaRuntime>(&client, &[], &[], &[], false)?;
    let foreign = CudaDevice::new(usize::MAX);
    let owners = Tensor::<CudaRuntime>::empty(&[0], DType::I64, &foreign)?;
    rejected(
        batch.recover_area_points_tensor(&owners, &area, &PrismaticRecoveryConfig::default()),
        false,
        None,
        "foreign owner device",
    )
}
