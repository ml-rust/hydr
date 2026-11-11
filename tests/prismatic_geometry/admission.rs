use super::fixtures::{
    ADMISSION_REJECTIONS, FORWARD_CASES, PROFILES, QUERY_REJECTIONS, Quantity, UNRESOLVED_INVERSE,
};
use super::support::{self, Client, Result};
use hydr::channel::{CrossSectionBatch, CrossSectionError, PrismaticRecoveryConfig};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn input(error: CrossSectionError, section: Option<usize>, label: &str) {
    assert!(
        matches!(error, CrossSectionError::InvalidInput { .. }),
        "{label}: {error}"
    );
    assert_eq!(error.section(), section, "{label}: {error}");
}

fn rejected<T>(
    result: std::result::Result<T, CrossSectionError>,
    label: &str,
) -> Result<CrossSectionError> {
    result
        .err()
        .ok_or_else(|| format!("Accepted {label}").into())
}

fn queries<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let definitions = batch.definitions().to_vec();
    let baseline_owners = FORWARD_CASES
        .iter()
        .map(|case| i64::try_from(case.profile))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let baseline_depths: Vec<_> = FORWARD_CASES
        .iter()
        .map(|case| f64::from_bits(case.depth_bits))
        .collect();
    let (owners, depths) = support::inputs::<R>(client, &baseline_owners, &baseline_depths, false)?;
    let retained = batch.evaluate_depths_tensor(&owners, &depths)?;
    let expected = retained.properties()?;
    for case in QUERY_REJECTIONS {
        let owner = i64::try_from(case.profile)?;
        let (owners, values) =
            support::inputs::<R>(client, &[owner], &[f64::from_bits(case.bits)], false)?;
        let error = match case.quantity {
            Quantity::Depth => rejected(
                batch.evaluate_depths_tensor(&owners, &values),
                "reference depth",
            )?,
            Quantity::Area => rejected(
                batch.recover_depths_tensor(&owners, &values, &PrismaticRecoveryConfig::default()),
                "reference area",
            )?,
        };
        input(error, Some(case.profile), "frozen query rejection");
    }
    for owner in [-1, i64::try_from(PROFILES.len())?] {
        let (owners, values) = support::inputs::<R>(client, &[owner], &[1.0], false)?;
        input(
            rejected(
                batch.evaluate_depths_tensor(&owners, &values),
                "out-of-range owner",
            )?,
            None,
            "owner range",
        );
        input(
            rejected(
                batch.recover_depths_tensor(&owners, &values, &PrismaticRecoveryConfig::default()),
                "out-of-range inverse owner",
            )?,
            None,
            "inverse owner range",
        );
    }
    assert_eq!(batch.definitions(), definitions.as_slice());
    assert_eq!(retained.properties()?, expected);
    assert_eq!(
        batch
            .evaluate_depths_tensor(&owners, &depths)?
            .properties()?,
        expected
    );
    Ok(())
}

fn metadata_and_empty<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let owner = Tensor::<R>::from_slice(&[0i64], &[1], client.device())?;
    let value = Tensor::<R>::from_slice(&[1.0f64], &[1], client.device())?;
    let owner_scalar = Tensor::<R>::from_slice(&[0i64], &[], client.device())?;
    let value_matrix = Tensor::<R>::from_slice(&[1.0f64], &[1, 1], client.device())?;
    let owners_two = Tensor::<R>::from_slice(&[0i64, 0], &[2], client.device())?;
    let owner_float = Tensor::<R>::from_slice(&[0.0f64], &[1], client.device())?;
    let value_f32 = Tensor::<R>::from_slice(&[1.0f32], &[1], client.device())?;
    let empty_owner = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
    let empty_value = Tensor::<R>::empty(&[0], DType::F64, client.device())?;
    let empty_owner_float = Tensor::<R>::empty(&[0], DType::F64, client.device())?;
    let empty_value_f32 = Tensor::<R>::empty(&[0], DType::F32, client.device())?;
    for (label, owners, values) in [
        ("scalar owners", &owner_scalar, &value),
        ("matrix values", &owner, &value_matrix),
        ("unequal row counts", &owners_two, &value),
        ("floating owners", &owner_float, &value),
        ("F32 values", &owner, &value_f32),
        ("empty floating owners", &empty_owner_float, &empty_value),
        ("empty F32 values", &empty_owner, &empty_value_f32),
    ] {
        input(
            rejected(batch.evaluate_depths_tensor(owners, values), label)?,
            None,
            label,
        );
        input(
            rejected(
                batch.recover_depths_tensor(owners, values, &PrismaticRecoveryConfig::default()),
                label,
            )?,
            None,
            label,
        );
    }
    let query = batch.evaluate_depths_tensor(&empty_owner, &empty_value)?;
    assert!(query.properties()?.is_empty());
    assert_eq!(query.owners().shape(), &[0]);
    assert_eq!(query.requested_depths_m().shape(), &[0]);
    assert_eq!(query.dry_mask().shape(), &[0]);
    for field in [
        query.depth_m(),
        query.area_m2(),
        query.top_width_m(),
        query.hydrostatic_integral_m3(),
        query.hydraulic_depth_m(),
    ] {
        support::native_bounds(field, 0);
    }
    let recovered = batch.recover_depths_tensor(
        &empty_owner,
        &empty_value,
        &PrismaticRecoveryConfig::default(),
    )?;
    assert!(recovered.properties()?.is_empty());
    assert_eq!(recovered.owners().shape(), &[0]);
    assert_eq!(recovered.requested_areas_m2().shape(), &[0]);
    assert_eq!(recovered.dry_mask().shape(), &[0]);
    for field in [
        recovered.depth_m(),
        recovered.forward_area_m2(),
        recovered.top_width_m(),
        recovered.hydrostatic_integral_m3(),
        recovered.hydraulic_depth_m(),
    ] {
        support::native_bounds(field, 0);
    }
    Ok(())
}

fn recovery_policy<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let (owners, areas) = support::inputs::<R>(client, &[0], &[2.0], false)?;
    for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for config in [
            PrismaticRecoveryConfig {
                depth_atol_m: value,
                depth_rtol: 0.0,
            },
            PrismaticRecoveryConfig {
                depth_atol_m: 0.0,
                depth_rtol: value,
            },
        ] {
            input(
                rejected(
                    batch.recover_depths_tensor(&owners, &areas, &config),
                    "invalid width allowance",
                )?,
                None,
                "width allowance",
            );
        }
    }
    let exact = PrismaticRecoveryConfig {
        depth_atol_m: 0.0,
        depth_rtol: 0.0,
    };
    let (owners, areas) = support::inputs::<R>(client, &[0, 0, 3], &[2.0, 6.0, 14.0], false)?;
    let rows = batch
        .recover_depths_tensor(&owners, &areas, &exact)?
        .properties()?;
    assert_eq!(rows.len(), 3);
    for (row, depth) in rows.iter().zip([1.0, 3.0, 3.0]) {
        assert_eq!(row.depth_m, [depth, depth]);
    }
    let tiny = CrossSectionBatch::<R>::new(
        client.clone(),
        vec![support::definition(UNRESOLVED_INVERSE.profile)],
    )?
    .certify_prismatic()?;
    let (owners, areas) = support::inputs::<R>(
        client,
        &[0],
        &[f64::from_bits(UNRESOLVED_INVERSE.area_bits)],
        false,
    )?;
    let error = rejected(
        tiny.recover_depths_tensor(
            &owners,
            &areas,
            &PrismaticRecoveryConfig {
                depth_atol_m: 1.0,
                depth_rtol: 1.0,
            },
        ),
        "positive unrepresented depth",
    )?;
    assert!(
        matches!(error, CrossSectionError::Numerical { .. }),
        "{error}"
    );
    assert_eq!(error.section(), Some(0));
    Ok(())
}

fn geometry<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    for case in ADMISSION_REJECTIONS {
        let error = match CrossSectionBatch::<R>::new(
            client.clone(),
            vec![support::definition(case.profile)],
        ) {
            Ok(source) => {
                let definitions = source.definitions().to_vec();
                let error = rejected(source.certify_prismatic(), case.reason)?;
                assert_eq!(source.definitions(), definitions.as_slice());
                error
            }
            Err(error) => error,
        };
        input(error, Some(0), case.reason);
    }
    Ok(())
}

fn unresolved_membership_and_width<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let loose = PrismaticRecoveryConfig {
        depth_atol_m: 1.0,
        depth_rtol: 1.0,
    };
    let exact = PrismaticRecoveryConfig {
        depth_atol_m: 0.0,
        depth_rtol: 0.0,
    };
    for (id, quantity, operand, config, label) in [
        (
            "represented_decimal_rectangle",
            Quantity::Area,
            2.7,
            loose,
            "unresolved cap area despite loose width allowances",
        ),
        (
            "represented_decimal_elevation",
            Quantity::Depth,
            2.9,
            loose,
            "unresolved cap depth",
        ),
        (
            "represented_decimal_rectangle",
            Quantity::Area,
            1.0,
            exact,
            "nonzero inverse width under exact-width-only policy",
        ),
    ] {
        let section = PROFILES
            .iter()
            .position(|profile| profile.id == id)
            .ok_or_else(|| format!("Missing conditioning profile {id}"))?;
        let owner = i64::try_from(section)?;
        let (owners, values) = support::inputs::<R>(client, &[owner], &[operand], false)?;
        let error = match quantity {
            Quantity::Depth => rejected(batch.evaluate_depths_tensor(&owners, &values), label)?,
            Quantity::Area => rejected(
                batch.recover_depths_tensor(&owners, &values, &config),
                label,
            )?,
        };
        assert!(
            matches!(error, CrossSectionError::Numerical { .. }),
            "{label}: {error}"
        );
        assert_eq!(error.section(), Some(section), "{label}: {error}");
    }
    Ok(())
}

pub(super) fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    queries::<R>(&client)?;
    metadata_and_empty::<R>(&client)?;
    recovery_policy::<R>(&client)?;
    unresolved_membership_and_width::<R>(&client)?;
    geometry::<R>(&client)
}

#[test]
fn cpu_prismatic_admission_preserves_supported_geometry_and_positive_water() -> Result {
    run::<crate::runtime::Runtime>(crate::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_prismatic_admission_preserves_supported_geometry_and_positive_water() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_rejects_foreign_device_empty_geometry_operands() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    let batch = support::batch::<CudaRuntime>(&client)?;
    let local_owner = Tensor::<CudaRuntime>::empty(&[0], DType::I64, client.device())?;
    let local_value = Tensor::<CudaRuntime>::empty(&[0], DType::F64, client.device())?;
    let foreign = CudaDevice::new(usize::MAX);
    let foreign_owner = Tensor::<CudaRuntime>::empty(&[0], DType::I64, &foreign)?;
    let foreign_value = Tensor::<CudaRuntime>::empty(&[0], DType::F64, &foreign)?;
    for (owners, values) in [
        (&foreign_owner, &local_value),
        (&local_owner, &foreign_value),
    ] {
        input(
            rejected(
                batch.evaluate_depths_tensor(owners, values),
                "foreign empty operands",
            )?,
            None,
            "foreign device",
        );
        input(
            rejected(
                batch.recover_depths_tensor(owners, values, &PrismaticRecoveryConfig::default()),
                "foreign empty inverse operands",
            )?,
            None,
            "foreign inverse device",
        );
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_rejects_f64_profile_without_cpu_substitution() -> Result {
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    assert!(!WgpuRuntime::computes_float_dtype(DType::F64));
    let client = WgpuClient::new(WgpuDevice::new(0))?;
    let profile = PROFILES.first().ok_or("Missing reference profile")?;
    assert!(
        CrossSectionBatch::<WgpuRuntime>::new(client, vec![support::definition(*profile)]).is_err()
    );
    Ok(())
}
