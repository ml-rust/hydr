use super::data::CASE_GROUPS;
use super::support::{self, Client, Result};
use hydr::channel::{
    ChannelError, CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint,
    PRISMATIC_BAND_WORK_LIMIT, PrismaticRiemannConfig, PrismaticRiemannError,
    PrismaticRiemannRejection, PrismaticSectionBatch,
};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn invalid<T>(result: std::result::Result<T, PrismaticRiemannError>, label: &str) -> Result {
    match result {
        Err(PrismaticRiemannError::Channel(ChannelError::InvalidInput { cell, context })) => {
            assert_eq!(cell, None, "{label}");
            assert!(!context.is_empty(), "{label}");
        }
        Err(error) => return Err(format!("{label}: wrong typed error: {error}").into()),
        Ok(_) => return Err(format!("{label}: accepted malformed input").into()),
    }
    Ok(())
}

fn call<R>(
    batch: &PrismaticSectionBatch<R>,
    input: &[Tensor<R>; 5],
    config: &PrismaticRiemannConfig,
) -> std::result::Result<hydr::channel::PrismaticRiemannResult<R>, PrismaticRiemannError>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    batch.riemann_flux_tensor(
        &input[0], &input[1], &input[2], &input[3], &input[4], config,
    )
}

fn metadata<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)
        .map_err(|error| format!("create metadata geometry: {error}"))?;
    let case = &CASE_GROUPS[0][0];
    let config = support::config(case);
    let normal = support::inputs::<R>(client, &[case], false)
        .map_err(|error| format!("create metadata operands: {error}"))?;
    let scalar_owner = Tensor::<R>::from_slice(&[0i64], &[], client.device())?;
    let float_owner = Tensor::<R>::from_slice(&[0f64], &[1], client.device())?;
    for owners in [&scalar_owner, &float_owner] {
        invalid(
            batch.riemann_flux_tensor(
                owners, &normal[1], &normal[2], &normal[3], &normal[4], &config,
            ),
            "owner metadata",
        )?;
    }
    for field in 1..5 {
        for replacement in [
            Tensor::<R>::from_slice(&[1.0f32], &[1], client.device())?,
            Tensor::<R>::from_slice(&[1.0f64], &[1, 1], client.device())?,
            Tensor::<R>::from_slice(&[1.0f64, 1.0], &[2], client.device())?,
        ] {
            let mut input = normal.clone();
            input[field] = replacement;
            invalid(call(&batch, &input, &config), "state metadata")?;
        }
    }
    for owner in [-1i64, 5] {
        let mut input = normal.clone();
        input[0] = Tensor::from_slice(&[owner], &[1], client.device())?;
        invalid(call(&batch, &input, &config), "section owner range")?;
    }
    for field in 1..5 {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            if value == -1.0 && field % 2 == 0 {
                continue;
            }
            let mut input = normal.clone();
            input[field] = Tensor::from_slice(&[value], &[1], client.device())?;
            invalid(call(&batch, &input, &config), "nonphysical state")?;
        }
    }
    for (area, discharge) in [(1usize, 2usize), (3, 4)] {
        for q in [f64::from_bits(1), -f64::from_bits(1), 1.0] {
            let mut input = normal.clone();
            input[area] = Tensor::from_slice(&[0.0f64], &[1], client.device())?;
            input[discharge] = Tensor::from_slice(&[q], &[1], client.device())?;
            invalid(call(&batch, &input, &config), "exact dry discharge")?;
        }
    }
    for count in [
        PRISMATIC_BAND_WORK_LIMIT / batch.maximum_band_count() + 1,
        usize::MAX / 8 + 1,
        usize::MAX / batch.maximum_band_count() + 1,
    ] {
        let huge = normal.clone().map(|tensor| {
            tensor
                .broadcast_to(&[count])
                .map_err(|error| format!("broadcast admission operands to count={count}: {error}"))
        });
        let [owners, la, lq, ra, rq] = huge;
        let input = [owners?, la?, lq?, ra?, rq?];
        invalid(
            call(&batch, &input, &config),
            "padded count and byte admission",
        )?;
    }
    let empty = support::inputs::<R>(client, &[], false)
        .map_err(|error| format!("create empty operands: {error}"))?;
    let result =
        call(&batch, &empty, &config).map_err(|error| format!("empty Riemann call: {error}"))?;
    super::accuracy::check(&result, &[], &config)
        .map_err(|error| format!("check empty Riemann output: {error}"))?;
    assert_eq!(
        result.work(),
        hydr::channel::PrismaticRiemannWork::default()
    );
    for field in 0..5 {
        let mut malformed = empty.clone();
        malformed[field] = Tensor::empty(
            &[0],
            if field == 0 { DType::F64 } else { DType::F32 },
            client.device(),
        )?;
        invalid(call(&batch, &malformed, &config), "empty operand dtype")?;
    }
    Ok(())
}

fn policies<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let case = &CASE_GROUPS[0][0];
    let input = support::inputs::<R>(client, &[case], false)?;
    let empty = support::inputs::<R>(client, &[], false)?;
    for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for field in 0..8 {
            let mut config = support::config(case);
            match field {
                0 => config.gravity_m_per_s2 = value,
                1 => config.recovery.depth_atol_m = value,
                2 => config.recovery.depth_rtol = value,
                3 => config.water_flux_atol_m3_per_s = value,
                4 => config.momentum_flux_atol_m4_per_s2 = value,
                5 => config.flux_rtol = value,
                6 => config.root.atol = value,
                _ => config.root.rtol = value,
            }
            invalid(call(&batch, &input, &config), "invalid numerical config")?;
            invalid(call(&batch, &empty, &config), "empty config admission")?;
        }
    }
    let mut config = support::config(case);
    config.gravity_m_per_s2 = 0.0;
    invalid(call(&batch, &input, &config), "zero gravity")?;
    for (initial, maximum) in [(0, 4), (3, 4), (4, 3), (8, 4)] {
        let mut config = support::config(case);
        config.initial_simpson_panels = initial;
        config.max_simpson_panels = maximum;
        invalid(call(&batch, &input, &config), "panel config")?;
    }
    Ok(())
}

fn uncertain_cap<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definition = CrossSectionDefinition {
        id: "represented-decimal-cap".into(),
        vertical_datum: "Original represented datum".into(),
        points: vec![
            CrossSectionPoint {
                station_m: 0.1,
                elevation_m: 3.0,
            },
            CrossSectionPoint {
                station_m: 0.1,
                elevation_m: 0.0,
            },
            CrossSectionPoint {
                station_m: 1.0,
                elevation_m: 0.0,
            },
            CrossSectionPoint {
                station_m: 1.0,
                elevation_m: 3.0,
            },
        ],
        maximum_stage_m: 3.0,
        profile_provenance: "Exact represented decimal rectangle".into(),
    };
    let batch =
        CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?;
    let owner = support::values::<R, _>(client, &[0i64], false)?;
    let area = support::values::<R, _>(client, &[2.7f64], false)?;
    let q = support::values::<R, _>(client, &[0.0f64], false)?;
    match batch.riemann_flux_tensor(
        &owner,
        &area,
        &q,
        &area,
        &q,
        &PrismaticRiemannConfig::default(),
    ) {
        Err(PrismaticRiemannError::Rejected {
            reason,
            query,
            section_owner,
            ..
        }) => {
            assert_eq!(reason, PrismaticRiemannRejection::CapUnresolved);
            assert_eq!(query, Some(0));
            assert_eq!(section_owner, Some(0));
        }
        Err(error) => return Err(format!("Uncertain cap returned {error}").into()),
        Ok(_) => return Err("Accepted uncertain original-area cap membership".into()),
    }
    Ok(())
}

fn original_rejection_identity<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let all: Vec<_> = CASE_GROUPS.iter().flat_map(|group| group.iter()).collect();
    let cap = all
        .iter()
        .copied()
        .find(|case| case.id == "three_bands_cap_compression")
        .ok_or("Missing frozen cap reference")?;
    let ordinary = all
        .iter()
        .copied()
        .find(|case| case.id == "rectangle_uniform_rest")
        .ok_or("Missing frozen rectangular reference")?;
    let dry = all
        .iter()
        .copied()
        .find(|case| {
            case.dry
                && case.rejection.is_none()
                && case.allowed.is_empty()
                && case.gravity == cap.gravity
        })
        .ok_or("Missing frozen dry reference")?;
    let mut cases: Vec<_> = (0..33)
        .map(|row| if row % 2 == 0 { ordinary } else { dry })
        .collect();
    cases[17] = cap;
    let batch = support::batch::<R>(client)?;
    let input = support::inputs::<R>(client, &cases, true)?;
    match call(&batch, &input, &support::config(cap)) {
        Err(PrismaticRiemannError::Rejected {
            reason,
            query,
            section_owner,
            ..
        }) => {
            assert_eq!(reason, PrismaticRiemannRejection::CapExceeded);
            assert_eq!(query, Some(17));
            assert_eq!(section_owner, Some(cap.profile));
        }
        Err(error) => return Err(format!("Original-row cap identity returned {error}").into()),
        Ok(_) => return Err("Accepted a frozen cap compression in a mixed batch".into()),
    }
    Ok(())
}

fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    metadata::<R>(&client).map_err(|error| format!("admission metadata: {error}"))?;
    policies::<R>(&client).map_err(|error| format!("admission policies: {error}"))?;
    uncertain_cap::<R>(&client).map_err(|error| format!("admission uncertain_cap: {error}"))?;
    original_rejection_identity::<R>(&client)
        .map_err(|error| format!("admission original_rejection_identity: {error}").into())
}

#[test]
fn configured_cpu_checks_original_input_and_allocation_admission() -> Result {
    run::<numr::runtime::cpu::CpuRuntime>(support::cpu())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_checks_original_input_and_allocation_admission() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_checks_each_empty_operand_device() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    let batch = support::batch::<CudaRuntime>(&client)?;
    let local = support::inputs::<CudaRuntime>(&client, &[], false)?;
    let foreign = CudaDevice::new(usize::MAX);
    for field in 0..5 {
        let mut input = local.clone();
        input[field] = Tensor::empty(
            &[0],
            if field == 0 { DType::I64 } else { DType::F64 },
            &foreign,
        )?;
        invalid(
            call(&batch, &input, &PrismaticRiemannConfig::default()),
            "foreign empty device",
        )?;
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_rejects_f64_surveyed_provider() -> Result {
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    assert!(!WgpuRuntime::computes_float_dtype(DType::F64));
    let client = WgpuClient::new(WgpuDevice::new(0))?;
    assert!(support::batch::<WgpuRuntime>(&client).is_err());
    Ok(())
}

#[test]
fn frozen_default_bits_and_work_ceilings_remain_unchanged() {
    let c = PrismaticRiemannConfig::default();
    for (actual, expected) in [
        (c.gravity_m_per_s2, 9.81f64),
        (c.recovery.depth_atol_m, 1e-12),
        (c.recovery.depth_rtol, 1e-12),
        (c.water_flux_atol_m3_per_s, 1e-9),
        (c.momentum_flux_atol_m4_per_s2, 1e-8),
        (c.flux_rtol, 1e-10),
        (c.root.atol, 1e-12),
        (c.root.rtol, 1e-12),
    ] {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(c.root.max_iterations, 256);
    assert_eq!(c.root.max_points, 1 << 20);
    assert_eq!(c.initial_simpson_panels, 4);
    assert_eq!(c.max_simpson_panels, 4096);
    assert_eq!(c.max_refinements, 10);
    assert_eq!(c.max_native_query_points, 1 << 24);
    assert_eq!(PRISMATIC_BAND_WORK_LIMIT, 1 << 20);
}
