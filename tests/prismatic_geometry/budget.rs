use super::fixtures::{PROFILES, QUERY_BAND_LIMIT};
use super::support::{self, Client, Result};
use hydr::channel::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionError, CrossSectionPoint,
    PRISMATIC_BAND_WORK_LIMIT, PrismaticRecoveryConfig,
};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn query_work<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(PRISMATIC_BAND_WORK_LIMIT, QUERY_BAND_LIMIT);
    let batch = support::batch::<R>(client)?;
    assert!(batch.maximum_band_count() > 0);
    let rows = QUERY_BAND_LIMIT / batch.maximum_band_count() + 1;
    let owners = Tensor::<R>::from_slice(&[0i64], &[1], client.device())?.broadcast_to(&[rows])?;
    let values =
        Tensor::<R>::from_slice(&[1.0f64], &[1], client.device())?.broadcast_to(&[rows])?;
    assert_eq!(owners.storage().len(), 1);
    assert_eq!(values.storage().len(), 1);
    let error = batch
        .evaluate_depths_tensor(&owners, &values)
        .err()
        .ok_or("Accepted excessive padded query work")?;
    assert!(
        matches!(error, CrossSectionError::InvalidInput { .. }),
        "{error}"
    );
    let error = batch
        .recover_depths_tensor(&owners, &values, &PrismaticRecoveryConfig::default())
        .err()
        .ok_or("Accepted excessive padded inverse work")?;
    assert!(
        matches!(error, CrossSectionError::InvalidInput { .. }),
        "{error}"
    );
    let (owners, values) = support::inputs::<R>(client, &[0], &[1.0], false)?;
    assert_eq!(
        batch
            .evaluate_depths_tensor(&owners, &values)?
            .properties()?
            .len(),
        1
    );
    Ok(())
}

fn compilation_work<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let bands = 2000usize;
    let section_count = QUERY_BAND_LIMIT / bands + 1;
    assert!(section_count * bands > QUERY_BAND_LIMIT);
    let ordinary = PROFILES.first().ok_or("Missing rectangle reference")?;
    let mut definitions: Vec<_> = (0..section_count - 1)
        .map(|index| {
            let mut definition = support::definition(*ordinary);
            definition.id = format!("ordinary-section-{index}");
            definition
        })
        .collect();
    let mut points = vec![
        CrossSectionPoint {
            station_m: -(bands as f64),
            elevation_m: bands as f64,
        },
        CrossSectionPoint {
            station_m: 0.0,
            elevation_m: 0.0,
        },
    ];
    points.extend((1..=bands).map(|height| CrossSectionPoint {
        station_m: height as f64,
        elevation_m: height as f64,
    }));
    definitions.push(CrossSectionDefinition {
        id: "many-band-section".into(),
        vertical_datum: "Analytical geometry datum".into(),
        points,
        maximum_stage_m: bands as f64,
        profile_provenance: "Continuous triangular banks with explicit elevation knots".into(),
    });
    let source = CrossSectionBatch::<R>::new(client.clone(), definitions)?;
    let before = source.definitions().to_vec();
    let error = source
        .certify_prismatic()
        .err()
        .ok_or("Accepted excessive padded compilation work")?;
    assert!(
        matches!(error, CrossSectionError::InvalidInput { .. }),
        "{error}"
    );
    assert_eq!(source.definitions(), before.as_slice());
    Ok(())
}

pub(super) fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    query_work::<R>(&client)?;
    compilation_work::<R>(&client)
}

#[test]
fn cpu_prismatic_work_limits_bound_padded_native_allocations() -> Result {
    run::<crate::runtime::Runtime>(crate::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_prismatic_work_limits_bound_padded_native_allocations() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
