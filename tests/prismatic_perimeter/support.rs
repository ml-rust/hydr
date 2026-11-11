use super::fixtures::PROFILES;
use super::model::Profile;
use hydr::channel::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticPerimeterBatch,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) trait Client<R: Runtime<DType = DType>>:
    TensorOps<R>
    + CompareOps<R>
    + NextRepresentableOps<R>
    + ScalarOps<R>
    + TwoSumOps<R>
    + HypotOps<R>
    + MidpointOps<R>
    + RuntimeClient<R>
{
}
impl<R, C> Client<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + RuntimeClient<R>,
{
}

pub(crate) fn definition(profile: Profile) -> CrossSectionDefinition {
    CrossSectionDefinition {
        id: profile.id.into(),
        vertical_datum: "Frozen original coordinate datum".into(),
        points: profile
            .points
            .iter()
            .map(|point| CrossSectionPoint {
                station_m: f64::from_bits(point[0]),
                elevation_m: f64::from_bits(point[1]),
            })
            .collect(),
        maximum_stage_m: f64::from_bits(profile.cap),
        profile_provenance: "Independent original-segment clipping".into(),
    }
}

pub(crate) fn batch<R>(client: &R::Client) -> Result<PrismaticPerimeterBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    Ok(CrossSectionBatch::<R>::new(
        client.clone(),
        PROFILES.iter().copied().map(definition).collect(),
    )?
    .certify_prismatic()?
    .certify_perimeter()?)
}

pub(crate) fn inputs<R>(
    client: &R::Client,
    owners: &[i64],
    values: &[f64],
    strided: bool,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(owners.len(), values.len());
    let n = owners.len();
    if strided && n > 0 {
        let owner_storage: Vec<_> = owners.iter().flat_map(|&v| [i64::MAX, v]).collect();
        let value_storage: Vec<_> = values.iter().flat_map(|&v| [f64::NAN, v]).collect();
        let owners = Tensor::from_slice(&owner_storage, &[n, 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        let values = Tensor::from_slice(&value_storage, &[n, 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(owners.offset() > 0 && values.offset() > 0);
        if n > 1 {
            assert!(!owners.is_contiguous() && !values.is_contiguous());
        }
        return Ok((owners, values));
    }
    Ok((
        Tensor::from_slice(owners, &[n], client.device())?,
        Tensor::from_slice(values, &[n], client.device())?,
    ))
}

pub(crate) fn enclosed(
    actual: [f64; 2],
    truth: [u64; 2],
    label: &str,
    absolute: f64,
    relative: f64,
) {
    assert!(
        actual[0].is_finite() && actual[1].is_finite(),
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] >= 0.0 && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] <= f64::from_bits(truth[0]) && actual[1] >= f64::from_bits(truth[1]),
        "{label}: {actual:?} excludes {truth:x?}"
    );
    let limit = absolute + relative * actual[0].abs().max(actual[1].abs());
    assert!(
        actual[1] - actual[0] <= limit,
        "{label}: {actual:?}, full width exceeds {limit}"
    );
}

pub(crate) fn native<R: Runtime<DType = DType>>(bounds: &Interval<R>, n: usize) {
    for endpoint in [bounds.lower(), bounds.upper()] {
        assert_eq!(endpoint.dtype(), DType::F64);
        assert_eq!(endpoint.shape(), &[n]);
    }
}

pub(crate) fn snapshots<R: Runtime<DType = DType>>(
    owners: &Tensor<R>,
    values: &Tensor<R>,
    dry: &Tensor<R>,
    expected: &[i64],
    bits: &[u64],
) -> Result {
    assert_eq!(owners.shape(), &[expected.len()]);
    assert_eq!(values.shape(), &[expected.len()]);
    assert_eq!(dry.shape(), &[expected.len()]);
    assert_eq!(owners.dtype(), DType::I64);
    assert_eq!(values.dtype(), DType::F64);
    assert_eq!(dry.dtype(), DType::F64);
    assert_eq!(owners.try_to_vec::<i64>()?, expected);
    assert_eq!(
        values
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        bits
    );
    assert_eq!(
        dry.try_to_vec::<f64>()?,
        bits.iter()
            .map(|&v| if f64::from_bits(v) == 0.0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>()
    );
    Ok(())
}

pub(crate) fn overwrite<R>(client: &R::Client, tensors: &[&Tensor<R>]) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    for tensor in tensors {
        client.copy_into(
            tensor,
            &Tensor::zeros(tensor.shape(), tensor.dtype(), client.device())?,
        )?;
    }
    Ok(())
}
