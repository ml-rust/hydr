use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval, with_interval_arithmetic};

pub(crate) type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) fn batch<R>(client: &R::Client) -> Result<hydr::channel::PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    use hydr::channel::{CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint};
    let definitions = super::fixtures::PROFILES
        .iter()
        .map(|profile| CrossSectionDefinition {
            id: profile.id.into(),
            vertical_datum: "Frozen binary geometry datum".into(),
            points: profile
                .coordinates
                .iter()
                .map(|point| CrossSectionPoint {
                    station_m: f64::from_bits(point[0]),
                    elevation_m: f64::from_bits(point[1]),
                })
                .collect(),
            maximum_stage_m: f64::from_bits(profile.cap_stage),
            profile_provenance: "Frozen original-coordinate area integration".into(),
        })
        .collect();
    Ok(CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?)
}

pub(crate) fn enclosed(actual: [f64; 2], truth: [u64; 2], label: &str) {
    let expected = truth.map(f64::from_bits);
    assert!(
        actual[0].is_finite() && actual[1].is_finite(),
        "{label}: {actual:?}"
    );
    assert!(
        0.0 <= actual[0] && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{label}: {actual:?} excludes {expected:?}"
    );
}

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

pub(crate) fn tensor<R>(client: &R::Client, values: &[f64], strided: bool) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    if strided && !values.is_empty() {
        let storage: Vec<_> = values.iter().flat_map(|&value| [f64::NAN, value]).collect();
        let result = Tensor::from_slice(&storage, &[values.len(), 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(result.offset() > 0);
        if values.len() > 1 {
            assert!(!result.is_contiguous());
        }
        Ok(result)
    } else {
        Ok(Tensor::from_slice(
            values,
            &[values.len()],
            client.device(),
        )?)
    }
}

pub(crate) fn inputs<R>(
    client: &R::Client,
    owners: &[i64],
    high: &[f64],
    low: &[f64],
    strided: bool,
) -> Result<(Tensor<R>, CertifiedExpansion<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(owners.len(), high.len());
    assert_eq!(owners.len(), low.len());
    let count = owners.len();
    let owners = if strided && count != 0 {
        let storage: Vec<_> = owners.iter().flat_map(|&owner| [i64::MAX, owner]).collect();
        let view = Tensor::from_slice(&storage, &[count, 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(view.offset() > 0);
        if count > 1 {
            assert!(!view.is_contiguous());
        }
        view
    } else {
        Tensor::from_slice(owners, &[count], client.device())?
    };
    let high = tensor::<R>(client, high, strided)?;
    let low = tensor::<R>(client, low, strided)?;
    let radius = Tensor::zeros(&[count], DType::F64, client.device())?;
    Ok((owners, expansion::<R>(client, &high, &low, &radius)?))
}

pub(crate) fn expansion<R>(
    client: &R::Client,
    high: &Tensor<R>,
    low: &Tensor<R>,
    radius: &Tensor<R>,
) -> Result<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    Ok(with_interval_arithmetic::<R, _, _, _>(
        client,
        high.dtype(),
        "area point test input",
        |a| a.expansion_bounds(high, low, radius),
    )?)
}

pub(crate) fn bits<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> Result<Vec<u64>> {
    Ok(tensor
        .try_to_vec::<f64>()?
        .into_iter()
        .map(f64::to_bits)
        .collect())
}

pub(crate) fn interval<R: Runtime<DType = DType>>(value: &Interval<R>, shape: &[usize]) {
    for endpoint in [value.lower(), value.upper()] {
        assert_eq!(endpoint.shape(), shape);
        assert_eq!(endpoint.dtype(), DType::F64);
    }
}

pub(crate) fn overwrite<R>(client: &R::Client, tensor: &Tensor<R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    client.copy_into(
        tensor,
        &Tensor::zeros(tensor.shape(), tensor.dtype(), client.device())?,
    )?;
    Ok(())
}
