use super::fixtures::{PROFILES, ProfileCase, Truth};
use hydr::channel::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticSectionBatch,
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

pub(crate) fn definition(profile: ProfileCase) -> CrossSectionDefinition {
    CrossSectionDefinition {
        id: profile.id.into(),
        vertical_datum: "Analytical geometry datum".into(),
        points: profile
            .points
            .iter()
            .map(|point| CrossSectionPoint {
                station_m: f64::from_bits(point[0]),
                elevation_m: f64::from_bits(point[1]),
            })
            .collect(),
        maximum_stage_m: f64::from_bits(profile.maximum_stage_bits),
        profile_provenance: "Exact original-coordinate segment integration".into(),
    }
}

pub(crate) fn profile(index: usize) -> Result<&'static ProfileCase> {
    PROFILES
        .get(index)
        .ok_or_else(|| format!("Reference profile index {index} is outside frozen profiles. Check the fixture owner mapping.").into())
}

pub(crate) fn batch<R>(client: &R::Client) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definitions = PROFILES.iter().copied().map(definition).collect();
    Ok(CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?)
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
    let count = owners.len();
    if strided && count != 0 {
        let owner_storage: Vec<_> = owners.iter().flat_map(|&owner| [i64::MAX, owner]).collect();
        let value_storage: Vec<_> = values.iter().flat_map(|&value| [f64::NAN, value]).collect();
        let owners = Tensor::from_slice(&owner_storage, &[count, 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        let values = Tensor::from_slice(&value_storage, &[count, 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        if count > 1 {
            assert!(!owners.is_contiguous());
            assert!(!values.is_contiguous());
        }
        assert!(owners.offset() > 0);
        assert!(values.offset() > 0);
        return Ok((owners, values));
    }
    Ok((
        Tensor::from_slice(owners, &[count], client.device())?,
        Tensor::from_slice(values, &[count], client.device())?,
    ))
}

pub(crate) fn snapshots<R>(
    owners: &Tensor<R>,
    values: &Tensor<R>,
    dry: &Tensor<R>,
    expected_owners: &[i64],
    expected_bits: &[u64],
) -> Result
where
    R: Runtime<DType = DType>,
{
    let count = expected_owners.len();
    assert_eq!(owners.shape(), &[count]);
    assert_eq!(owners.dtype(), DType::I64);
    assert_eq!(values.shape(), &[count]);
    assert_eq!(values.dtype(), DType::F64);
    assert_eq!(dry.shape(), &[count]);
    assert_eq!(dry.dtype(), DType::F64);
    assert_eq!(owners.try_to_vec::<i64>()?, expected_owners);
    let actual_bits: Vec<_> = values
        .try_to_vec::<f64>()?
        .into_iter()
        .map(f64::to_bits)
        .collect();
    assert_eq!(actual_bits, expected_bits);
    let expected_dry: Vec<_> = expected_bits
        .iter()
        .map(|&bits| {
            if f64::from_bits(bits) == 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    assert_eq!(dry.try_to_vec::<f64>()?, expected_dry);
    Ok(())
}

pub(crate) fn native_bounds<R: Runtime<DType = DType>>(bounds: &Interval<R>, count: usize) {
    for endpoint in [bounds.lower(), bounds.upper()] {
        assert_eq!(endpoint.shape(), &[count]);
        assert_eq!(endpoint.dtype(), DType::F64);
    }
}

pub(crate) fn dry_placeholders<R: Runtime<DType = DType>>(
    bounds: &Interval<R>,
    operand_bits: &[u64],
) -> Result {
    let lower = bounds.lower().try_to_vec::<f64>()?;
    let upper = bounds.upper().try_to_vec::<f64>()?;
    assert_eq!(lower.len(), operand_bits.len());
    assert_eq!(upper.len(), operand_bits.len());
    for ((lower, upper), &bits) in lower.into_iter().zip(upper).zip(operand_bits) {
        if f64::from_bits(bits) == 0.0 {
            assert_eq!(lower, 0.0);
            assert_eq!(upper, 0.0);
        }
    }
    Ok(())
}

pub(crate) fn enclosed(actual: [f64; 2], truth: Truth, label: &str) {
    assert!(
        actual[0].is_finite() && actual[1].is_finite(),
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] >= 0.0 && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    let lower = f64::from_bits(truth.lower_bits);
    let upper = f64::from_bits(truth.upper_bits);
    assert!(
        actual[0] <= lower,
        "{label}: lower {} excludes {lower}",
        actual[0]
    );
    assert!(
        actual[1] >= upper,
        "{label}: upper {} excludes {upper}",
        actual[1]
    );
    // Nearby nonnegative endpoints subtract exactly by Sterbenz's lemma.
    let width = actual[1] - actual[0];
    let maximum = f64::from_bits(truth.maximum_width_bits);
    assert!(
        width <= maximum,
        "{label}: width {width} exceeds frozen {maximum}"
    );
}

pub(crate) fn optional(actual: Option<[f64; 2]>, truth: Option<Truth>, label: &str) -> Result {
    match (actual, truth) {
        (Some(actual), Some(truth)) => enclosed(actual, truth, label),
        (None, None) => {}
        _ => {
            return Err(
                format!("{label}: wet/dry optional field disagrees with the reference").into(),
            );
        }
    }
    Ok(())
}

pub(crate) fn overwrite_inputs<R>(
    client: &R::Client,
    owners: &Tensor<R>,
    values: &Tensor<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    client.copy_into(
        owners,
        &Tensor::zeros(owners.shape(), DType::I64, client.device())?,
    )?;
    client.copy_into(
        values,
        &Tensor::zeros(values.shape(), DType::F64, client.device())?,
    )?;
    Ok(())
}
