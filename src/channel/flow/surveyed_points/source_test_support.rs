use super::fixtures::{PROFILES, Reference};
use crate::channel::flow::surveyed::error::HydraulicResult;
use crate::channel::section::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticSectionBatch,
};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval, with_interval_arithmetic};

pub(in crate::channel::flow::surveyed_points) type Result<T = ()> =
    std::result::Result<T, Box<dyn std::error::Error>>;

pub(in crate::channel::flow::surveyed_points) fn cases() -> &'static [super::fixtures::Case] {
    super::fixtures::CASES
}

pub(in crate::channel::flow::surveyed_points) fn profiles()
-> impl Iterator<Item = (&'static str, u64, &'static [[u64; 2]])> {
    PROFILES
        .iter()
        .map(|profile| (profile.id, profile.cap_stage, profile.coordinates))
}

pub(in crate::channel::flow::surveyed_points) trait Client<R: Runtime<DType = DType>>:
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

pub(in crate::channel::flow::surveyed_points) fn hydraulic<T>(
    result: HydraulicResult<T>,
) -> Result<T> {
    result.map_err(|error| std::io::Error::other(format!("{error:?}")).into())
}

pub(in crate::channel::flow::surveyed_points) fn batch<R>(
    client: &R::Client,
) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definitions = PROFILES
        .iter()
        .map(|profile| CrossSectionDefinition {
            id: profile.id.into(),
            vertical_datum: "Frozen source profile datum".into(),
            points: profile
                .coordinates
                .iter()
                .map(|point| CrossSectionPoint {
                    station_m: f64::from_bits(point[0]),
                    elevation_m: f64::from_bits(point[1]),
                })
                .collect(),
            maximum_stage_m: f64::from_bits(profile.cap_stage),
            profile_provenance: "Frozen original surveyed coordinates".into(),
        })
        .collect();
    Ok(CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?)
}

fn values<R>(client: &R::Client, values: &[f64], strided: bool) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    if strided && !values.is_empty() {
        let storage: Vec<_> = values.iter().flat_map(|&value| [f64::NAN, value]).collect();
        let view = Tensor::from_slice(&storage, &[values.len(), 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(view.offset() > 0);
        if values.len() > 1 {
            assert!(!view.is_contiguous());
        }
        Ok(view)
    } else {
        Ok(Tensor::from_slice(
            values,
            &[values.len()],
            client.device(),
        )?)
    }
}

pub(in crate::channel::flow::surveyed_points) fn expansion<R>(
    client: &R::Client,
    points: &[[u64; 3]],
    strided: bool,
) -> Result<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let tensors = (0..3)
        .map(|field| {
            values::<R>(
                client,
                &points
                    .iter()
                    .map(|point| f64::from_bits(point[field]))
                    .collect::<Vec<_>>(),
                strided,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(with_interval_arithmetic::<R, _, _, _>(
        client,
        DType::F64,
        "source fixture capture",
        |a| a.expansion_bounds(&tensors[0], &tensors[1], &tensors[2]),
    )?)
}

pub(in crate::channel::flow::surveyed_points) fn components<R: Runtime<DType = DType>>(
    value: &CertifiedExpansion<R>,
) -> Result<Vec<[u64; 3]>> {
    let high = value.high().try_to_vec::<f64>()?;
    let low = value.low().try_to_vec::<f64>()?;
    let radius = value.abs_error().try_to_vec::<f64>()?;
    Ok(high
        .into_iter()
        .zip(low)
        .zip(radius)
        .map(|((high, low), radius)| [high.to_bits(), low.to_bits(), radius.to_bits()])
        .collect())
}

pub(in crate::channel::flow::surveyed_points) fn bounds<R>(
    client: &R::Client,
    value: &CertifiedExpansion<R>,
) -> Result<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    Ok(with_interval_arithmetic::<R, _, _, _>(
        client,
        DType::F64,
        "source certificate projection",
        |a| a.expansion_interval(value),
    )?)
}

pub(in crate::channel::flow::surveyed_points) fn endpoints<R: Runtime<DType = DType>>(
    value: &Interval<R>,
) -> Result<Vec<[f64; 2]>> {
    Ok(value
        .lower()
        .try_to_vec::<f64>()?
        .into_iter()
        .zip(value.upper().try_to_vec::<f64>()?)
        .map(|(lower, upper)| [lower, upper])
        .collect())
}

pub(in crate::channel::flow::surveyed_points) fn contained(
    actual: [f64; 2],
    reference: Reference,
    label: &str,
) {
    let expected = reference.bounds.map(f64::from_bits);
    assert!(
        actual[0].is_finite() && actual[1].is_finite() && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{label}: {actual:?} excludes {expected:?}"
    );
}

pub(in crate::channel::flow::surveyed_points) fn zero(point: [u64; 3], label: &str) {
    assert!(
        point.map(f64::from_bits).iter().all(|&value| value == 0.0),
        "{label}: {point:x?}"
    );
}

/// True when a case's captured states are exact mirrors: both sides wet, equal areas in every
/// component, and discharges that are exact negatives with equal radii.
pub(in crate::channel::flow::surveyed_points) fn exact_mirror(id: &str) -> Result<bool> {
    let case = cases()
        .iter()
        .find(|case| case.id == id)
        .ok_or("Missing frozen point source")?;
    let [left_area, left_q, right_area, right_q] =
        case.inputs.map(|point| point.map(f64::from_bits));
    Ok(left_area[0] > 0.0
        && left_area == right_area
        && left_q[0] == -right_q[0]
        && left_q[1] == -right_q[1]
        && left_q[2] == right_q[2])
}

/// A mirror face holds the exact water value +0.0, which the independent truth must enclose.
/// Every other face's water bound encloses the truth.
pub(in crate::channel::flow::surveyed_points) fn water_enclosed(
    actual: [f64; 2],
    truth: [f64; 2],
    mirror: bool,
    label: &str,
) {
    if mirror {
        assert_eq!(actual.map(f64::to_bits), [0, 0], "{label}: {actual:?}");
        assert!(
            truth[0] <= 0.0 && 0.0 <= truth[1],
            "{label}: {truth:?} excludes the mirror value 0"
        );
    } else {
        assert!(
            actual[0] <= truth[0] && truth[1] <= actual[1],
            "{label}: {actual:?} excludes {truth:?}"
        );
    }
}
