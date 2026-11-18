use super::error::{Arithmetic, HydraulicResult, checked};
use super::potential_difference_test_data::Profile;
use crate::channel::section::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticSectionBatch,
};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) fn values<R, T>(
    client: &R::Client,
    values: &[T],
    strided: bool,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    T: numr::dtype::Element + Copy,
{
    let count = values.len();
    if strided {
        let repeated: Vec<T> = values.iter().flat_map(|&value| [value, value]).collect();
        Ok(Tensor::from_slice(&repeated, &[count, 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1)))
    } else {
        Ok(Tensor::from_slice(values, &[count], client.device())?)
    }
}

pub(super) fn operand<R>(a: &mut Arithmetic<'_, R>, bits: [u64; 2]) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.bounds(
        &Tensor::from_slice(&[f64::from_bits(bits[0])], &[1], a.client().device())?,
        &Tensor::from_slice(&[f64::from_bits(bits[1])], &[1], a.client().device())?,
    )?)
}

pub(crate) fn batch<R>(
    client: &R::Client,
    profile: &Profile,
) -> HydraulicResult<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let definition = CrossSectionDefinition {
        id: profile.id.into(),
        vertical_datum: "Frozen local survey datum".into(),
        points: profile
            .coordinates
            .iter()
            .map(|xy| CrossSectionPoint {
                station_m: f64::from_bits(xy[0]),
                elevation_m: f64::from_bits(xy[1]),
            })
            .collect(),
        maximum_stage_m: f64::from_bits(profile.cap),
        profile_provenance: "Independent potential difference reference coordinates".into(),
    };
    Ok(CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?)
}

pub(super) fn contains<R>(
    value: &Interval<R>,
    reference: [u64; 2],
    label: &str,
    tight: bool,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
{
    let lower = value.lower().try_to_vec::<f64>()?;
    let upper = value.upper().try_to_vec::<f64>()?;
    let reference = reference.map(f64::from_bits);
    for (&lower, &upper) in lower.iter().zip(&upper) {
        assert!(
            lower <= reference[0] && upper >= reference[1],
            "{label}: actual=[{lower},{upper}], reference={reference:?}"
        );
        if tight {
            let allowance = 1e-13 + 1e-12 * reference[0].abs().max(reference[1].abs());
            assert!(
                upper - lower <= allowance,
                "{label}: width={}, allowance={allowance}",
                upper - lower
            );
        }
    }
    Ok(())
}

pub(super) fn target<R>(
    client: &R::Client,
    bits: [u64; 2],
    strided: bool,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let lower = values::<R, _>(client, &[f64::from_bits(bits[0]); 3], strided)?;
    let upper = values::<R, _>(client, &[f64::from_bits(bits[1]); 3], strided)?;
    checked(client, "frozen complete target interval", |a| {
        Ok(a.bounds(&lower, &upper)?)
    })
}
