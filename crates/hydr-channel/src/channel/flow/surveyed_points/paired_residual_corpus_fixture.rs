use crate::channel::flow::surveyed_points::{
    source_fixture_support::{self as support, Client, Result, components, expansion, hydraulic},
    source_potential_fixtures as source_references,
    source_velocity::{self, SourceVelocity},
};
use crate::channel::section::PrismaticSectionBatch;
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;

pub(super) fn source_reference(id: &str) -> Result<&source_references::Case> {
    source_references::CASES
        .iter()
        .find(|case| case.id == id)
        .ok_or_else(|| "Missing original source potential reference".into())
}

pub(super) struct Corpus<R: Runtime<DType = DType>> {
    pub section: PrismaticSectionBatch<R>,
    pub velocity: SourceVelocity<R>,
    pub inputs: Vec<CertifiedExpansion<R>>,
    pub owners: Tensor<R>,
}

pub(super) fn corpus<R>(client: &R::Client, strided: bool) -> Result<Corpus<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let cases = support::cases();
    assert_eq!(cases.len(), 70);
    let section = support::batch::<R>(client)?;
    let owner_values = cases
        .iter()
        .map(|case| i64::try_from(case.owner))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let owners = if strided {
        let storage: Vec<_> = owner_values
            .iter()
            .flat_map(|&owner| [i64::MAX, owner])
            .collect();
        let view = Tensor::from_slice(&storage, &[cases.len(), 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(!view.is_contiguous() && view.offset() > 0);
        view
    } else {
        Tensor::from_slice(&owner_values, &[cases.len()], client.device())?
    };
    let inputs = (0..4)
        .map(|field| {
            expansion::<R>(
                client,
                &cases
                    .iter()
                    .map(|case| case.inputs[field])
                    .collect::<Vec<_>>(),
                strided,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let captured = hydraulic(
        section.capture_riemann_points(&owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3]),
    )?;
    let velocity = hydraulic(source_velocity::prepare::<R>(client, &captured))?;
    assert!(std::ptr::eq(velocity.input(), captured.as_ref()));
    Ok(Corpus {
        section,
        velocity,
        inputs,
        owners,
    })
}

#[derive(Debug, PartialEq)]
pub(super) struct Snapshot {
    owners: Vec<i64>,
    components: Vec<Vec<[u64; 3]>>,
    masks: Vec<Vec<u64>>,
}

pub(super) fn snapshot<R: Runtime<DType = DType>>(
    velocity: &SourceVelocity<R>,
) -> Result<Snapshot> {
    let input = velocity.input();
    Ok(Snapshot {
        owners: input.owners.try_to_vec::<i64>()?,
        components: [
            &input.left_area,
            &input.left_discharge,
            &input.right_area,
            &input.right_discharge,
            &velocity.left,
            &velocity.right,
            &velocity.difference,
        ]
        .into_iter()
        .map(components)
        .collect::<Result<_>>()?,
        masks: [
            &input.left_dry,
            &input.right_dry,
            &velocity.both_wet,
            &velocity.uniform,
        ]
        .into_iter()
        .map(|mask| {
            Ok(mask
                .try_to_vec::<f64>()?
                .into_iter()
                .map(f64::to_bits)
                .collect())
        })
        .collect::<Result<_>>()?,
    })
}

pub(super) fn reference(actual: [f64; 2], bounds: [u64; 2], allowance: u64, label: &str) {
    let expected = bounds.map(f64::from_bits);
    let allowance = f64::from_bits(allowance);
    assert!(
        actual[0].is_finite() && actual[1].is_finite() && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{label}: {actual:?} excludes {expected:?}"
    );
    let excess = (expected[1] - actual[0]).max(actual[1] - expected[0]);
    assert!(
        excess <= allowance,
        "{label}: excess={excess:e}, allowance={allowance:e}, {actual:?}"
    );
}
