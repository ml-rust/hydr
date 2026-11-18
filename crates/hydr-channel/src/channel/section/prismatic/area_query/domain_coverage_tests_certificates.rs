use super::super::{
    AreaAnchor, AreaDirection, AreaQuery,
    endpoints::PhysicalEndpoint,
    parameter_rebinding_tests::{Snapshot, snapshot},
};
use super::{CoveredDomain, support};
use crate::channel::{flow::surveyed::error, section::HydraulicGeometry};
use numr::{
    dtype::DType,
    ops::{BinaryOps, IndexingOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;
use support::{Client, Result, hydraulic};

fn bits<R: Runtime<DType = DType>>(value: &CertifiedExpansion<R>) -> Result<Vec<Vec<u64>>> {
    [value.high(), value.low(), value.abs_error()]
        .into_iter()
        .map(|field| {
            Ok(field
                .try_to_vec::<f64>()?
                .into_iter()
                .map(f64::to_bits)
                .collect())
        })
        .collect()
}

#[derive(Debug, PartialEq)]
struct PairState {
    lower: Snapshot,
    cap: Snapshot,
    witnesses: Vec<Vec<Vec<u64>>>,
}

fn state<R: Runtime<DType = DType>>(pair: &CoveredDomain<R>) -> Result<PairState> {
    Ok(PairState {
        lower: snapshot(&pair.lower)?,
        cap: snapshot(&pair.cap)?,
        witnesses: [
            &pair.cap_minus_lower,
            &pair.cap_minus_deficit,
            &pair.overlap_gap,
        ]
        .into_iter()
        .map(bits)
        .collect::<Result<Vec<_>>>()?,
    })
}

pub(super) fn check<R>(
    client: &R::Client,
    geometry: &HydraulicGeometry<R>,
    input_owners: &Tensor<R>,
    diagnostics: &Tensor<R>,
    owner: &Tensor<R>,
    proved: &Tensor<R>,
    pair: &CoveredDomain<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    hydraulic(error::checked::<R, _>(
        client,
        "covered pair retained identities and support",
        |a| {
            let bands = geometry.gather(a, input_owners)?.select(a, owner)?;
            let boundaries = geometry
                .boundary_areas(a)?
                .gather(a, input_owners)?
                .select(a, owner)?;
            let capacity = boundaries.boundary(a, boundaries.cap_indices())?;
            let capacity_bounds = a.expansion_interval(&capacity)?;
            let candidates = capacity_bounds
                .lower()
                .try_to_vec::<f64>()?
                .into_iter()
                .map(|lower| 0.75 * lower)
                .collect::<Vec<_>>();
            assert!(
                candidates
                    .iter()
                    .all(|&value| value > 0.0 && value.is_finite())
            );
            let width = Tensor::from_slice(&candidates, &[candidates.len()], client.device())?;
            let span = a.expansion_point(&width)?;
            for query in [&pair.lower, &pair.cap] {
                assert_eq!(
                    query
                        .query
                        .parameter
                        .upper()
                        .try_to_vec::<f64>()?
                        .into_iter()
                        .map(f64::to_bits)
                        .collect::<Vec<_>>(),
                    candidates
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            let expected_lower = a.expansion_sub(&capacity, &span)?;
            let expected_deficit = a.expansion_sub(&capacity, &span)?;
            let expected_gap = a.expansion_sub(&expected_lower, &span)?;
            for (actual, expected) in [
                (&pair.cap_minus_lower, &expected_lower),
                (&pair.cap_minus_deficit, &expected_deficit),
                (&pair.overlap_gap, &expected_gap),
            ] {
                assert_eq!(
                    bits(actual).map_err(|error| {
                        crate::channel::flow::surveyed::error::HydraulicError::Numerical {
                            context: error.to_string(),
                        }
                    })?,
                    bits(expected).map_err(|error| {
                        crate::channel::flow::surveyed::error::HydraulicError::Numerical {
                            context: error.to_string(),
                        }
                    })?
                );
            }
            let zero = Tensor::zeros(owner.shape(), DType::F64, client.device())?;
            let one = Tensor::ones(owner.shape(), DType::F64, client.device())?;
            let scale = a.expansion_point(&one)?;
            let parameter = a.bounds(&zero, &width)?;
            let lower = AreaQuery::band_affine(
                a,
                &bands,
                AreaAnchor::Zero,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?;
            let cap = AreaQuery::cap_deficit(a, &bands, &scale, &parameter)?;
            let selected_diagnostics = client.index_select(diagnostics, 0, owner)?;
            client.copy_into(&lower.original_rows, &selected_diagnostics)?;
            client.copy_into(&cap.original_rows, &selected_diagnostics)?;
            let lower = lower.prove_support(a, &bands)?;
            let cap = cap.prove_support(a, &bands)?;
            assert!(Arc::ptr_eq(&lower.query.model, &pair.lower.query.model));
            assert!(Arc::ptr_eq(&cap.query.model, &pair.cap.query.model));
            assert_eq!(
                snapshot(&lower).map_err(|error| {
                    crate::channel::flow::surveyed::error::HydraulicError::Numerical {
                        context: error.to_string(),
                    }
                })?,
                snapshot(&pair.lower).map_err(|error| {
                    crate::channel::flow::surveyed::error::HydraulicError::Numerical {
                        context: error.to_string(),
                    }
                })?
            );
            assert_eq!(
                snapshot(&cap).map_err(|error| {
                    crate::channel::flow::surveyed::error::HydraulicError::Numerical {
                        context: error.to_string(),
                    }
                })?,
                snapshot(&pair.cap).map_err(|error| {
                    crate::channel::flow::surveyed::error::HydraulicError::Numerical {
                        context: error.to_string(),
                    }
                })?
            );
            let low_start = pair.lower.endpoint(a, &bands, PhysicalEndpoint::Lower)?;
            let low_end = pair.lower.endpoint(a, &bands, PhysicalEndpoint::Upper)?;
            let cap_start = pair.cap.endpoint(a, &bands, PhysicalEndpoint::Lower)?;
            let cap_end = pair.cap.endpoint(a, &bands, PhysicalEndpoint::Upper)?;
            let actual = [
                low_start.retained_endpoint_area(a, &bands, &boundaries)?,
                low_end.retained_endpoint_area(a, &bands, &boundaries)?,
                cap_start.retained_endpoint_area(a, &bands, &boundaries)?,
                cap_end.retained_endpoint_area(a, &bands, &boundaries)?,
            ];
            let zero = a.expansion_point(&zero)?;
            for (actual, expected) in
                actual
                    .iter()
                    .zip([&zero, &span, &expected_deficit, &capacity])
            {
                let difference = a.expansion_sub(actual, expected)?;
                let interval = a.expansion_interval(&difference)?;
                assert!(
                    interval
                        .lower()
                        .try_to_vec::<f64>()?
                        .into_iter()
                        .all(|value| value <= 0.0)
                );
                assert!(
                    interval
                        .upper()
                        .try_to_vec::<f64>()?
                        .into_iter()
                        .all(|value| value >= 0.0)
                );
            }
            Ok(())
        },
    ))?;
    let before = state(pair)?;
    for value in [input_owners, diagnostics, owner, proved] {
        client.copy_into(
            value,
            &Tensor::zeros(value.shape(), value.dtype(), client.device())?,
        )?;
    }
    assert_eq!(state(pair)?, before);
    Ok(())
}
