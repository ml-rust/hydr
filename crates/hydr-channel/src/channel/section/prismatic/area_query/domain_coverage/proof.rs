use super::super::{admission, selection};
use super::ProofCandidates;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;

/// Assess retained cap certificates independently of model binding and query construction.
pub(super) fn proof_candidates<R>(
    a: &mut Arithmetic<'_, R>,
    capacity: &CertifiedExpansion<R>,
) -> HydraulicResult<ProofCandidates<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let count = capacity.high().numel();
    for component in [capacity.high(), capacity.low(), capacity.abs_error()] {
        admission::metadata(a, component, &[count], a.dtype())?;
    }
    let c = a.client();
    let interval = a.expansion_interval(capacity)?;
    // This represented point is used for both chart widths. Its rounded value receives no floor.
    let span = a.expansion_point(&c.mul_scalar(interval.lower(), 0.75)?)?;
    let zero = Tensor::zeros(&[count], a.dtype(), c.device())?;
    let owner = c.nonzero(&c.gt(span.high(), &zero)?)?.flatten()?;
    let capacity = selection::expansion(a, capacity, &owner)?;
    let span = selection::expansion(a, &span, &owner)?;
    let cap_minus_lower = a.expansion_sub(&capacity, &span)?;
    let cap_minus_deficit = a.expansion_sub(&capacity, &span)?;
    let overlap_gap = a.expansion_sub(&cap_minus_lower, &span)?;
    let lower = a.expansion_interval(&cap_minus_lower)?;
    let deficit = a.expansion_interval(&cap_minus_deficit)?;
    let overlap = a.expansion_interval(&overlap_gap)?;
    let zero = Tensor::zeros(owner.shape(), a.dtype(), c.device())?;
    let proved = c.mul(
        &c.mul(&c.ge(lower.lower(), &zero)?, &c.ge(deficit.lower(), &zero)?)?,
        &c.le(overlap.upper(), &zero)?,
    )?;
    Ok(ProofCandidates {
        owner,
        span,
        proved,
        #[cfg(test)]
        cap_minus_lower,
        #[cfg(test)]
        cap_minus_deficit,
        #[cfg(test)]
        overlap_gap,
    })
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::{
        AreaAnchor, AreaDirection, AreaQuery,
        endpoints::PhysicalEndpoint,
        query_fixture::{Snapshot, snapshot},
        test_support as support,
    };
    use super::super::CoveredDomain;
    use super::*;
    use crate::channel::{flow::surveyed::error, section::HydraulicGeometry};
    use numr::ops::IndexingOps;
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

    pub(in super::super) fn check<R>(
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

    pub(in super::super) fn wide_arithmetic<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        // These are honest arithmetic certificates, not substituted compiled-model boundaries.
        let capacity = support::expansion::<R>(
            client,
            &[
                [4.0_f64.to_bits(), 0, 0],
                [4.0_f64.to_bits(), 0, 2.0_f64.to_bits()],
            ],
            false,
        )?;
        hydraulic(error::checked::<R, _>(
            client,
            "wide capacity overlap proof",
            |a| {
                let candidates = proof_candidates(a, &capacity)?;
                assert_eq!(candidates.owner.try_to_vec::<i64>()?, [0, 1]);
                let span = candidates.span.high().try_to_vec::<f64>()?;
                assert_eq!(span[0], 3.0);
                assert!(span[1].is_finite() && span[1] > 0.0 && span[1] <= 1.5);
                let certified_lower = a
                    .expansion_interval(&capacity)?
                    .lower()
                    .try_to_vec::<f64>()?;
                assert_eq!(
                    span.into_iter().map(f64::to_bits).collect::<Vec<_>>(),
                    certified_lower
                        .into_iter()
                        .map(|lower| (0.75 * lower).to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(candidates.span.low().try_to_vec::<f64>()?, [0.0, 0.0]);
                assert_eq!(candidates.span.abs_error().try_to_vec::<f64>()?, [0.0, 0.0]);
                assert_eq!(candidates.proved.try_to_vec::<f64>()?, [1.0, 0.0]);
                let low = a.expansion_interval(&candidates.cap_minus_lower)?;
                let cap = a.expansion_interval(&candidates.cap_minus_deficit)?;
                let gap = a.expansion_interval(&candidates.overlap_gap)?;
                assert_eq!(low.lower().try_to_vec::<f64>()?[0], 1.0);
                assert_eq!(low.upper().try_to_vec::<f64>()?[0], 1.0);
                assert_eq!(cap.lower().try_to_vec::<f64>()?[0], 1.0);
                assert_eq!(cap.upper().try_to_vec::<f64>()?[0], 1.0);
                assert_eq!(gap.lower().try_to_vec::<f64>()?[0], -2.0);
                assert_eq!(gap.upper().try_to_vec::<f64>()?[0], -2.0);
                assert!(
                    low.lower().try_to_vec::<f64>()?[1] > 0.0
                        && cap.lower().try_to_vec::<f64>()?[1] > 0.0
                );
                assert!(
                    gap.lower().try_to_vec::<f64>()?[1] < 0.0
                        && gap.upper().try_to_vec::<f64>()?[1] > 0.0
                );
                Ok(())
            },
        ))
    }
}
