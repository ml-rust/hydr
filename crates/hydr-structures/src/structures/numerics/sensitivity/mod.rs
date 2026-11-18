//! Relative head sensitivity of each law: a bound on `|dQ/dH| / |Q|` per metre of either head.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::structures::tensor::TensorStructureContext;
use numr::tensor::Tensor;

#[cfg(test)]
pub(crate) mod fixture;

/// Exponent of the upstream head in the free weir law and the free gate strip.
pub(super) const HEAD_EXPONENT: f64 = 1.5;

/// Exponent of the head gap in the orifice law and the submerged gate strip.
pub(super) const GAP_EXPONENT: f64 = 0.5;

/// `coefficient / head` where `head` is positive, else zero.
pub(super) fn per_metre<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    head: &Tensor<R>,
    coefficient: f64,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::CompareOps<R> + numr::ops::ConditionalOps<R>,
{
    let c = &context.client;
    let scalar = |value: f64| Tensor::<R>::full_scalar(&[], DType::F64, value, &context.device);
    let zero = scalar(0.0)?;
    let positive = c.gt(head, &zero)?;
    let safe = c.where_cond(&positive, head, &scalar(1.0)?)?;
    Ok(c.where_cond(&positive, &c.div(&scalar(coefficient)?, &safe)?, &zero)?)
}

/// Sensitivity of the weir and orifice laws.
///
/// - Weir: `1.5 / h_u + 0.385 / gap`. `h_u` is the upstream head above the crest. `gap` is the
///   head difference of submerged flow and zero for free flow. The submerged law is
///   `Q ∝ h_u^1.115 gap^0.385 k^0.385` with `k = 1 + r / (1 + sqrt r)`, `r = h_d / h_u`.
///   `d ln k / dr` lies in `[0, 1]`, so each head moves `ln Q` by at most this sum.
/// - Orifice: `Q ∝ sqrt(gap)`, so `0.5 / gap`.
pub(super) fn passive<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    orifice: &Tensor<R>,
    upstream: &Tensor<R>,
    gap: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::CompareOps<R> + numr::ops::ConditionalOps<R>,
{
    let c = &context.client;
    let weir = c.add(
        &per_metre(context, upstream, HEAD_EXPONENT)?,
        &per_metre(context, gap, super::weir::VILLEMONTE_EXPONENT)?,
    )?;
    Ok(c.where_cond(orifice, &per_metre(context, gap, GAP_EXPONENT)?, &weir)?)
}

#[cfg(test)]
mod tests {
    use super::fixture::{
        BED, Lane, TestResult, check_bound, column, flag, grid, lanes, structures,
    };
    use crate::structures::coupling::CouplingStructures;
    use crate::structures::{
        DirectionPolicy, OrificeApplicability, StructureLaw, StructureOperatingPoint,
        SubmergedOrificeParameters, SuppressedRectangularWeirParameters, WeirApplicability,
    };
    use numr::dtype::DType;
    use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn weir() -> TestResult<StructureLaw> {
        Ok(StructureLaw::SuppressedRectangularWeir(
            SuppressedRectangularWeirParameters::new(
                2.0,
                1.0,
                1.7,
                WeirApplicability {
                    coefficient_provenance: "Analytical coefficient".into(),
                    free_nappe_and_approach_assertion: "Analytical suppressed crest".into(),
                    submergence_correction_assertion: "Adopted Villemonte correction".into(),
                    symmetric_reversal: true,
                },
            )?,
        ))
    }

    fn passive_structures(
        law: &StructureLaw,
        count: usize,
    ) -> TestResult<CouplingStructures<CpuRuntime>> {
        structures(
            |_| {
                (
                    law.clone(),
                    DirectionPolicy::Bidirectional,
                    StructureOperatingPoint::Passive,
                )
            },
            count,
        )
    }

    /// The surface/channel resolution slope as it stood before the shared field, verbatim.
    fn coordinator_slope(
        client: &CpuClient,
        upstream: &Tensor<CpuRuntime>,
        gap: &Tensor<CpuRuntime>,
    ) -> TestResult<Vec<f64>> {
        let device = CpuDevice::new();
        let scalar =
            |value: f64| Tensor::<CpuRuntime>::full_scalar(&[], DType::F64, value, &device);
        let zero = scalar(0.0)?;
        let per_metre =
            |head: &Tensor<CpuRuntime>, exponent: f64| -> TestResult<Tensor<CpuRuntime>> {
                let positive = client.gt(head, &zero)?;
                let safe = client.where_cond(&positive, head, &scalar(1.0)?)?;
                Ok(client.where_cond(&positive, &client.div(&scalar(exponent)?, &safe)?, &zero)?)
            };
        Ok(client
            .add(&per_metre(upstream, 1.5)?, &per_metre(gap, 0.385)?)?
            .try_to_vec()?)
    }

    #[test]
    fn weir_sensitivity_matches_the_coordinator_slope_bit_for_bit() -> TestResult {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut draw = |low: f64, high: f64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            low + (high - low) * ((state >> 11) as f64 / (1_u64 << 53) as f64)
        };
        let mut cases = Vec::new();
        for index in 0..4000 {
            // Free, submerged, near-level, below-crest, and one-side-dry heads.
            let a = draw(0.5, 6.0);
            let b = match index % 4 {
                0 => a + draw(-1e-9, 1e-9),
                _ => draw(0.5, 6.0),
            };
            cases.push(Lane {
                stage: [a, b],
                wet: [true, index % 7 != 0],
                step: 0.0,
            });
        }
        cases.push(Lane {
            stage: [2.0, 2.0],
            wet: [true, true],
            step: 0.0,
        });
        let n = cases.len();
        let structures = passive_structures(&weir()?, n)?;
        let law = structures.evaluate_heads(
            &column(cases.iter().map(|l| l.stage[0]).collect())?,
            &column(cases.iter().map(|l| l.stage[1]).collect())?,
            &column(vec![BED; n])?,
            &column(vec![BED; n])?,
            &column(vec![1.0; n])?,
            &column(cases.iter().map(|l| flag(l.wet[1])).collect())?,
        )?;
        let client = CpuClient::new(CpuDevice::new());
        let expected = coordinator_slope(&client, &law.upstream_head, &law.head_gap)?;
        let actual: Vec<f64> = law.head_sensitivity.try_to_vec()?;
        assert_eq!(actual.len(), expected.len());
        let mut positive = 0;
        for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "lane {index}: {:?}",
                cases[index].stage
            );
            positive += usize::from(*a > 0.0);
        }
        assert!(
            positive > n / 2,
            "most lanes carry water: {positive} of {n}"
        );
        Ok(())
    }

    #[test]
    fn weir_sensitivity_bounds_free_and_submerged_flow() -> TestResult {
        let values = grid(0.2, 4.0, 40);
        let mut cases = lanes(&values, &[1.0], [true, true]);
        cases.extend(lanes(&values, &[1.0], [true, false]));
        let flowing = check_bound(&passive_structures(&weir()?, cases.len())?, &cases)?;
        assert!(flowing > cases.len() / 2, "{flowing} of {}", cases.len());
        Ok(())
    }

    #[test]
    fn orifice_sensitivity_bounds_submerged_flow() -> TestResult {
        let law = StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
            0.5,
            0.0,
            1.0,
            0.6,
            9.81,
            OrificeApplicability {
                coefficient_provenance: "Analytical coefficient".into(),
                fully_submerged_applicability: "Both stages above the crown".into(),
                symmetric_reversal: true,
            },
        )?);
        let cases = lanes(&grid(1.01, 6.0, 40), &[1.0], [true, true]);
        let flowing = check_bound(&passive_structures(&law, cases.len())?, &cases)?;
        assert_eq!(flowing, cases.len());
        Ok(())
    }
}
