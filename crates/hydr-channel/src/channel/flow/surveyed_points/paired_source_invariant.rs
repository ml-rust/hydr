use super::paired_sources::{PairedSources, SourceInvariant};
use super::potential_range::select_expansion;
use super::wave_curve_types::{CurveValue, scatter_value, unavailable};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

#[cfg(test)]
pub(super) fn evaluate<R>(parent: &mut PairedSources<'_, R>) -> HydraulicResult<SourceInvariant<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = parent.geometry.client();
    let n = parent.original_rows.numel();
    let rows = c.nonzero(&parent.velocity.both_wet)?.flatten()?;
    let source = parent.source_values([&rows, &rows])?;
    let value = checked(c, "paired source invariant cancellation", |a| {
        let du = select_expansion(a, &parent.velocity.difference, &rows)?;
        let retained = a.expansion_sub(&du, &source[0].retained)?;
        let retained = a.expansion_sub(&retained, &source[1].retained)?;
        let physical = a.expansion_interval(&retained)?;
        let value = CurveValue {
            retained,
            physical,
            available: Tensor::ones(rows.shape(), a.dtype(), c.device())?,
        };
        let empty = unavailable(a, n)?;
        scatter_value(a, empty, &rows, &value)
    })?;
    checked::<R, _>(c, "paired source invariant classification", |a| {
        let input = parent.velocity.input();
        classify(a, n, value, &input.left_dry, &input.right_dry)
    })
}

pub(super) fn evaluate_on_rows<R>(
    parent: &mut PairedSources<'_, R>,
    rows: &Tensor<R>,
) -> HydraulicResult<SourceInvariant<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = parent.geometry.client();
    let selected = checked::<R, _>(c, "paired source invariant row admission", |a| {
        parent.select_sources(a, rows)
    })?;
    let n = selected.rows.numel();
    let owner = c.nonzero(&selected.velocity.both_wet)?.flatten()?;
    let rows = c.index_select(&selected.rows, 0, &owner)?;
    let source = parent.source_values([&rows, &rows])?;
    let value = checked::<R, _>(c, "paired selected source invariant cancellation", |a| {
        let du = select_expansion(a, &selected.velocity.difference, &owner)?;
        let retained = a.expansion_sub(&du, &source[0].retained)?;
        let retained = a.expansion_sub(&retained, &source[1].retained)?;
        let physical = a.expansion_interval(&retained)?;
        let value = CurveValue {
            retained,
            physical,
            available: Tensor::ones(owner.shape(), a.dtype(), c.device())?,
        };
        let empty = unavailable(a, n)?;
        scatter_value(a, empty, &owner, &value)
    })?;
    checked::<R, _>(c, "paired selected source invariant classification", |a| {
        let input = selected.velocity.input();
        classify(a, n, value, &input.left_dry, &input.right_dry)
    })
}

fn classify<R>(
    a: &mut crate::channel::flow::surveyed::error::Arithmetic<'_, R>,
    n: usize,
    value: CurveValue<R>,
    left_dry: &Tensor<R>,
    right_dry: &Tensor<R>,
) -> HydraulicResult<SourceInvariant<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
    let vacuum = c.mul(&value.available, &c.gt(value.physical.lower(), &zero)?)?;
    let wet_candidate = c.mul(&value.available, &c.lt(value.physical.upper(), &zero)?)?;
    let touching = c.mul(
        &value.available,
        &c.mul(
            &c.eq(value.retained.high(), &zero)?,
            &c.mul(
                &c.eq(value.retained.low(), &zero)?,
                &c.eq(value.retained.abs_error(), &zero)?,
            )?,
        )?,
    )?;
    #[cfg(test)]
    let known = c.maximum(&touching, &c.maximum(&vacuum, &wet_candidate)?)?;
    #[cfg(test)]
    let unresolved = c.mul(&value.available, &c.eq(&known, &zero)?)?;
    let both_dry = c.mul(left_dry, right_dry)?;
    Ok(SourceInvariant {
        value,
        vacuum,
        touching,
        wet_candidate,
        #[cfg(test)]
        unresolved,
        both_dry,
    })
}

#[cfg(test)]
pub(in crate::channel::flow::surveyed_points) mod tests {
    use super::*;
    use crate::channel::flow::surveyed::{
        PrismaticRiemannConfig,
        error::{self, HydraulicError, PrismaticRiemannRejection},
        work::WorkLedger,
    };
    use crate::channel::flow::surveyed_points::{
        paired_residual_corpus_fixture::{self as capture, Corpus},
        potential_range::PotentialEvaluation,
        source_fixture_support::{
            self as support, Client, Result, components, endpoints, hydraulic,
        },
    };
    use crate::channel::section::HydraulicGeometry;

    pub(in crate::channel::flow::surveyed_points) fn controls<R>(
        client: &R::Client,
        corpus: &Corpus<R>,
    ) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let cases = [
            (
                "scalar_rectangle_vacuum",
                2.0_f64.to_bits(),
                [1.0, 0.0, 0.0],
            ),
            ("scalar_rectangle_vacuum_threshold", 0, [0.0, 1.0, 0.0]),
            (
                "scalar_rectangle_vacuum_threshold_outside",
                0x3cc0000000000000,
                [1.0, 0.0, 0.0],
            ),
            (
                "scalar_rectangle_vacuum_threshold_inside",
                0xbcb0000000000000,
                [0.0, 0.0, 1.0],
            ),
        ];
        let source_cases = support::cases();
        for (id, expected, masks) in cases {
            let truth = capture::source_reference(id)?;
            let original = &source_cases[truth.source_index];
            assert_eq!(original.id, id);
            assert_eq!(truth.gravity_bits, 1.0_f64.to_bits());
            for side in &truth.sides {
                assert_eq!(side.phi.bounds, [2.0_f64.to_bits(); 2]);
                assert!(side.rectangular && !side.dry);
            }
            let rows = Tensor::from_slice(
                &[i64::try_from(truth.source_index)?; 3],
                &[3],
                client.device(),
            )?;
            let velocity = hydraulic(error::checked::<R, _>(
                client,
                "original vacuum source binding",
                |a| corpus.velocity.select(a, &rows),
            ))?;
            let before = capture::snapshot(&velocity)?;
            let geometry = HydraulicGeometry::from_batch(&corpus.section);
            let input = velocity.input();
            let config = PrismaticRiemannConfig {
                gravity_m_per_s2: f64::from_bits(truth.gravity_bits),
                ..PrismaticRiemannConfig::default()
            };
            let left = corpus.section.recover_area_points_tensor(
                &input.owners,
                &input.left_area,
                &config.recovery,
            )?;
            let right = corpus.section.recover_area_points_tensor(
                &input.owners,
                &input.right_area,
                &config.recovery,
            )?;
            let (bands, boundaries) = hydraulic(error::checked::<R, _>(
                client,
                "original vacuum geometry",
                |a| {
                    Ok((
                        geometry.gather(a, &input.owners)?,
                        geometry.boundary_areas(a)?.gather(a, &input.owners)?,
                    ))
                },
            ))?;
            let originals = Tensor::from_slice(&[0_i64, 1, 2], &[3], client.device())?;
            let work = WorkLedger::new(&config);
            let mut pair = hydraulic(PairedSources::new(
                &geometry,
                &bands,
                [&left, &right],
                &velocity,
                &boundaries,
                PotentialEvaluation {
                    config: &config,
                    panels: 1024,
                    work: &work,
                },
                &originals,
            ))?;
            let invariant = hydraulic(pair.source_invariant())?;
            assert_eq!(
                invariant.value.available.try_to_vec::<f64>()?,
                [1.0; 3],
                "{id}"
            );
            assert_eq!(
                components(&invariant.value.retained)?,
                [[expected, 0, 0]; 3],
                "{id}"
            );
            for interval in [
                &invariant.value.physical,
                &support::bounds::<R>(client, &invariant.value.retained)?,
            ] {
                assert_eq!(
                    endpoints(interval)?,
                    [[f64::from_bits(expected); 2]; 3],
                    "{id}"
                );
            }
            assert_eq!(invariant.vacuum.try_to_vec::<f64>()?, [masks[0]; 3], "{id}");
            assert_eq!(
                invariant.touching.try_to_vec::<f64>()?,
                [masks[1]; 3],
                "{id}"
            );
            assert_eq!(
                invariant.wet_candidate.try_to_vec::<f64>()?,
                [masks[2]; 3],
                "{id}"
            );
            assert_eq!(invariant.unresolved.try_to_vec::<f64>()?, [0.0; 3], "{id}");
            assert_eq!(invariant.both_dry.try_to_vec::<f64>()?, [0.0; 3], "{id}");
            for cache in &pair.cache {
                assert_eq!(cache.ready.try_to_vec::<f64>()?, [1.0; 3], "{id}");
                assert_eq!(
                    components(&cache.value.retained)?,
                    [[2.0_f64.to_bits(), 0, 0]; 3],
                    "{id}"
                );
            }
            assert_eq!(capture::snapshot(&velocity)?, before);
            assert_eq!(work.snapshot(), Default::default());
        }
        failed_fill::<R>(client, corpus)
    }

    fn failed_fill<R>(client: &R::Client, corpus: &Corpus<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let truth = capture::source_reference("scalar_three_bands_uniform_upper")?;
        assert!(
            truth
                .sides
                .iter()
                .all(|side| !side.physical_eligible && !side.within_first_band)
        );
        let rows = Tensor::from_slice(
            &[i64::try_from(truth.source_index)?; 3],
            &[3],
            client.device(),
        )?;
        let velocity = hydraulic(error::checked::<R, _>(
            client,
            "failed-fill original source binding",
            |a| corpus.velocity.select(a, &rows),
        ))?;
        let before = capture::snapshot(&velocity)?;
        let geometry = HydraulicGeometry::from_batch(&corpus.section);
        let input = velocity.input();
        let config = PrismaticRiemannConfig {
            gravity_m_per_s2: f64::from_bits(truth.gravity_bits),
            max_native_query_points: 0,
            ..PrismaticRiemannConfig::default()
        };
        let left = corpus.section.recover_area_points_tensor(
            &input.owners,
            &input.left_area,
            &config.recovery,
        )?;
        let right = corpus.section.recover_area_points_tensor(
            &input.owners,
            &input.right_area,
            &config.recovery,
        )?;
        let (bands, boundaries) = hydraulic(error::checked::<R, _>(
            client,
            "failed-fill original geometry",
            |a| {
                Ok((
                    geometry.gather(a, &input.owners)?,
                    geometry.boundary_areas(a)?.gather(a, &input.owners)?,
                ))
            },
        ))?;
        let originals = Tensor::from_slice(&[0_i64, 1, 2], &[3], client.device())?;
        let work = WorkLedger::new(&config);
        let mut pair = hydraulic(PairedSources::new(
            &geometry,
            &bands,
            [&left, &right],
            &velocity,
            &boundaries,
            PotentialEvaluation {
                config: &config,
                panels: 1024,
                work: &work,
            },
            &originals,
        ))?;
        for _ in 0..2 {
            assert!(matches!(
                pair.source_invariant(),
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                })
            ));
            for cache in &pair.cache {
                assert_eq!(cache.ready.try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(components(&cache.value.retained)?, [[0; 3]; 3]);
                assert_eq!(endpoints(&cache.value.physical)?, [[0.0; 2]; 3]);
            }
            assert_eq!(capture::snapshot(&velocity)?, before);
            assert_eq!(work.snapshot(), Default::default());
        }
        Ok(())
    }
}
