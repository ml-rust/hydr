use super::paired_source_cache::{self, SourcePotentialCache};
use super::potential_range::PotentialEvaluation;
use super::source_binding::{RootSourceIdentity, RootSourceWitness};
use super::source_terms::SourceTerms;
use super::types::SourceVelocity;
use super::wave_curve_types::CurveValue;
#[cfg(test)]
use crate::channel::flow::surveyed::error::checked;
use crate::channel::{
    flow::surveyed::error::HydraulicResult,
    section::{
        BoundaryAreas, GatheredBands, HydraulicGeometry, PhysicalEndpoint,
        PrismaticAreaPointRecovery, SupportedAreaQuery,
    },
};
#[cfg(test)]
use numr::ops::{BinaryOps, UtilityOps};
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;
use std::sync::Arc;

#[path = "paired_source_admission.rs"]
mod admission;

/// One immutable source graph and two fixed source-potential caches serve every target callback.
pub(super) struct PairedSources<'a, R: Runtime<DType = DType>> {
    pub(super) geometry: &'a HydraulicGeometry<R>,
    pub(super) bands: &'a GatheredBands<R>,
    pub(super) sources: [&'a PrismaticAreaPointRecovery<R>; 2],
    pub(super) velocity: &'a SourceVelocity<R>,
    pub(super) boundaries: &'a BoundaryAreas<R>,
    pub(super) evaluation: PotentialEvaluation<'a>,
    pub(super) original_rows: Tensor<R>,
    pub(super) cache: [SourcePotentialCache<R>; 2],
    pub(super) panel_cache: Vec<(usize, [SourcePotentialCache<R>; 2])>,
    /// Source geometry of each side's rows, filled once per row over every callback.
    pub(super) source_terms: [SourceTerms<R>; 2],
    identity: Arc<RootSourceIdentity>,
    construction_witness: RootSourceWitness,
}

#[cfg(test)]
pub(super) struct PairedRange<R: Runtime<DType = DType>> {
    pub(super) lower: CurveValue<R>,
    pub(super) upper: CurveValue<R>,
    pub(super) curve: CurveValue<R>,
}

pub(super) struct SourceInvariant<R: Runtime<DType = DType>> {
    pub(super) value: CurveValue<R>,
    pub(super) vacuum: Tensor<R>,
    pub(super) touching: Tensor<R>,
    pub(super) wet_candidate: Tensor<R>,
    #[cfg(test)]
    pub(super) unresolved: Tensor<R>,
    /// Exact captured dry-state identity, not an available wet velocity residual.
    pub(super) both_dry: Tensor<R>,
}

/// Scoped owned selections have no cache and borrow none of their own storage.
pub(super) struct SelectedRows<R: Runtime<DType = DType>> {
    pub(super) rows: Tensor<R>,
    pub(super) bands: GatheredBands<R>,
    pub(super) sources: [PrismaticAreaPointRecovery<R>; 2],
    pub(super) velocity: SourceVelocity<R>,
    pub(super) boundaries: BoundaryAreas<R>,
    pub(super) original_rows: Tensor<R>,
}

impl<R> PairedSources<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn root_witness(&self) -> HydraulicResult<RootSourceWitness> {
        self.check_construction_policy()?;
        Ok(self.construction_witness.clone())
    }

    pub(super) fn check_root_witness(&self, witness: &RootSourceWitness) -> HydraulicResult<()> {
        self.check_construction_policy()?;
        witness.check(&self.identity, &self.evaluation)
    }

    fn check_construction_policy(&self) -> HydraulicResult<()> {
        self.construction_witness
            .check(&self.identity, &self.evaluation)
    }
}

impl<R> PairedSources<'_, R>
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
    #[cfg(test)]
    pub(super) fn endpoint(
        &mut self,
        query: &SupportedAreaQuery<R>,
        endpoint: PhysicalEndpoint,
    ) -> HydraulicResult<CurveValue<R>> {
        let rows = self.geometry.client().arange(
            0.0,
            self.original_rows.numel() as f64,
            1.0,
            DType::I64,
        )?;
        self.endpoint_on_rows(&rows, query, endpoint)
    }
    /// Rows evaluate at the initial panel level.
    pub(super) fn endpoint_on_rows(
        &mut self,
        rows: &Tensor<R>,
        query: &SupportedAreaQuery<R>,
        endpoint: PhysicalEndpoint,
    ) -> HydraulicResult<CurveValue<R>> {
        let level = self.panel_level(0)?;
        super::paired_residual::endpoint_at_level(self, rows, query, endpoint, &level)
    }
    #[cfg(test)]
    pub(super) fn range(
        &mut self,
        query: &SupportedAreaQuery<R>,
    ) -> HydraulicResult<PairedRange<R>> {
        let rows = self.geometry.client().arange(
            0.0,
            self.original_rows.numel() as f64,
            1.0,
            DType::I64,
        )?;
        self.range_on_rows(&rows, query)
    }
    #[cfg(test)]
    pub(super) fn range_on_rows(
        &mut self,
        rows: &Tensor<R>,
        query: &SupportedAreaQuery<R>,
    ) -> HydraulicResult<PairedRange<R>> {
        let lower = self.endpoint_on_rows(rows, query, PhysicalEndpoint::Lower)?;
        let upper = self.endpoint_on_rows(rows, query, PhysicalEndpoint::Upper)?;
        let curve = checked(
            self.geometry.client(),
            "paired monotone residual range",
            |a| {
                let physical = a.bounds(lower.physical.lower(), upper.physical.upper())?;
                Ok(CurveValue {
                    retained: a.expansion_from_interval(&physical)?,
                    physical,
                    available: a.client().mul(&lower.available, &upper.available)?,
                })
            },
        )?;
        Ok(PairedRange {
            lower,
            upper,
            curve,
        })
    }
    #[cfg(test)]
    pub(super) fn source_invariant(&mut self) -> HydraulicResult<SourceInvariant<R>> {
        super::paired_source_invariant::evaluate(self)
    }
    pub(super) fn source_invariant_on_rows(
        &mut self,
        rows: &Tensor<R>,
    ) -> HydraulicResult<SourceInvariant<R>> {
        super::paired_source_invariant::evaluate_on_rows(self, rows)
    }
    pub(super) fn source_values(
        &mut self,
        rows: [&Tensor<R>; 2],
    ) -> HydraulicResult<[CurveValue<R>; 2]> {
        self.check_construction_policy()?;
        let [left, right] = &mut self.cache;
        Ok([
            paired_source_cache::ensure(
                left,
                self.geometry,
                self.bands,
                self.sources[0],
                self.boundaries,
                &self.evaluation,
                rows[0],
            )?,
            paired_source_cache::ensure(
                right,
                self.geometry,
                self.bands,
                self.sources[1],
                self.boundaries,
                &self.evaluation,
                rows[1],
            )?,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::{PrismaticRiemannConfig, error, work::WorkLedger};
    use crate::channel::flow::surveyed_points::{
        paired_residual_corpus_fixture as capture, paired_residual_fixtures as frozen,
        paired_source_invariant::tests as invariant,
        source_fixture_support::{
            self as support, Client, Result, bounds, components, endpoints, hydraulic,
        },
        source_potential_fixtures as source_references,
    };
    use crate::channel::section::{AreaAnchor, AreaDirection, AreaQuery};
    use cache::cache_controls;
    use std::collections::BTreeMap;

    mod cache {
        use super::*;

        #[derive(Debug, PartialEq)]
        struct CacheSnapshot {
            ready: Vec<u64>,
            retained: Vec<[u64; 3]>,
            physical: Vec<[u64; 2]>,
        }

        fn cache_snapshot<R: Runtime<DType = DType>>(
            pair: &PairedSources<'_, R>,
        ) -> Result<Vec<CacheSnapshot>> {
            pair.cache
                .iter()
                .map(|cache| {
                    Ok(CacheSnapshot {
                        ready: cache
                            .ready
                            .try_to_vec::<f64>()?
                            .into_iter()
                            .map(f64::to_bits)
                            .collect(),
                        retained: components(&cache.value.retained)?,
                        physical: support::endpoints(&cache.value.physical)?
                            .into_iter()
                            .map(|value| value.map(f64::to_bits))
                            .collect(),
                    })
                })
                .collect()
        }

        pub(super) fn cache_controls<R>(client: &R::Client, corpus: &capture::Corpus<R>) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
        {
            let truth: &source_references::Case = source_references::CASES
                .iter()
                .find(|case| case.id == "scalar_three_bands_uniform_upper")
                .ok_or("Missing independent multiband source potential")?;
            assert!(
                truth
                    .sides
                    .iter()
                    .all(
                        |side: &source_references::SideReference| !side.physical_eligible
                            && !side.within_first_band
                            && !side.dry
                    )
            );
            let source_rows = Tensor::from_slice(
                &[i64::try_from(truth.source_index)?; 3],
                &[3],
                client.device(),
            )?;
            let velocity = hydraulic(error::checked::<R, _>(
                client,
                "cached multiband source binding",
                |a| corpus.velocity.select(a, &source_rows),
            ))?;
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
            let (bands, boundaries, dry, equal) = hydraulic(error::checked::<R, _>(
                client,
                "cached multiband source graph",
                |a| {
                    let bands = geometry.gather(a, &input.owners)?;
                    let boundaries = geometry.boundary_areas(a)?.gather(a, &input.owners)?;
                    let zero = Tensor::zeros(&[3], a.dtype(), client.device())?;
                    let scale = a.expansion_point(&zero)?;
                    let parameter = a.point(&zero)?;
                    let dry = AreaQuery::band_affine(
                        a,
                        &bands,
                        AreaAnchor::Zero,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    )?
                    .prove_support(a, &bands)?;
                    let equal = AreaQuery::source_affine(
                        a,
                        &bands,
                        &left,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    )?
                    .prove_support(a, &bands)?;
                    Ok((bands, boundaries, dry, equal))
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
            assert_eq!(work.snapshot(), Default::default());
            for cache in &pair.cache {
                assert_eq!(cache.ready.try_to_vec::<f64>()?, [0.0; 3]);
            }
            let storage = Tensor::from_slice(&[99_i64, 2, 99, 0, 99, 2], &[3, 2], client.device())?;
            let rows = storage.narrow(1, 1, 1)?.squeeze(Some(1));
            let selected = hydraulic(error::checked::<R, _>(
                client,
                "cached compact dry query",
                |a| dry.select(a, &rows),
            ))?;
            let value =
                hydraulic(pair.endpoint_on_rows(&rows, &selected, PhysicalEndpoint::Lower))?;
            assert_eq!(value.available.try_to_vec::<f64>()?, [1.0; 3]);
            assert!(
                value
                    .physical
                    .upper()
                    .try_to_vec::<f64>()?
                    .iter()
                    .all(|&upper| upper < 0.0)
            );
            for (side, cache) in pair.cache.iter().enumerate() {
                let phi: &source_references::Reference = &truth.sides[side].phi;
                assert_eq!(cache.ready.try_to_vec::<f64>()?, [1.0, 0.0, 1.0]);
                for interval in [
                    &cache.value.physical,
                    &support::bounds::<R>(client, &cache.value.retained)?,
                ] {
                    let values = support::endpoints(interval)?;
                    for row in [0, 2] {
                        capture::reference(values[row], phi.bounds, phi.native_tolerance, truth.id);
                    }
                }
            }
            let partial = cache_snapshot(&pair)?;
            let partial_work = work.snapshot();
            assert!(
                partial_work.simpson_value_points > 0 && partial_work.simpson_derivative_points > 0
            );
            let _ = hydraulic(pair.endpoint_on_rows(&rows, &selected, PhysicalEndpoint::Upper))?;
            assert_eq!(cache_snapshot(&pair)?, partial);
            assert_eq!(work.snapshot(), partial_work);
            let _ = hydraulic(pair.source_invariant())?;
            for (side, cache) in pair.cache.iter().enumerate() {
                assert_eq!(cache.ready.try_to_vec::<f64>()?, [1.0; 3]);
                for interval in [
                    &cache.value.physical,
                    &support::bounds::<R>(client, &cache.value.retained)?,
                ] {
                    for value in support::endpoints(interval)? {
                        capture::reference(
                            value,
                            truth.sides[side].phi.bounds,
                            truth.sides[side].phi.native_tolerance,
                            truth.id,
                        );
                    }
                }
            }
            let complete = cache_snapshot(&pair)?;
            let completed_work = work.snapshot();
            assert!(completed_work.simpson_value_points > partial_work.simpson_value_points);
            let _ = hydraulic(pair.source_invariant())?;
            let identity = hydraulic(pair.endpoint(&equal, PhysicalEndpoint::Lower))?;
            assert_eq!(components(&identity.retained)?, [[0; 3]; 3]);
            assert_eq!(cache_snapshot(&pair)?, complete);
            assert_eq!(work.snapshot(), completed_work);
            let wrong_rows = Tensor::from_slice(&[1_i64, 0, 2], &[3], client.device())?;
            let wrong_originals = hydraulic(error::checked::<R, _>(
                client,
                "misordered paired original rows",
                |a| dry.select(a, &wrong_rows),
            ))?;
            assert!(
                pair.endpoint(&wrong_originals, PhysicalEndpoint::Lower)
                    .is_err()
            );
            assert_eq!(cache_snapshot(&pair)?, complete);
            assert_eq!(work.snapshot(), completed_work);
            let foreign_section = support::batch::<R>(client)?;
            let foreign_geometry = HydraulicGeometry::from_batch(&foreign_section);
            let foreign_query = hydraulic(error::checked::<R, _>(
                client,
                "foreign paired query graph",
                |a| {
                    let foreign_bands = foreign_geometry.gather(a, &input.owners)?;
                    let zero = Tensor::zeros(&[3], a.dtype(), client.device())?;
                    let scale = a.expansion_point(&zero)?;
                    let parameter = a.point(&zero)?;
                    AreaQuery::band_affine(
                        a,
                        &foreign_bands,
                        AreaAnchor::Zero,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    )?
                    .prove_support(a, &foreign_bands)
                },
            ))?;
            let mut untouched = hydraulic(PairedSources::new(
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
            let unfilled = cache_snapshot(&untouched)?;
            assert!(
                untouched
                    .endpoint(&wrong_originals, PhysicalEndpoint::Lower)
                    .is_err()
            );
            assert!(
                untouched
                    .endpoint(&foreign_query, PhysicalEndpoint::Lower)
                    .is_err()
            );
            assert!(
                pair.endpoint(&foreign_query, PhysicalEndpoint::Lower)
                    .is_err()
            );
            let other_owners = Tensor::from_slice(&[1_i64; 3], &[3], client.device())?;
            assert_ne!(input.owners.try_to_vec::<i64>()?, [1_i64; 3]);
            let other_query = hydraulic(error::checked::<R, _>(
                client,
                "different paired query owners",
                |a| {
                    let other_bands = geometry.gather(a, &other_owners)?;
                    let zero = Tensor::zeros(&[3], a.dtype(), client.device())?;
                    let scale = a.expansion_point(&zero)?;
                    let parameter = a.point(&zero)?;
                    AreaQuery::band_affine(
                        a,
                        &other_bands,
                        AreaAnchor::Zero,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    )?
                    .prove_support(a, &other_bands)
                },
            ))?;
            assert!(
                untouched
                    .endpoint(&other_query, PhysicalEndpoint::Lower)
                    .is_err()
            );
            assert_eq!(cache_snapshot(&untouched)?, unfilled);
            assert!(
                pair.endpoint(&other_query, PhysicalEndpoint::Lower)
                    .is_err()
            );
            assert_eq!(cache_snapshot(&pair)?, complete);
            assert_eq!(work.snapshot(), completed_work);
            assert_eq!(work.snapshot().root_solves, 0);
            assert_eq!(work.snapshot().bracket_solves, 0);
            assert_eq!(work.snapshot().refinements, 0);
            Ok(())
        }
    }

    fn check<R>(
        client: &R::Client,
        value: &CurveValue<R>,
        reference: &frozen::Reference,
        label: &str,
    ) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        assert!(
            value
                .available
                .try_to_vec::<f64>()?
                .iter()
                .all(|&mask| mask == 1.0),
            "{label}"
        );
        for interval in [&value.physical, &bounds::<R>(client, &value.retained)?] {
            for actual in endpoints(interval)? {
                capture::reference(actual, reference.bounds, reference.allowance, label);
                match reference.sign {
                    frozen::Sign::Negative => assert!(actual[1] < 0.0, "{label}: {actual:?}"),
                    frozen::Sign::Positive => assert!(actual[0] > 0.0, "{label}: {actual:?}"),
                    frozen::Sign::ContainsZero => {
                        assert!(actual[0] <= 0.0 && actual[1] >= 0.0, "{label}: {actual:?}")
                    }
                }
            }
        }
        Ok(())
    }

    fn check_expected<R>(
        client: &R::Client,
        value: &CurveValue<R>,
        expected: &frozen::Expected,
        upper: bool,
        label: &str,
    ) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        match expected {
            frozen::Expected::Unavailable => {
                assert!(
                    value
                        .available
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|&mask| mask == 0.0),
                    "{label}"
                );
            }
            frozen::Expected::ExactZero => {
                assert!(
                    value
                        .available
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|&mask| mask == 1.0),
                    "{label}"
                );
                for point in components(&value.retained)? {
                    assert_eq!(point, [0; 3], "{label}");
                }
                for physical in endpoints(&value.physical)? {
                    assert_eq!(physical, [0.0; 2], "{label}");
                }
            }
            frozen::Expected::Point(reference) => check::<R>(client, value, reference, label)?,
            frozen::Expected::Range { lower, upper: high } => {
                check::<R>(client, value, if upper { high } else { lower }, label)?
            }
        }
        Ok(())
    }

    fn chart<R>(
        client: &R::Client,
        bands: &crate::channel::section::GatheredBands<R>,
        source: &crate::channel::section::PrismaticAreaPointRecovery<R>,
        request: &frozen::Request,
    ) -> error::HydraulicResult<crate::channel::section::SupportedAreaQuery<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        error::checked::<R, _>(client, "paired frozen residual chart", |a| {
            let count = bands.section_owner.numel();
            let zero = Tensor::zeros(&[count], a.dtype(), client.device())?;
            let scale = a.expansion_bounds(
                &Tensor::from_slice(
                    &vec![f64::from_bits(request.query.scale[0]); count],
                    &[count],
                    client.device(),
                )?,
                &Tensor::from_slice(
                    &vec![f64::from_bits(request.query.scale[1]); count],
                    &[count],
                    client.device(),
                )?,
                &zero,
            )?;
            let parameter = a.bounds(
                &Tensor::from_slice(
                    &vec![f64::from_bits(request.query.parameter[0]); count],
                    &[count],
                    client.device(),
                )?,
                &Tensor::from_slice(
                    &vec![f64::from_bits(request.query.parameter[1]); count],
                    &[count],
                    client.device(),
                )?,
            )?;
            let direction = if request.query.add {
                AreaDirection::Add
            } else {
                AreaDirection::Subtract
            };
            let query = match request.query.anchor {
                frozen::Anchor::LeftSource => {
                    AreaQuery::source_affine(a, bands, source, direction, &scale, &parameter)?
                }
                frozen::Anchor::Zero => AreaQuery::band_affine(
                    a,
                    bands,
                    AreaAnchor::Zero,
                    direction,
                    &scale,
                    &parameter,
                )?,
            };
            query.prove_support(a, bands)
        })
    }

    fn matrix<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        assert_eq!(frozen::REQUESTS.len(), 13);
        for partition in [
            frozen::Partition::Local,
            frozen::Partition::Below,
            frozen::Partition::Exact,
            frozen::Partition::Above,
            frozen::Partition::Crossing,
        ] {
            assert!(
                frozen::REQUESTS
                    .iter()
                    .any(|request| request.partition == partition)
            );
        }
        let source_cases = support::cases();
        let mut groups = BTreeMap::new();
        for request in frozen::REQUESTS {
            let source_index = source_cases
                .iter()
                .position(|case| case.id == request.source_case_id)
                .ok_or("Paired request source is absent from original70 corpus")?;
            groups
                .entry((source_index, request.gravity_bits))
                .or_insert_with(Vec::new)
                .push(request);
        }
        let mut request_count = 0;
        for strided in [false, true] {
            let corpus = capture::corpus::<R>(client, strided)?;
            let corpus_before = capture::snapshot(&corpus.velocity)?;
            assert_eq!(
                corpus.owners.contiguous()?.try_to_vec::<i64>()?,
                corpus.velocity.input().owners.try_to_vec::<i64>()?
            );
            // The bound source context owns conserved operands independently from the original caller tensors.
            for value in corpus
                .inputs
                .iter()
                .flat_map(|value| [value.high(), value.low(), value.abs_error()])
            {
                client.copy_into(
                    value,
                    &Tensor::zeros(value.shape(), value.dtype(), client.device())?,
                )?;
            }
            cache_controls::<R>(client, &corpus)?;
            invariant::controls::<R>(client, &corpus)?;
            assert_eq!(capture::snapshot(&corpus.velocity)?, corpus_before);
            for (&(source_index, gravity_bits), requests) in &groups {
                let rows =
                    Tensor::from_slice(&[i64::try_from(source_index)?; 3], &[3], client.device())?;
                let velocity = hydraulic(error::checked::<R, _>(
                    client,
                    "paired original source binding",
                    |a| corpus.velocity.select(a, &rows),
                ))?;
                let source_before = capture::snapshot(&velocity)?;
                let geometry = HydraulicGeometry::from_batch(&corpus.section);
                let input = velocity.input();
                let config = PrismaticRiemannConfig {
                    gravity_m_per_s2: f64::from_bits(gravity_bits),
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
                    "paired original geometry",
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
                if requests
                    .iter()
                    .any(|request| request.id == "film_dry_target")
                {
                    let invariant = hydraulic(pair.source_invariant())?;
                    assert_eq!(invariant.value.available.try_to_vec::<f64>()?, [1.0; 3]);
                    assert_eq!(
                        components(&invariant.value.retained)?,
                        [[
                            frozen::FILM_SOURCE_INVARIANT[0],
                            frozen::FILM_SOURCE_INVARIANT[1],
                            0
                        ]; 3]
                    );
                    assert_eq!(invariant.vacuum.try_to_vec::<f64>()?, [0.0; 3]);
                    assert_eq!(invariant.touching.try_to_vec::<f64>()?, [0.0; 3]);
                    assert_eq!(invariant.wet_candidate.try_to_vec::<f64>()?, [1.0; 3]);
                    assert_eq!(invariant.unresolved.try_to_vec::<f64>()?, [0.0; 3]);
                    assert_eq!(invariant.both_dry.try_to_vec::<f64>()?, [0.0; 3]);
                }
                for request in requests {
                    request_count += 1;
                    let query = hydraulic(chart::<R>(client, &bands, &left, request))?;
                    for (endpoint, upper) in [
                        (PhysicalEndpoint::Lower, false),
                        (PhysicalEndpoint::Upper, true),
                    ] {
                        let value = hydraulic(pair.endpoint(&query, endpoint))?;
                        check_expected::<R>(client, &value, &request.expected, upper, request.id)?;
                    }
                    let range = hydraulic(pair.range(&query))?;
                    check_expected::<R>(
                        client,
                        &range.lower,
                        &request.expected,
                        false,
                        request.id,
                    )?;
                    check_expected::<R>(client, &range.upper, &request.expected, true, request.id)?;
                    if let frozen::Expected::Range { lower, upper } = &request.expected {
                        for interval in [
                            &range.curve.physical,
                            &bounds::<R>(client, &range.curve.retained)?,
                        ] {
                            for actual in endpoints(interval)? {
                                let low = lower.bounds.map(f64::from_bits);
                                let high = upper.bounds.map(f64::from_bits);
                                assert!(
                                    actual[0] <= low[0] && high[1] <= actual[1],
                                    "{}: {actual:?}",
                                    request.id
                                );
                                assert!(
                                    low[1] - actual[0] <= f64::from_bits(lower.allowance),
                                    "{}: lower range gate",
                                    request.id
                                );
                                assert!(
                                    actual[1] - high[0] <= f64::from_bits(upper.allowance),
                                    "{}: upper range gate",
                                    request.id
                                );
                            }
                        }
                    } else {
                        check_expected::<R>(
                            client,
                            &range.curve,
                            &request.expected,
                            false,
                            request.id,
                        )?;
                    }
                    let row_storage = Tensor::from_slice(
                        &[99_i64, 2, 99, 0, 99, 2, 99, 1],
                        &[4, 2],
                        client.device(),
                    )?;
                    let rows = row_storage.narrow(1, 1, 1)?.squeeze(Some(1));
                    let selected = hydraulic(error::checked::<R, _>(
                        client,
                        "paired callback query selection",
                        |a| query.select(a, &rows),
                    ))?;
                    assert_eq!(
                        selected.query.original_rows.try_to_vec::<i64>()?,
                        [2, 0, 2, 1]
                    );
                    let selected_value = hydraulic(pair.endpoint_on_rows(
                        &rows,
                        &selected,
                        PhysicalEndpoint::Lower,
                    ))?;
                    check_expected::<R>(
                        client,
                        &selected_value,
                        &request.expected,
                        false,
                        request.id,
                    )?;
                    let selected_range = hydraulic(pair.range_on_rows(&rows, &selected))?;
                    check_expected::<R>(
                        client,
                        &selected_range.lower,
                        &request.expected,
                        false,
                        request.id,
                    )?;
                    check_expected::<R>(
                        client,
                        &selected_range.upper,
                        &request.expected,
                        true,
                        request.id,
                    )?;
                    assert_eq!(capture::snapshot(&velocity)?, source_before);
                }
                assert_eq!(work.snapshot().root_solves, 0);
                assert_eq!(work.snapshot().bracket_solves, 0);
                assert_eq!(work.snapshot().refinements, 0);
            }
            assert_eq!(capture::snapshot(&corpus.velocity)?, corpus_before);
        }
        assert_eq!(request_count, 26);
        Ok(())
    }

    #[test]
    fn configured_cpu_paired_residuals_preserve_frozen_ranges_and_source_caches() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        matrix::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_paired_residuals_preserve_frozen_ranges_and_source_caches() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
