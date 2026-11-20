mod value;

use super::source_terms::SourceRows;
use super::target_states::TargetRows;
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig,
        error::{HydraulicError, HydraulicResult, checked},
        work::WorkLedger,
    },
    section::{
        AreaDirection, AreaQuery, BoundaryAreas, GatheredBands, HydraulicGeometry,
        PrismaticAreaPointRecovery, SupportedAreaQuery, source_difference,
    },
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) use value::{PotentialValue, from_interval, scatter, select_expansion, zero_value};

pub(super) struct PotentialEvaluation<'a> {
    pub(super) config: &'a PrismaticRiemannConfig,
    pub(super) panels: usize,
    pub(super) work: &'a WorkLedger,
}

/// Construction checks one model/owner graph. Leaves perform only their requested integral.
pub(super) struct PotentialContext<'a, R: Runtime<DType = DType>> {
    pub(super) geometry: &'a HydraulicGeometry<R>,
    pub(super) bands: &'a GatheredBands<R>,
    pub(super) source: &'a PrismaticAreaPointRecovery<R>,
    pub(super) query: &'a SupportedAreaQuery<R>,
    pub(super) boundaries: &'a BoundaryAreas<R>,
    pub(super) evaluation: PotentialEvaluation<'a>,
    /// Parent rows of a fixed paired side whose source terms persist over target callbacks.
    pub(super) source_rows: Option<SourceRows<'a, R>>,
    /// Rows of a kept singleton target. Present only when `query` selects that target's rows.
    pub(super) target_rows: Option<TargetRows<'a, R>>,
}

impl<'a, R> PotentialContext<'a, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn new(
        geometry: &'a HydraulicGeometry<R>,
        bands: &'a GatheredBands<R>,
        source: &'a PrismaticAreaPointRecovery<R>,
        query: &'a SupportedAreaQuery<R>,
        boundaries: &'a BoundaryAreas<R>,
        evaluation: PotentialEvaluation<'a>,
    ) -> HydraulicResult<Self> {
        admit_evaluation(geometry, bands, &evaluation)?;
        checked(geometry.client(), "point potential authority", |a| {
            source_difference(a, bands, query, source, boundaries)?;
            Ok(())
        })?;
        Ok(Self {
            geometry,
            bands,
            source,
            query,
            boundaries,
            evaluation,
            source_rows: None,
            target_rows: None,
        })
    }

    /// This context with its rows bound to the parent rows of a fixed paired side.
    pub(super) fn with_source_rows(mut self, rows: Option<SourceRows<'a, R>>) -> Self {
        self.source_rows = rows;
        self
    }

    /// This context with its rows bound to the rows of a kept target.
    /// `query` must be the row selection `rows.rows` of that target's query.
    pub(super) fn with_target_rows(mut self, rows: Option<TargetRows<'a, R>>) -> Self {
        self.target_rows = rows;
        self
    }

    pub(super) fn source_query(&self) -> HydraulicResult<SupportedAreaQuery<R>> {
        source_query(
            self.geometry,
            self.bands,
            self.source,
            &self.query.query.original_rows,
        )
    }
}

pub(super) fn admit_evaluation<R>(
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    evaluation: &PotentialEvaluation<'_>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    evaluation.config.check()?;
    if !evaluation.panels.is_power_of_two()
        || evaluation.panels > evaluation.config.max_simpson_panels
    {
        return Err(HydraulicError::Geometry(
            crate::channel::section::CrossSectionError::InvalidInput {
                section: None,
                context: format!(
                    "potential panels={} exceed the admitted positive power-of-two ceiling={}. Supply admitted panels",
                    evaluation.panels, evaluation.config.max_simpson_panels,
                ),
            },
        ));
    }
    if !geometry.same_model(bands) {
        return Err(HydraulicError::Geometry(crate::channel::section::CrossSectionError::InvalidInput { section: None, context:
                "potential geometry and bands belong to different models. Supply one checked graph".into() }));
    }
    Ok(())
}

/// Original rows already belong to the checked source graph and remain diagnostic identities.
pub(super) fn source_query<R>(
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    source: &PrismaticAreaPointRecovery<R>,
    original_rows: &Tensor<R>,
) -> HydraulicResult<SupportedAreaQuery<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(geometry.client(), "point source potential chart", |a| {
        let zero = Tensor::zeros(source.owners().shape(), a.dtype(), a.client().device())?;
        let scale = a.expansion_point(&zero)?;
        let parameter = a.point(&zero)?;
        let mut query =
            AreaQuery::source_affine(a, bands, source, AreaDirection::Add, &scale, &parameter)?;
        let rows = Tensor::empty(original_rows.shape(), DType::I64, a.client().device())?;
        a.client().copy_into(&rows, original_rows)?;
        query.original_rows = rows;
        query.prove_support(a, bands)
    })
}

#[cfg(test)]
mod tests {
    use super::super::corpus_coverage::Coverage;
    use super::super::potential_absolute::{endpoint_potential, source_potential};
    use super::super::potential_increment::{endpoint_increment, film};
    use super::super::potential_increment_fixtures as frozen;
    use super::super::potential_range_fixture::{
        check, check_value, config, section, target_check,
    };
    use super::super::source_fixture_support::{
        self as support, Client, Result, components, endpoints, expansion, hydraulic, zero,
    };
    use super::*;
    use crate::channel::{
        flow::surveyed::{
            PrismaticRiemannRejection,
            equation_ops::{gravity_depth_root, scalar},
            error,
        },
        section::{AreaAnchor, PhysicalEndpoint, PrismaticRecoveryConfig},
    };
    use solvr::integrate::CertifiedSimpsonAlgorithms;

    fn guarded_prefix<R>(
        client: &R::Client,
        context: &PotentialContext<'_, R>,
        endpoint: PhysicalEndpoint,
        reference: frozen::Reference,
    ) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let query = hydraulic(error::checked::<R, _>(
            client,
            "zero-crossing analytic admission control",
            |a| {
                let query = context.query.endpoint(a, context.bands, endpoint)?;
                let raw = query.retained_endpoint_area(a, context.bands, context.boundaries)?;
                let state = context
                    .geometry
                    .recover_query(a, context.bands, &query)?
                    .state;
                let physical = a.expansion_interval(&raw)?;
                let zero = Tensor::zeros(raw.high().shape(), DType::F64, client.device())?;
                // Widen only the certificate. The original frozen physical query and recovered geometry remain unchanged.
                let radius = client.mul_scalar(physical.upper(), 2.0)?;
                let widened = a.expansion_bounds(&zero, &zero, &radius)?;
                let projected = a.expansion_interval(&widened)?;
                assert!(
                    projected
                        .lower()
                        .try_to_vec::<f64>()?
                        .into_iter()
                        .all(|x| x < 0.0)
                );
                let analytic = super::super::potential_absolute::admitted_analytic(
                    a,
                    context.bands,
                    &widened,
                    &state.depth,
                    context.evaluation.config.gravity_m_per_s2,
                )?;
                assert_eq!(analytic.eligible.try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(analytic.rectangular.try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(analytic.pointed.try_to_vec::<f64>()?, [0.0; 3]);
                Ok(query)
            },
        ))?;
        let prefix = hydraulic(super::super::potential_absolute::prefix_potential(
            context,
            context.bands,
            &query,
            context.boundaries,
        ))?;
        check_value::<R>(
            client,
            &prefix,
            reference,
            "zero-crossing certificate retains the physical dry prefix",
        )
    }

    fn bindings<R>(client: &R::Client, strided: bool) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let section = section::<R>(client, frozen::query_profiles::PROFILES)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let config = config(frozen::query_profiles::GRAVITY_BITS);
        let table = hydraulic(error::checked::<R, _>(
            client,
            "frozen potential boundary table",
            |a| geometry.boundary_areas(a),
        ))?;
        // Strided storage changes only the input layout, so one case per profile reaches it.
        let coverage = if strided {
            Coverage::Sample
        } else {
            Coverage::Full
        };
        let cases = coverage.select(frozen::query_profiles::CASES, |case| {
            frozen::query_profiles::QUERIES[case.query_index].profile_index
        });
        for case in cases {
            let case: &frozen::query_profiles::Case = case;
            let chart: &frozen::query_profiles::Chart =
                &frozen::query_profiles::QUERIES[case.query_index];
            assert_eq!(chart.id, case.query_id);
            // Repeated owners exercise independent owned rows. Input components retain their original strided storage.
            let owners = Tensor::from_slice(
                &[i64::try_from(chart.profile_index)?; 3],
                &[3],
                client.device(),
            )?;
            let source_area = expansion::<R>(
                client,
                &[[case.source_components[0], case.source_components[1], 0]; 3],
                strided,
            )?;
            let before = components(&source_area)?;
            let source =
                section.recover_area_points_tensor(&owners, &source_area, &config.recovery)?;
            let anchor_area = expansion::<R>(
                client,
                &[[
                    chart.anchor_source_components[0],
                    chart.anchor_source_components[1],
                    0,
                ]; 3],
                strided,
            )?;
            let anchor_source =
                section.recover_area_points_tensor(&owners, &anchor_area, &config.recovery)?;
            let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
                client,
                "frozen potential binding",
                |a| {
                    let bands = geometry.gather(a, &owners)?;
                    let boundaries = table.gather(a, &owners)?;
                    let scale = a.expansion_bounds(
                        &Tensor::from_slice(
                            &[f64::from_bits(chart.scale_components[0]); 3],
                            &[3],
                            client.device(),
                        )?,
                        &Tensor::from_slice(
                            &[f64::from_bits(chart.scale_components[1]); 3],
                            &[3],
                            client.device(),
                        )?,
                        &Tensor::zeros(&[3], DType::F64, client.device())?,
                    )?;
                    let parameter = a.bounds(
                        &Tensor::from_slice(
                            &[f64::from_bits(chart.parameter_bounds[0]); 3],
                            &[3],
                            client.device(),
                        )?,
                        &Tensor::from_slice(
                            &[f64::from_bits(chart.parameter_bounds[1]); 3],
                            &[3],
                            client.device(),
                        )?,
                    )?;
                    let direction = if chart.add {
                        AreaDirection::Add
                    } else {
                        AreaDirection::Subtract
                    };
                    let query = if chart.anchor == frozen::query_profiles::AnchorKind::Source {
                AreaQuery::source_affine(a, &bands, &anchor_source, direction, &scale, &parameter)?
            } else {
                let indices = Tensor::from_slice(&[chart.band; 3], &[3], client.device())?;
                let anchor = match chart.anchor {
                    frozen::query_profiles::AnchorKind::Zero => AreaAnchor::Zero,
                    frozen::query_profiles::AnchorKind::BandStart => AreaAnchor::BandStart(indices),
                    frozen::query_profiles::AnchorKind::BandEnd => AreaAnchor::BandEnd(indices),
                    frozen::query_profiles::AnchorKind::Cap => AreaAnchor::Cap,
                    frozen::query_profiles::AnchorKind::Source => return Err(HydraulicError::Numerical {
                        context: "source chart reached a geometric-anchor branch. Check fixture dispatch".into() }),
                };
                AreaQuery::band_affine(a, &bands, anchor, direction, &scale, &parameter)?
            }.prove_support(a, &bands)?;
                    Ok((bands, boundaries, query))
                },
            ))?;
            let work = WorkLedger::new(&config);
            let context = hydraulic(PotentialContext::new(
                &geometry,
                &bands,
                &source,
                &query,
                &boundaries,
                PotentialEvaluation {
                    config: &config,
                    panels: 1024,
                    work: &work,
                },
            ))?;
            let source_value = hydraulic(source_potential(&context))?;
            check_value::<R>(client, &source_value, case.source_potential, case.id)?;
            let source_sound =
                hydraulic(error::checked::<R, _>(client, "frozen source sound", |a| {
                    let g = scalar(a, config.gravity_m_per_s2)?;
                    gravity_depth_root(a, &g, source.hydraulic_depth_m(), None)
                }))?;
            for actual in endpoints(&source_sound)? {
                check(actual, case.source_sound, case.id);
            }
            let state = hydraulic(error::checked::<R, _>(
                client,
                "frozen potential range geometry",
                |a| geometry.recover_query(a, &bands, &query),
            ))?
            .state;
            assert_eq!(
                state.wet_possible.try_to_vec::<f64>()?,
                [if case.wet_possible { 1.0 } else { 0.0 }; 3]
            );
            assert_eq!(
                state.dry_possible.try_to_vec::<f64>()?,
                [if case.dry_possible { 1.0 } else { 0.0 }; 3]
            );
            for (endpoint, index) in [
                (PhysicalEndpoint::Lower, case.minimum_endpoint),
                (PhysicalEndpoint::Upper, case.maximum_endpoint),
            ] {
                let value = hydraulic(endpoint_potential(&context, endpoint))?;
                let increment = hydraulic(endpoint_increment(&context, endpoint))?;
                target_check::<R>(client, &value, &increment, &case.endpoints[index], case.id)?;
                if case.query_index == 14 && matches!(endpoint, PhysicalEndpoint::Lower) {
                    guarded_prefix::<R>(
                        client,
                        &context,
                        endpoint,
                        case.endpoints[index].potential,
                    )?;
                }
                let sound = hydraulic(error::checked::<R, _>(
                    client,
                    "frozen endpoint sound",
                    |a| {
                        let target = query.endpoint(a, &bands, endpoint)?;
                        let state = geometry.recover_query(a, &bands, &target)?.state;
                        let g = scalar(a, config.gravity_m_per_s2)?;
                        gravity_depth_root(a, &g, &state.hydraulic_depth, None)
                    },
                ))?;
                for actual in endpoints(&sound)? {
                    check(actual, case.endpoints[index].sound, case.id);
                }
                if let Some(analytic) = &value.analytic {
                    let eligible = analytic.eligible.try_to_vec::<f64>()?;
                    let projected = support::bounds::<R>(client, &analytic.sound)?;
                    for (row, actual) in endpoints(&projected)?.into_iter().enumerate() {
                        if eligible[row] != 0.0 {
                            check(actual, case.endpoints[index].sound, case.id);
                        }
                    }
                }
            }
            assert_eq!(
                before,
                components(&source_area)?,
                "{}: input mutation",
                case.id
            );
            assert_eq!(work.snapshot().root_solves, 0);
            assert_eq!(work.snapshot().bracket_solves, 0);
            assert_eq!(work.snapshot().refinements, 0);
        }
        Ok(())
    }

    fn lazy_controls<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let section = section::<R>(client, frozen::query_profiles::PROFILES)?;
        let foreign = super::super::potential_range_fixture::section::<R>(
            client,
            frozen::query_profiles::PROFILES,
        )?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let other_geometry = HydraulicGeometry::from_batch(&foreign);
        let owners = Tensor::from_slice(&[3_i64], &[1], client.device())?;
        let area = expansion::<R>(client, &[[1.0_f64.to_bits(), 0, 0]], true)?;
        let source = section.recover_area_points_tensor(
            &owners,
            &area,
            &PrismaticRecoveryConfig::default(),
        )?;
        let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
            client,
            "equal potential control",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                let scale = a.expansion_point(&zero)?;
                let parameter = a.point(&zero)?;
                let query = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                Ok((bands, boundaries, query))
            },
        ))?;
        let mut config = config(frozen::query_profiles::GRAVITY_BITS);
        config.max_native_query_points = 0;
        let work = WorkLedger::new(&config);
        let context = hydraulic(PotentialContext::new(
            &geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config: &config,
                panels: 4,
                work: &work,
            },
        ))?;
        let increment = hydraulic(endpoint_increment(&context, PhysicalEndpoint::Lower))?;
        zero(
            components(&increment.retained)?[0],
            "equal nonanalytic potential",
        );
        assert_eq!(endpoints(&increment.physical)?, [[0.0; 2]]);
        assert_eq!(work.snapshot(), Default::default());
        assert!(matches!(
            source_potential(&context),
            Err(HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::WorkLimit,
                ..
            })
        ));
        assert_eq!(work.snapshot(), Default::default());
        assert!(matches!(
            PotentialContext::new(
                &other_geometry,
                &bands,
                &source,
                &query,
                &boundaries,
                PotentialEvaluation {
                    config: &config,
                    panels: 4,
                    work: &work
                }
            ),
            Err(HydraulicError::Geometry(_))
        ));
        Ok(())
    }

    fn matrix<R>(client: R::Client, strided: bool) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        bindings::<R>(&client, strided)?;
        if strided {
            return Ok(());
        }
        lazy_controls::<R>(&client)?;
        film::run::<R>(&client)
    }

    fn configured_cpu(strided: bool) -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        let retained = client.clone();
        matrix::<CpuRuntime>(client, strided)?;
        assert_eq!(retained.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[test]
    fn configured_cpu_potential_ranges_preserve_independent_leaves_and_film_invariants() -> Result {
        configured_cpu(false)
    }

    #[test]
    fn configured_cpu_potential_ranges_preserve_independent_leaves_from_strided_views() -> Result {
        configured_cpu(true)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_potential_ranges_preserve_independent_leaves_and_film_invariants() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        matrix::<CudaRuntime>(client.clone(), true)?;
        matrix::<CudaRuntime>(client, false)
    }
}
