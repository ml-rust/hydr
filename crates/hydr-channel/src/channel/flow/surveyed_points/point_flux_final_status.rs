use super::{
    point_flux_faces::{Continuation, Faces},
    point_flux_masks::{code, equals, flag},
    point_flux_types::PointFluxStatus,
    point_quadrature_types::PointQuadratureOutcome as Outcome,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

impl<R> Faces<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Records `LevelLimit` for each over-budget face whose possible continuation last
    /// advanced to `last_level`, the final admissible level. No further advance can run there.
    pub(super) fn close_ceiling(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        last_level: usize,
    ) -> HydraulicResult<()> {
        let c = a.client();
        let shape = self.status.shape().to_vec();
        let budget = equals(a, &self.status, PointFluxStatus::FluxBudgetExceeded as i64)?;
        let limit = Tensor::full_scalar(
            &shape,
            DType::I64,
            Outcome::LevelLimit as i64 as f64,
            c.device(),
        )?;
        let zero = Tensor::zeros(&shape, a.dtype(), c.device())?;
        let last = i64::try_from(last_level)
            .map_err(|_| super::point_root_session::invalid("last panel level exceeds i64"))?;
        for continuation in Continuation::ALL {
            let index = continuation.index();
            let advanced = equals(a, &self.outcome[index], Outcome::Advanced as i64)?;
            let ceiling = flag(a, &c.ge(&self.level[index], &code(a, last)?)?)?;
            let mask = c.mul(
                &c.mul(&budget, &self.possible[index])?,
                &c.mul(&advanced, &ceiling)?,
            )?;
            let closed = c.gt(&mask, &zero)?;
            self.outcome[index] = c.where_cond(&closed, &limit, &self.outcome[index])?;
            self.terminal[index] = c.maximum(&self.terminal[index], &mask)?;
            if let Continuation::Fan(_) = continuation {
                self.sonic_outcome = c.where_cond(&closed, &limit, &self.sonic_outcome)?;
            }
        }
        Ok(())
    }

    /// Maps each face still over budget to the reason its possible continuations stopped.
    /// A work ceiling outranks an unresolved root. Every other stop stays `FluxBudgetExceeded`.
    pub(super) fn final_status(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<Tensor<R>> {
        let c = a.client();
        let shape = self.status.shape();
        let full = |value: PointFluxStatus| {
            Tensor::full_scalar(shape, DType::I64, value as i64 as f64, c.device())
        };
        let zero = Tensor::zeros(shape, a.dtype(), c.device())?;
        let (mut work, mut unresolved) = (zero.clone(), zero.clone());
        for continuation in Continuation::ALL {
            let index = continuation.index();
            let possible = &self.possible[index];
            let outcome = &self.outcome[index];
            let limited = c.mul(possible, &equals(a, outcome, Outcome::WorkLimit as i64)?)?;
            work = c.maximum(&work, &limited)?;
            let open = c.mul(possible, &equals(a, outcome, Outcome::Unresolved as i64)?)?;
            unresolved = c.maximum(&unresolved, &open)?;
        }
        let reason = c.where_cond(
            &c.gt(&work, &zero)?,
            &full(PointFluxStatus::WorkLimit)?,
            &c.where_cond(
                &c.gt(&unresolved, &zero)?,
                &full(PointFluxStatus::RootBracketUnresolved)?,
                &full(PointFluxStatus::FluxBudgetExceeded)?,
            )?,
        )?;
        Ok(c.where_cond(
            &c.eq(
                &self.status,
                &code(a, PointFluxStatus::FluxBudgetExceeded as i64)?,
            )?,
            &reason,
            &self.status,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::super::corpus_coverage::Coverage;
    use super::super::point_flux::fixture::coherence;
    use super::super::{
        chart_fixture, source_fixture_support as frozen, wet_depth::fixture as support,
    };
    use frozen::{Result, hydraulic};

    #[test]
    fn configured_cpu_point_fluxes_match_the_frozen_corpus() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = CpuClient::new(CpuDevice::new()).with_parallelism(config);
        corpus::run::<CpuRuntime>(&client, Coverage::Full)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_fluxes_match_the_frozen_corpus() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        corpus::run::<CudaRuntime>(&client, Coverage::Sample)
    }

    mod corpus {
        use super::super::super::{
            point_candidates::fixture::{fixtures, flux_fixtures},
            point_flux_types::PointFluxStatus,
            point_root_session::PointRootSession,
            source_sonic_quadrature::tests::CASES,
        };
        use super::{chart_fixture, coherence, frozen, hydraulic, support};
        use crate::channel::flow::{
            surveyed::error::{HydraulicError, PrismaticRiemannRejection},
            surveyed_points::corpus_coverage::Coverage,
        };
        use frozen::Result;
        use numr::{dtype::DType, runtime::Runtime};
        use std::collections::{BTreeMap, BTreeSet};

        pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let corpus = fixtures::executions();
            assert_eq!(corpus.len(), 70);
            let executions = coverage.select(corpus, |row| (row.outcome, row.physical));
            let references = flux_fixtures::REFERENCES
                .iter()
                .map(|r| (r.id, r))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(references.len(), 70);
            let mut outcomes = [0_usize; 7];
            let mut baseline = [0_usize; 4];
            let mut baseline_exceeded = BTreeSet::new();
            let mut evaluated = 0;
            let mut caps = 0;
            let mut work_limits = 0;
            for (config_index, original) in fixtures::CONFIGS.iter().enumerate() {
                let rows = executions
                    .iter()
                    .filter(|e| e.config == config_index)
                    .collect::<Vec<_>>();
                if rows.is_empty() {
                    continue;
                }
                let ids = rows.iter().map(|e| e.id).collect::<Vec<_>>();
                let config = original.build();
                chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                    let parents = (0..rows.len())
                        .map(i64::try_from)
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let parents = support::rows::<R>(client, &parents)?;
                    let mut session = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    let result = session.resolve_point_fluxes(&parents);
                    if rows.iter().all(|row| row.outcome == "WorkLimit") {
                        assert!(matches!(
                            result,
                            Err(HydraulicError::Rejected {
                                reason: PrismaticRiemannRejection::WorkLimit,
                                ..
                            })
                        ));
                        work_limits += rows.len();
                        assert_eq!(work.snapshot().residual_points, 0);
                        return Ok(());
                    }
                    let result = hydraulic(result)?;
                    if let Some(support) = &result.positive_support {
                        eprintln!(
                            "POINT_FLUX_POSITIVE_SUPPORT\towners={:?}\toutcomes={:?}\tprogress={:?}\tattempts={:?}\tadditional_points={}",
                            support.candidates.invocation_owner.try_to_vec::<i64>()?,
                            support.status.try_to_vec::<i64>()?,
                            support.progressed.try_to_vec::<f64>()?,
                            support.attempts.try_to_vec::<i64>()?,
                            support.additional_points
                        );
                        assert!(support.callback_points <= session.callback_points());
                        assert_eq!(support.point_limit, config.root.max_points);
                    }
                    coherence::coherent(&session, &result)?;
                    assert_eq!(
                        result.invocation_owner.try_to_vec::<i64>()?,
                        parents.try_to_vec::<i64>()?
                    );
                    assert!(result.rounds <= config.max_refinements);
                    assert_eq!(work.snapshot().refinements, result.rounds);
                    let initial = coherence::baseline(&result)?;
                    let levels = result.sonic_level.try_to_vec::<i64>()?;
                    let selection = result.selection.try_to_vec::<i64>()?;
                    let status = result.status.try_to_vec::<i64>()?;
                    let water = frozen::endpoints(&result.physical_water)?;
                    let momentum = frozen::endpoints(&result.physical_momentum)?;
                    let joint_q = frozen::endpoints(&result.joint_water)?;
                    let joint_j = frozen::endpoints(&result.joint_momentum)?;
                    for (i, row) in rows.iter().enumerate() {
                        let reference = references
                            .get(row.id)
                            .ok_or("Execution lacks independent full flux truth")?;
                        assert_eq!(row.config_identity, reference.config_identity);
                        assert_eq!(row.expected_identity, reference.expected_identity);
                        assert_eq!(row.source_group, reference.source_group);
                        assert_eq!(row.group, reference.physical_group);
                        assert_eq!(row.outcome, reference.outcome);
                        eprintln!(
                            "POINT_FLUX_ROW\t{}\tstatus={}\tinitial={}\tlevel={}\tphysical_q={:?}\tphysical_j={:?}\tjoint_q={:?}\tjoint_j={:?}\tpoints={}",
                            row.id,
                            status[i],
                            initial[i],
                            levels[i],
                            water[i],
                            momentum[i],
                            joint_q[i],
                            joint_j[i],
                            result.callback_points
                        );
                        if reference.water.is_none() || reference.momentum.is_none() {
                            assert!(
                                row.outcome == "CapExceeded"
                                    || row.outcome == "physical_cap_rejection"
                            );
                            assert_eq!(status[i], PointFluxStatus::PendingCandidates as i64);
                            caps += 1;
                            continue;
                        }
                        let owning = result
                            .selections
                            .get(usize::try_from(selection[i])?)
                            .ok_or("Face names an absent selection")?;
                        assert_ne!(
                            status[i],
                            PointFluxStatus::PendingCandidates as i64,
                            "{} loses existing candidate coverage",
                            row.id
                        );
                        if status[i] == PointFluxStatus::PendingPositiveArea as i64 {
                            let owner = i64::try_from(i)?;
                            for candidate in &owning.candidates.candidates {
                                if let super::super::super::point_candidate_types::PointCandidate::Fan(
                                    fan,
                                ) = candidate
                                {
                                    let certificate = &fan.certificate;
                                    let owners = certificate.invocation_owner.try_to_vec::<i64>()?;
                                    if let Some(position) =
                                        owners.iter().position(|value| *value == owner)
                                    {
                                        let area = frozen::endpoints(&certificate.query.area)?;
                                        let parameter =
                                            frozen::endpoints(&certificate.query.query.parameter)?;
                                        let scale =
                                            frozen::components(&certificate.query.query.scale)?;
                                        let depth = frozen::endpoints(&certificate.depth)?;
                                        let normalized =
                                            frozen::endpoints(&certificate.normalized_depth)?;
                                        let physical_status =
                                            certificate.status.try_to_vec::<i64>()?;
                                        let depth_status =
                                            certificate.depth_status.try_to_vec::<i64>()?;
                                        let active = fan.active.try_to_vec::<f64>()?;
                                        eprintln!(
                                            "POINT_FLUX_POSITIVE_AREA_DIAGNOSTIC\t{}\towner={owner}\tside={:?}\tactive={}\tarea={:?}\tparameter={:?}\tscale_bits={:?}\tdepth={:?}\tnormalized_depth={:?}\tclassification={:?}\tphysical_status={}\tdepth_status={}",
                                            row.id,
                                            certificate.side,
                                            active[position],
                                            area[position],
                                            parameter[position],
                                            scale[position],
                                            depth[position],
                                            normalized[position],
                                            certificate.classification,
                                            physical_status[position],
                                            depth_status[position]
                                        );
                                    }
                                }
                            }
                        }
                        assert_ne!(
                            status[i],
                            PointFluxStatus::PendingPositiveArea as i64,
                            "{} loses positive water proof",
                            row.id
                        );
                        let mirror = frozen::exact_mirror(row.id)?;
                        frozen::water_enclosed(
                            water[i],
                            reference
                                .water
                                .ok_or("Flux water truth absent")?
                                .map(f64::from_bits),
                            mirror,
                            row.id,
                        );
                        let owner = i64::try_from(i)?;
                        for point in &owning.selected {
                            let owners = point.invocation_owner.try_to_vec::<i64>()?;
                            let reflected = point.reflected.try_to_vec::<f64>()?;
                            if let Some(position) = owners.iter().position(|value| *value == owner)
                            {
                                let expected = if mirror { 1. } else { 0. };
                                assert_eq!(reflected[position], expected, "{} reflection", row.id);
                            }
                        }
                        contains(
                            momentum[i],
                            reference.momentum.ok_or("Flux momentum truth absent")?,
                            row.id,
                        );
                        if reference.positive_water_required {
                            let owner = i64::try_from(i)?;
                            let mut positive = false;
                            for point in &owning.selected {
                                let owners = point.invocation_owner.try_to_vec::<i64>()?;
                                let bounds =
                                    hydraulic(crate::channel::flow::surveyed::error::checked::<
                                        R,
                                        _,
                                    >(
                                        client,
                                        "declared positive selected water",
                                        |a| Ok(a.expansion_interval(&point.area)?),
                                    ))?;
                                let bounds = frozen::endpoints(&bounds)?;
                                for (index, selected_owner) in owners.iter().enumerate() {
                                    if *selected_owner == owner {
                                        assert!(
                                            bounds[index][0] > 0.,
                                            "{} loses declared positive water",
                                            row.id
                                        );
                                        positive = true;
                                    }
                                }
                            }
                            assert!(
                                positive,
                                "{} lacks its selected positive water point",
                                row.id
                            );
                        }
                        for declaration in reference.quantization {
                            let projected = declaration.projected.map(f64::from_bits);
                            let quantum = projected[1] - projected[0];
                            assert_eq!(quantum, f64::from_bits(1));
                            let bounds = if declaration.quantity == "sampled_complete_momentum" {
                                momentum[i]
                            } else if declaration.quantity == "sampled_pressure" {
                                let mut pressure_bounds = None;
                                let owner = i64::try_from(i)?;
                                for point in &owning.selected {
                                    let owners = point.invocation_owner.try_to_vec::<i64>()?;
                                    if let Some(position) =
                                        owners.iter().position(|value| *value == owner)
                                    {
                                        let pressure = point.pressure.as_ref().ok_or(
                                            "Positive quantum water has no pressure certificate",
                                        )?;
                                        pressure_bounds = Some(
                                            frozen::endpoints(&pressure.evaluation)?[position],
                                        );
                                    }
                                }
                                pressure_bounds.ok_or(
                                    "Declared quantum pressure has no selected certificate",
                                )?
                            } else {
                                return Err("Unknown frozen flux quantization quantity".into());
                            };
                            contains(bounds, declaration.projected, row.id);
                            assert!(
                                bounds[1] - bounds[0] <= quantum,
                                "{} {} exceeds its declared native quantum: {bounds:?}",
                                row.id,
                                declaration.quantity
                            );
                        }
                        if CASES.contains(&row.id) {
                            assert!(
                                levels[i] >= 1
                                    && usize::try_from(levels[i])? <= config.max_refinements,
                                "{} ends at sonic level {}",
                                row.id,
                                levels[i]
                            );
                        } else {
                            assert_eq!(
                                levels[i], -1,
                                "{} advances outside the sonic cases",
                                row.id
                            );
                        }
                        if initial[i] == PointFluxStatus::FluxBudgetExceeded as i64 {
                            baseline_exceeded.insert(row.id);
                        }
                        baseline[usize::try_from(initial[i])?] += 1;
                        outcomes[usize::try_from(status[i])?] += 1;
                        evaluated += 1;
                    }
                    assert_eq!(session.callback_points(), work.snapshot().residual_points);
                    assert!(session.callback_points() <= config.root.max_points);
                    Ok(())
                })?;
            }
            assert_eq!(evaluated + caps + work_limits, executions.len());
            assert_eq!(
                outcomes[PointFluxStatus::Accepted as usize],
                evaluated,
                "Evaluated rows stay unaccepted: {outcomes:?}"
            );
            assert!(baseline_exceeded.iter().all(|id| CASES.contains(id)));
            if coverage.is_full() {
                assert_eq!([evaluated, caps, work_limits], [66, 3, 1]);
                assert_eq!(baseline, [59, 0, 0, 7]);
                assert_eq!(
                    baseline_exceeded,
                    CASES.into_iter().collect::<BTreeSet<_>>()
                );
            }
            eprintln!(
                "POINT_FLUX_SUMMARY\tevaluated={evaluated}\tcap={caps}\twork={work_limits}\tbaseline={baseline:?}\tstatuses={outcomes:?}"
            );
            Ok(())
        }

        fn contains(actual: [f64; 2], expected: [u64; 2], label: &str) {
            let truth = expected.map(f64::from_bits);
            assert!(
                actual[0] <= truth[0] && truth[1] <= actual[1],
                "{label}: {actual:?} excludes {truth:?}"
            );
        }
    }
}
