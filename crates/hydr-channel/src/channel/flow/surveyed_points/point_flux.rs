use super::{
    point_candidate_types::PointCandidateStatus,
    point_flux_assembly::{Assembly, hull},
    point_flux_association, point_flux_operands,
    point_flux_types::{PointFluxCertificates, PointFluxKind, PointFluxStatus},
    point_root_session::PointRootSession,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

impl<R> PointRootSession<'_, '_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    pub(super) fn select_point_fluxes(
        &mut self,
        owners: &Tensor<R>,
    ) -> HydraulicResult<PointFluxCertificates<R>> {
        let candidates = self.certify_point_candidates(owners)?;
        self.invocation.check(&candidates.invocation_witness)?;
        let c = self.sources.geometry.client();
        let (mut physical, mut joint, mut chosen, mut pending, complete, association) =
            checked::<R, _>(c, "point flux coverage initialization", |a| {
                let count = candidates.invocation_owner.numel();
                let zero = Tensor::zeros(&[count], a.dtype(), c.device())?;
                let complete = c.where_cond(
                    &c.eq(
                        &candidates.status,
                        &Tensor::full_scalar(
                            &[],
                            DType::I64,
                            PointCandidateStatus::Complete as i64 as f64,
                            c.device(),
                        )?,
                    )?,
                    &Tensor::ones(candidates.status.shape(), a.dtype(), c.device())?,
                    &Tensor::zeros(candidates.status.shape(), a.dtype(), c.device())?,
                )?;
                let association = point_flux_association::inverse(
                    a,
                    self.invocation_rows.numel(),
                    &candidates.invocation_owner,
                )?;
                Ok((
                    Assembly::new(a, count)?,
                    Assembly::new(a, count)?,
                    zero.clone(),
                    zero,
                    complete,
                    association,
                ))
            })?;
        let mut selected = Vec::new();
        for index in 0..candidates.candidates.len() {
            let operand = checked::<R, _>(c, "point flux complete candidate operands", |a| {
                point_flux_operands::candidate(a, &candidates, index)
            })?;
            let Some(operand) = operand else {
                continue;
            };
            let (safe_rows, select_rows) =
                checked::<R, _>(c, "point flux positive area and coherent selection", |a| {
                    let positions =
                        point_flux_association::positions(a, &association, &operand.owners)?;
                    let zero = Tensor::zeros(operand.owners.shape(), a.dtype(), c.device())?;
                    let safe = if operand.kind == PointFluxKind::Dry {
                        Tensor::ones(operand.owners.shape(), a.dtype(), c.device())?
                    } else {
                        c.gt(operand.query.area.lower(), &zero)?
                    };
                    let unsafe_mask = c.eq(&safe, &zero)?;
                    let unsafe_mask = c.where_cond(
                        &unsafe_mask,
                        &Tensor::ones(operand.owners.shape(), a.dtype(), c.device())?,
                        &zero,
                    )?;
                    let previous = c.index_select(&pending, 0, &positions)?;
                    pending = c.index_put(
                        &pending,
                        0,
                        &positions,
                        &c.maximum(&previous, &unsafe_mask)?,
                    )?;
                    let ready = c.mul(&safe, &c.index_select(&complete, 0, &positions)?)?;
                    let available = c.eq(&c.index_select(&chosen, 0, &positions)?, &zero)?;
                    let take = c.mul(&ready, &available)?;
                    Ok((c.nonzero(&safe)?.flatten()?, c.nonzero(&take)?.flatten()?))
                })?;
            if safe_rows.numel() != 0 {
                checked::<R, _>(c, "point flux complete physical candidate hull", |a| {
                    let safe = point_flux_operands::select(a, &operand, &safe_rows)?;
                    let momentum = if safe.kind == PointFluxKind::Dry {
                        a.point(&Tensor::zeros(safe.owners.shape(), a.dtype(), c.device())?)?
                    } else {
                        let side = match safe.kind {
                            PointFluxKind::Exterior(side) | PointFluxKind::Fan(side) => {
                                side.index()
                            }
                            _ => 0,
                        };
                        let pressure = super::point_flux_pressure::physical(
                            a,
                            self.sources,
                            &safe.parent,
                            &safe.query,
                            side,
                        )?;
                        let velocity =
                            a.div_positive_interval(&safe.discharge, &safe.query.area)?;
                        let mut advective = a.mul(&safe.discharge, &velocity)?;
                        let stage = self.sources.select_rows(&safe.parent, &safe.query)?;
                        if let Some([area, discharge]) = &safe.captured {
                            let (_, retained) =
                                super::point_flux_arithmetic::advection(a, area, discharge)?;
                            advective = super::point_flux_arithmetic::intersect_advection(
                                a, &advective, &retained,
                            )?;
                        } else if safe.kind == PointFluxKind::Star {
                            let uniform_rows = c.nonzero(&stage.velocity.uniform)?.flatten()?;
                            if uniform_rows.numel() != 0 {
                                let parent = c.index_select(&safe.parent, 0, &uniform_rows)?;
                                let uniform = self.sources.select_sources(a, &parent)?;
                                let input = uniform.velocity.input();
                                let (_, retained) = super::point_flux_arithmetic::advection(
                                    a,
                                    &input.left_area,
                                    &input.left_discharge,
                                )?;
                                let ordinary =
                                    crate::channel::flow::surveyed::equation_ops::select_interval(
                                        a,
                                        &advective,
                                        &uniform_rows,
                                    )?;
                                let intersection =
                                    super::point_flux_arithmetic::intersect_advection(
                                        a, &ordinary, &retained,
                                    )?;
                                advective = a.bounds(
                                    &c.index_put(
                                        advective.lower(),
                                        0,
                                        &uniform_rows,
                                        intersection.lower(),
                                    )?,
                                    &c.index_put(
                                        advective.upper(),
                                        0,
                                        &uniform_rows,
                                        intersection.upper(),
                                    )?,
                                )?;
                            }
                        }
                        let momentum = a.add(&advective, &pressure)?;
                        let depth = self
                            .sources
                            .geometry
                            .recover_query(a, &stage.bands, &safe.query)?
                            .state
                            .depth;
                        super::point_flux_quantum::momentum(
                            a,
                            super::point_flux_quantum::MomentumInputs {
                                area: &safe.query.area,
                                discharge: &safe.discharge,
                                depth_upper: depth.upper(),
                                pressure: &pressure,
                                gravity: self.sources.evaluation.config.gravity_m_per_s2,
                            },
                            &momentum,
                        )?
                    };
                    let positions =
                        point_flux_association::positions(a, &association, &safe.owners)?;
                    physical.merge(a, &positions, &safe.discharge, &momentum)?;
                    joint.merge(a, &positions, &safe.discharge, &momentum)
                })?;
            }
            if select_rows.numel() == 0 {
                continue;
            }
            let operand = checked::<R, _>(c, "point flux one candidate selection", |a| {
                point_flux_operands::select(a, &operand, &select_rows)
            })?;
            let point = super::point_flux_selection::evaluate(self.sources, operand)?;
            checked::<R, _>(c, "point flux selected evaluation envelope", |a| {
                let water = a.expansion_interval(&point.discharge)?;
                let water = hull(a, &water, &point.discharge_physical)?;
                let arithmetic = a.expansion_interval(&point.momentum_arithmetic)?;
                let numerical = a.expansion_interval(&point.momentum)?;
                let momentum = hull(a, &point.momentum_evaluation, &arithmetic)?;
                let momentum = hull(a, &momentum, &numerical)?;
                let positions =
                    point_flux_association::positions(a, &association, &point.invocation_owner)?;
                joint.merge(a, &positions, &water, &momentum)?;
                chosen = c.index_put(
                    &chosen,
                    0,
                    &positions,
                    &Tensor::ones(point.invocation_owner.shape(), a.dtype(), c.device())?,
                )?;
                Ok(())
            })?;
            selected.push(point);
        }
        checked::<R, _>(c, "point flux unchanged full-width acceptance", |a| {
            let owners = &candidates.invocation_owner;
            let mut physical_water = physical.water;
            let physical_momentum = physical.momentum;
            let mut joint_water = joint.water;
            let joint_momentum = joint.momentum;
            let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
            let physical_valid =
                c.mul(&complete, &c.mul(&physical.seen, &c.eq(&pending, &zero)?)?)?;
            let joint_valid = c.mul(&chosen, &physical_valid)?;
            let ready = &joint_valid;
            let unreflected = super::point_flux_reflection::apply(
                a,
                self.sources.velocity.input(),
                &self.invocation_rows,
                super::point_flux_reflection::ReflectedFaces {
                    owners,
                    association: &association,
                    ready,
                    physical_water: &mut physical_water,
                    joint_water: &mut joint_water,
                    joint_momentum: &joint_momentum,
                },
                &mut selected,
            )?;
            let accepted = crate::channel::flow::surveyed::acceptance::flux_acceptance_mask(
                a,
                &joint_water,
                &joint_momentum,
                self.sources.evaluation.config,
            )?;
            let status = Tensor::full_scalar(
                owners.shape(),
                DType::I64,
                PointFluxStatus::PendingCandidates as i64 as f64,
                c.device(),
            )?;
            let status = c.where_cond(
                &complete,
                &c.where_cond(
                    &c.gt(&pending, &zero)?,
                    &Tensor::full_scalar(
                        owners.shape(),
                        DType::I64,
                        PointFluxStatus::PendingPositiveArea as i64 as f64,
                        c.device(),
                    )?,
                    &c.where_cond(
                        ready,
                        &c.where_cond(
                            &accepted,
                            &Tensor::full_scalar(
                                owners.shape(),
                                DType::I64,
                                PointFluxStatus::Accepted as i64 as f64,
                                c.device(),
                            )?,
                            &Tensor::full_scalar(
                                owners.shape(),
                                DType::I64,
                                PointFluxStatus::FluxBudgetExceeded as i64 as f64,
                                c.device(),
                            )?,
                        )?,
                        &status,
                    )?,
                )?,
                &status,
            )?;
            // A mirror face the reflection identity cannot certify has an unresolved trace.
            let status = c.where_cond(
                &c.gt(&unreflected, &zero)?,
                &Tensor::full_scalar(
                    owners.shape(),
                    DType::I64,
                    PointFluxStatus::WaveEnvelopeUnresolved as i64 as f64,
                    c.device(),
                )?,
                &status,
            )?;
            Ok(PointFluxCertificates {
                candidates,
                status,
                physical_valid,
                joint_valid,
                physical_water,
                physical_momentum,
                joint_water,
                joint_momentum,
                selected,
                callback_points: self.callback_points(),
                point_limit: self.point_limit(),
            })
        })
    }
}

#[cfg(test)]
#[path = "point_flux_fixture.rs"]
pub(super) mod fixture;

#[cfg(test)]
mod tests {
    use super::super::corpus_coverage::Coverage;
    use super::super::{
        chart_fixture,
        point_flux_types::{PointFluxKind, PointFluxStatus},
        point_root_session::PointRootSession,
        source_fixture_support as frozen,
        wet_depth::fixture as support,
    };
    use super::fixture::{coherence, diagnostics};
    use frozen::{Result, hydraulic};
    use numr::{dtype::DType, runtime::Runtime};

    fn basic<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_fixture::config();
        support::with_rectangle::<R, _>(
            client,
            1.,
            4.,
            &[[1., 1., 1., 1.], [0., 0., 0., 0.]],
            &config,
            |pair, work| {
                let owners = support::rows::<R>(client, &[1, 0, 0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let demand = support::rows::<R>(client, &[2, 0, 1])?;
                let result = hydraulic(session.select_point_fluxes(&demand))?;
                hydraulic(
                    session
                        .invocation
                        .check(&result.candidates.invocation_witness),
                )?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::Accepted as i64; 3]
                );
                let water = frozen::endpoints(&result.physical_water)?;
                let momentum = frozen::endpoints(&result.physical_momentum)?;
                for (row, q, j) in [(0, 1., 1.5), (1, 0., 0.), (2, 1., 1.5)] {
                    assert!(water[row][0] <= q && q <= water[row][1]);
                    assert!(momentum[row][0] <= j && j <= momentum[row][1]);
                }
                let mut count = 0;
                for point in &result.selected {
                    let area = frozen::components(&point.area)?;
                    let q = frozen::components(&point.discharge)?;
                    for row in 0..area.len() {
                        let value = if point.kind == PointFluxKind::Dry {
                            0.0_f64
                        } else {
                            1.0_f64
                        };
                        assert_eq!(area[row], [value.to_bits(), 0, 0]);
                        assert_eq!(q[row], [value.to_bits(), 0, 0]);
                        count += 1;
                    }
                }
                assert_eq!(count, 3);
                assert_eq!(result.callback_points, session.callback_points());
                assert_eq!(result.point_limit, config.root.max_points);
                assert_eq!(session.callback_points(), work.snapshot().residual_points);
                Ok(())
            },
        )
    }

    mod controller {
        use super::super::super::{
            point_candidates::fixture::fixtures,
            point_flux_types::{PointFluxKind, PointFluxStatus},
            point_quadrature_types::PointQuadratureOutcome as Outcome,
            point_root_session::PointRootSession,
            source_sonic_quadrature::tests::CASES,
        };
        use super::coherence::coherent;
        use super::{chart_fixture, frozen, hydraulic, support};
        use crate::channel::flow::surveyed::PrismaticRiemannConfig;
        use frozen::Result;
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            ceiling::<R>(client)?;
            exhausted::<R>(client)?;
            star_terminal::<R>(client)?;
            star_level_limit::<R>(client)?;
            Ok(())
        }

        fn config(id: &str) -> Result<PrismaticRiemannConfig> {
            let rows = fixtures::executions();
            let row = rows
                .iter()
                .find(|row| row.id == id)
                .ok_or("Missing frozen controller execution")?;
            Ok(fixtures::CONFIGS[row.config].build())
        }

        /// A one-level ceiling stops one sonic row after one round, still over budget.
        fn ceiling<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let id = CASES[6];
            let mut config = config(id)?;
            config.max_refinements = 1;
            chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let result = hydraulic(session.resolve_point_fluxes(&owners))?;
                coherent(&session, &result)?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::FluxBudgetExceeded as i64]
                );
                assert_eq!(result.rounds, 1);
                assert_eq!(result.selections.len(), 2);
                assert_eq!(result.sonic_level.try_to_vec::<i64>()?, [1]);
                // Level 1 is the last admissible level, so the outcome names the level ceiling.
                assert_eq!(
                    result.sonic_outcome.try_to_vec::<i64>()?,
                    [Outcome::LevelLimit as i64]
                );
                let executed = work.snapshot();
                assert_eq!(executed.refinements, 1);
                assert!(executed.maximum_simpson_panels <= config.initial_simpson_panels * 2);
                assert_eq!(session.callback_points(), executed.residual_points);
                Ok(())
            })
        }

        /// A point limit just past the initial selection stops the continuation at its work ceiling.
        fn exhausted<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let id = CASES[6];
            let config = config(id)?;
            let initial = chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, _| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                Ok(hydraulic(session.select_point_fluxes(&owners))?.callback_points)
            })?;
            let limit = initial + 1;
            chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session = hydraulic(PointRootSession::new(pair, &owners, limit))?;
                let result = hydraulic(session.resolve_point_fluxes(&owners))?;
                coherent(&session, &result)?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::WorkLimit as i64]
                );
                assert_eq!(
                    result.sonic_outcome.try_to_vec::<i64>()?,
                    [Outcome::WorkLimit as i64]
                );
                assert_eq!(result.rounds, 0);
                assert_eq!(result.point_limit, limit);
                let executed = work.snapshot();
                assert_eq!(executed.refinements, 0);
                assert_eq!(session.callback_points(), executed.residual_points);
                assert_eq!(result.callback_points, executed.residual_points);
                assert!(result.callback_points <= limit);
                Ok(())
            })
        }

        /// A zero flux budget refines a star row in every round, up to the refinement ceiling.
        /// The row stays over budget, and its sonic continuation never runs.
        fn star_terminal<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let id = "scalar_rectangle_depth_jump";
            let mut config = config(id)?;
            config.water_flux_atol_m3_per_s = 0.;
            config.momentum_flux_atol_m4_per_s2 = 0.;
            config.flux_rtol = 0.;
            chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let result = hydraulic(session.resolve_point_fluxes(&owners))?;
                coherent(&session, &result)?;
                let kinds = result.selections[0]
                    .selected
                    .iter()
                    .map(|point| point.kind)
                    .collect::<Vec<_>>();
                assert_eq!(kinds, [PointFluxKind::Star], "{id} selects {kinds:?}");
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::FluxBudgetExceeded as i64]
                );
                let star_level = result.star_level.try_to_vec::<i64>()?;
                eprintln!(
                    "POINT_FLUX_STAR_TERMINAL\t{id}\trounds={}\tstar_outcome={:?}\tstar_level={star_level:?}",
                    result.rounds,
                    result.star_outcome.try_to_vec::<i64>()?
                );
                // The last round reaches the refinement ceiling, so the outcome names the level ceiling.
                assert_eq!(
                    result.star_outcome.try_to_vec::<i64>()?,
                    [Outcome::LevelLimit as i64]
                );
                assert_eq!(star_level, [i64::try_from(config.max_refinements)?]);
                assert_eq!(result.rounds, config.max_refinements);
                assert_eq!(result.selections.len(), result.rounds + 1);
                assert_eq!(
                    result.sonic_outcome.try_to_vec::<i64>()?,
                    [Outcome::NotApplicable as i64]
                );
                assert_eq!(result.sonic_level.try_to_vec::<i64>()?, [-1]);
                let executed = work.snapshot();
                assert_eq!(executed.refinements, result.rounds);
                assert!(executed.maximum_simpson_panels <= config.max_simpson_panels);
                assert_eq!(session.callback_points(), executed.residual_points);
                Ok(())
            })
        }

        /// A panel ceiling below the round ceiling makes the star continuation terminal first.
        /// The round after the last level advances nothing, so the loop stops there.
        fn star_level_limit<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let id = "scalar_rectangle_depth_jump";
            let mut config = config(id)?;
            config.water_flux_atol_m3_per_s = 0.;
            config.momentum_flux_atol_m4_per_s2 = 0.;
            config.flux_rtol = 0.;
            config.max_simpson_panels = config.initial_simpson_panels * 4;
            chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let result = hydraulic(session.resolve_point_fluxes(&owners))?;
                coherent(&session, &result)?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::FluxBudgetExceeded as i64]
                );
                assert_eq!(
                    result.star_outcome.try_to_vec::<i64>()?,
                    [Outcome::LevelLimit as i64]
                );
                assert_eq!(result.star_level.try_to_vec::<i64>()?, [2]);
                assert_eq!(result.rounds, 2);
                assert_eq!(result.selections.len(), 3);
                let executed = work.snapshot();
                assert_eq!(executed.refinements, 2);
                assert_eq!(executed.maximum_simpson_panels, config.max_simpson_panels);
                assert_eq!(session.callback_points(), executed.residual_points);
                Ok(())
            })
        }
    }

    mod controls {
        use super::super::super::{
            point_flux_admission, point_flux_arithmetic, point_flux_types::PointFluxStatus,
            point_root_session::PointRootSession,
        };
        use super::{chart_fixture, diagnostics, frozen, hydraulic, support};
        use crate::channel::flow::surveyed::error::{
            self, HydraulicError, PrismaticRiemannRejection,
        };
        use frozen::Result;
        use numr::{
            dtype::DType,
            ops::BinaryOps,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            arithmetic::<R>(client)?;
            demand_and_isolation::<R>(client)
        }

        fn arithmetic<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let tiny = 2.0_f64.powi(-80);
            let (velocity, momentum) = hydraulic(error::checked::<R, _>(
                client,
                "independent exact dyadic momentum controls",
                |a| {
                    let area = a.expansion_point(&Tensor::from_slice(
                        &[2., 0.5, 1.],
                        &[3],
                        client.device(),
                    )?)?;
                    let discharge = a.expansion_point(&Tensor::from_slice(
                        &[3., -0.25, 0.],
                        &[3],
                        client.device(),
                    )?)?;
                    let pressure = a.expansion_bounds(
                        &Tensor::from_slice(&[0.625, 0.125, 1.], &[3], client.device())?,
                        &Tensor::from_slice(&[0., 0., tiny], &[3], client.device())?,
                        &Tensor::zeros(&[3], a.dtype(), client.device())?,
                    )?;
                    point_flux_arithmetic::momentum(a, &area, &discharge, &pressure)
                },
            ))?;
            let velocity = frozen::components(&velocity)?;
            let momentum = frozen::components(&momentum)?;
            for (row, expected_u, expected_j, expected_low) in
                [(0, 1.5, 5.125, 0.), (1, -0.5, 0.25, 0.), (2, 0., 1., tiny)]
            {
                let u = velocity[row].map(f64::from_bits);
                let j = momentum[row].map(f64::from_bits);
                assert_eq!([u[0], u[1]], [expected_u, 0.]);
                assert_eq!([j[0], j[1]], [expected_j, expected_low]);
                assert!(u[2] >= 0. && u[2] <= 2.0_f64.powi(-100));
                assert!(j[2] >= 0. && j[2] <= 2.0_f64.powi(-100));
            }
            Ok(())
        }

        fn demand_and_isolation<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                4.,
                &[[1., 1., 1., 1.], [0., 0., 0., 0.]],
                &config,
                |pair, work| {
                    let parents = support::rows::<R>(client, &[0, 1, 0, 1])?;
                    let mut session = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    let before = work.snapshot();
                    let empty = Tensor::empty(&[0], DType::I64, client.device())?;
                    let empty = hydraulic(session.select_point_fluxes(&empty))?;
                    assert!(empty.selected.is_empty());
                    assert_eq!(empty.status.numel(), 0);
                    for interval in [
                        &empty.physical_water,
                        &empty.physical_momentum,
                        &empty.joint_water,
                        &empty.joint_momentum,
                    ] {
                        assert_eq!(interval.lower().numel(), 0);
                        assert_eq!(interval.upper().numel(), 0);
                    }
                    assert_eq!(work.snapshot(), before);
                    let storage =
                        Tensor::from_slice(&[777_i64, 2, 888, 0], &[2, 2], client.device())?;
                    let demand = storage.narrow(1, 1, 1)?.squeeze(Some(1));
                    assert!(!demand.is_contiguous());
                    let result = hydraulic(session.select_point_fluxes(&demand))?;
                    assert_eq!(
                        result.candidates.invocation_owner.try_to_vec::<i64>()?,
                        [2, 0]
                    );
                    assert_eq!(result.candidates.parent_rows.try_to_vec::<i64>()?, [0, 0]);
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [PointFluxStatus::Accepted as i64; 2]
                    );
                    assert_eq!(result.physical_water.lower().numel(), 2);
                    assert_eq!(result.joint_momentum.upper().numel(), 2);
                    diagnostics::check(&session, &result)?;
                    let counts = work.snapshot();
                    let callbacks = session.callback_points();
                    for point in &result.selected {
                        let zeros =
                            Tensor::zeros(point.area.high().shape(), DType::F64, client.device())?;
                        client.copy_into(point.area.high(), &zeros)?;
                        client.copy_into(point.discharge.high(), &zeros)?;
                        client.copy_into(point.query.query.parameter.lower(), &zeros)?;
                    }
                    let repeated = hydraulic(session.select_point_fluxes(&demand))?;
                    diagnostics::check(&session, &repeated)?;
                    assert_eq!(session.callback_points(), callbacks);
                    assert_eq!(work.snapshot().residual_points, counts.residual_points);
                    assert_eq!(work.snapshot().root_solves, counts.root_solves);
                    assert_eq!(work.snapshot().root_iterations, counts.root_iterations);
                    for point in &repeated.selected {
                        for state in [
                            frozen::components(&point.area)?,
                            frozen::components(&point.discharge)?,
                        ] {
                            assert!(
                                state
                                    .iter()
                                    .all(|value| *value == [1.0_f64.to_bits(), 0, 0])
                            );
                        }
                    }
                    let query = &repeated
                        .selected
                        .first()
                        .ok_or("No repeated selected point")?
                        .query;
                    let original = query.query.original_rows.try_to_vec::<i64>()?[0];
                    let profile = query.query.owners.try_to_vec::<i64>()?[0];
                    let rejected = error::checked::<R, _>(
                        client,
                        "typed selected positive geometry rejection",
                        |a| {
                            point_flux_admission::positive(
                                a,
                                query,
                                &Tensor::ones(
                                    query.query.owners.shape(),
                                    a.dtype(),
                                    client.device(),
                                )?,
                                "control depth",
                            )
                        },
                    );
                    assert!(matches!(rejected, Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::GeometryUnresolved,
            query: Some(row), section_owner: Some(owner), ..
        }) if row == usize::try_from(original)? && owner == usize::try_from(profile)?));
                    drop(session);
                    let foreign = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    assert!(
                        foreign
                            .invocation
                            .check(&repeated.candidates.invocation_witness)
                            .is_err()
                    );
                    Ok(())
                },
            )
        }
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

    mod positive_support {
        use super::super::super::{
            point_candidates::fixture::fixtures,
            point_flux_association, point_flux_support_metrics,
            point_flux_types::{PointFluxStatus, PointPositiveStatus},
            point_root_session::PointRootSession,
            source_sonic_types::SonicSide,
        };
        use super::{chart_fixture, diagnostics, frozen, hydraulic, support};
        use crate::channel::flow::surveyed::error::checked;
        use frozen::Result;
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            quarter::<R>(client)?;
            advance::<R>(client, false)?;
            advance::<R>(client, true)
        }

        fn quarter<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let (value, positive) = hydraulic(checked::<R, _>(
                client,
                "finite positive runtime target admission",
                |a| {
                    let upper =
                        Tensor::from_slice(&[0., f64::from_bits(1), 1.], &[3], client.device())?;
                    point_flux_support_metrics::finite_quarter(a, &upper)
                },
            ))?;
            assert_eq!(value.try_to_vec::<f64>()?, [0., 0., 0.25]);
            assert_eq!(positive.try_to_vec::<f64>()?, [0., 0., 1.]);
            Ok(())
        }

        fn advance<R>(client: &R::Client, exhausted: bool) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let executions = fixtures::executions();
            let case = executions
                .iter()
                .find(|case| case.id == "scalar_rectangle_vacuum_threshold_inside")
                .ok_or("Missing original inside-threshold execution")?;
            let config = fixtures::CONFIGS[case.config].build();
            chart_fixture::with_pair::<R, _>(client, &[case.id], &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let initial = hydraulic(session.select_point_fluxes(&owners))?;
                diagnostics::check(&session, &initial)?;
                assert_eq!(
                    initial.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::PendingPositiveArea as i64]
                );
                assert_eq!(initial.physical_valid.try_to_vec::<f64>()?, [0.]);
                assert_eq!(initial.joint_valid.try_to_vec::<f64>()?, [0.]);
                assert!(initial.selected.is_empty());
                let unchanged_progress = hydraulic(checked::<R, _>(
                    client,
                    "unchanged enclosure never establishes progress",
                    |a| {
                        let inverse = point_flux_association::inverse(
                            a,
                            session.invocation_rows.numel(),
                            &initial.candidates.invocation_owner,
                        )?;
                        let before =
                            point_flux_support_metrics::collect(a, &initial.candidates, &inverse)?;
                        let after =
                            point_flux_support_metrics::collect(a, &initial.candidates, &inverse)?;
                        point_flux_support_metrics::improved(
                            a,
                            &before.fans[SonicSide::Right.index()],
                            &after.fans[SonicSide::Right.index()],
                        )
                    },
                ))?;
                assert_eq!(unchanged_progress.try_to_vec::<f64>()?, [0.]);
                if exhausted {
                    let ceiling = session.callback_points();
                    drop(session);
                    let mut limited = hydraulic(PointRootSession::new(pair, &owners, ceiling))?;
                    let raw = hydraulic(limited.select_point_fluxes(&owners))?;
                    assert_eq!(
                        raw.status.try_to_vec::<i64>()?,
                        [PointFluxStatus::PendingPositiveArea as i64]
                    );
                    assert_eq!(limited.callback_points(), ceiling);
                    let before = work.snapshot();
                    let result = hydraulic(limited.advance_point_flux_positive_support(&owners))?;
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [PointPositiveStatus::WorkLimit as i64]
                    );
                    assert_eq!(result.attempts.try_to_vec::<i64>()?, [0]);
                    assert_eq!(result.progressed.try_to_vec::<f64>()?, [0.]);
                    assert_eq!(result.additional_points, 0);
                    assert_eq!(result.point_limit, ceiling);
                    assert_eq!(result.callback_points, ceiling);
                    assert_eq!(work.snapshot(), before);
                    return Ok(());
                }
                let wet = session
                    .wet
                    .domains
                    .iter()
                    .map(|domain| {
                        Ok((
                            domain.query.query.parameter.lower().try_to_vec::<f64>()?,
                            domain.query.query.parameter.upper().try_to_vec::<f64>()?,
                            domain.iterations,
                            domain.callback_points,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let points = session.callback_points();
                let result = hydraulic(session.advance_point_flux_positive_support(&owners))?;
                hydraulic(
                    session
                        .invocation
                        .check(&result.candidates.invocation_witness),
                )?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointPositiveStatus::Supported as i64]
                );
                assert_eq!(result.attempts.try_to_vec::<i64>()?, [1]);
                assert_eq!(result.progressed.try_to_vec::<f64>()?, [1.]);
                assert!(result.additional_points > 0);
                assert_eq!(result.callback_points - points, result.additional_points);
                assert_eq!(result.point_limit, config.root.max_points);
                assert!(
                    session
                        .sonic
                        .requests
                        .iter()
                        .all(|request| request.binding.side == SonicSide::Right)
                );
                let after_wet = session
                    .wet
                    .domains
                    .iter()
                    .map(|domain| {
                        Ok((
                            domain.query.query.parameter.lower().try_to_vec::<f64>()?,
                            domain.query.query.parameter.upper().try_to_vec::<f64>()?,
                            domain.iterations,
                            domain.callback_points,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                assert_eq!(wet, after_wet);
                let accepted = hydraulic(session.select_point_fluxes(&owners))?;
                diagnostics::check(&session, &accepted)?;
                assert_eq!(
                    accepted.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::Accepted as i64]
                );
                assert_eq!(accepted.physical_valid.try_to_vec::<f64>()?, [1.]);
                assert_eq!(accepted.joint_valid.try_to_vec::<f64>()?, [1.]);
                let points = session.callback_points();
                let repeated = hydraulic(session.advance_point_flux_positive_support(&owners))?;
                assert_eq!(
                    repeated.status.try_to_vec::<i64>()?,
                    [PointPositiveStatus::Supported as i64]
                );
                assert_eq!(repeated.attempts.try_to_vec::<i64>()?, [0]);
                assert_eq!(repeated.progressed.try_to_vec::<f64>()?, [0.]);
                assert_eq!(repeated.additional_points, 0);
                assert_eq!(session.callback_points(), points);
                Ok(())
            })
        }
    }

    mod precision {
        use super::super::super::{
            point_candidates::fixture::{fixtures, flux_fixtures},
            point_flux_types::{PointFluxKind, PointFluxStatus},
            point_root_session::PointRootSession,
        };
        use super::{chart_fixture, diagnostics, frozen, hydraulic, support};
        use crate::channel::flow::surveyed::error::checked;
        use frozen::Result;
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let cases = fixtures::executions();
            for id in ["retained_vacuum_inside_film", "tiny_rectangle_compression"] {
                let case = cases
                    .iter()
                    .find(|e| e.id == id)
                    .ok_or("Missing frozen precision execution")?;
                let reference = flux_fixtures::REFERENCES
                    .iter()
                    .find(|r| r.id == id)
                    .ok_or("Missing frozen full flux precision truth")?;
                let config = fixtures::CONFIGS[case.config].build();
                chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, work| {
                    let source_a = frozen::components(&pair.velocity.input().left_area)?;
                    let source_q = frozen::components(&pair.velocity.input().left_discharge)?;
                    let cap = pair.bands.cap_depth.upper().try_to_vec::<f64>()?[0];
                    let owners = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                    let initial = hydraulic(session.select_point_fluxes(&owners))?;
                    if id == "retained_vacuum_inside_film" {
                        let star = case.star.ok_or("Film lacks independent star area")?;
                        let atol = f64::from_bits(star.area[0]) / cap * 1e-12;
                        assert!(atol > 0. && atol < config.root.atol);
                        hydraulic(
                            session.refine_wet_depth_in_place(CertifiedRootTolerance {
                                atol,
                                rtol: 0.,
                            }),
                        )?;
                        hydraulic(
                            session.refine_source_sonic(CertifiedRootTolerance { atol, rtol: 0. }),
                        )?;
                    }
                    let before = work.snapshot();
                    let result = hydraulic(session.select_point_fluxes(&owners))?;
                    diagnostics::check(&session, &result)?;
                    assert_eq!(session.callback_points(), before.residual_points);
                    assert_eq!(work.snapshot().root_solves, before.root_solves);
                    assert_eq!(result.point_limit, initial.point_limit);
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [PointFluxStatus::Accepted as i64]
                    );
                    let point = result
                        .selected
                        .first()
                        .ok_or("Precision case lacks selected point")?;
                    let (water, momentum) = hydraulic(checked::<R, _>(
                        client,
                        "selected numerical full flux precision",
                        |a| {
                            Ok((
                                a.expansion_interval(&point.discharge)?,
                                a.expansion_interval(&point.momentum)?,
                            ))
                        },
                    ))?;
                    for (actual, expected) in [
                        (
                            frozen::endpoints(&water)?[0],
                            reference.water.ok_or("Missing water truth")?,
                        ),
                        (
                            frozen::endpoints(&momentum)?[0],
                            reference.momentum.ok_or("Missing momentum truth")?,
                        ),
                    ] {
                        let expected = expected.map(f64::from_bits);
                        let scale = expected[0].abs().max(expected[1].abs());
                        assert!(scale > 0.);
                        let error = (actual[0] - expected[1])
                            .abs()
                            .max((actual[1] - expected[0]).abs());
                        eprintln!(
                            "POINT_FLUX_PRECISION\t{id}\tselected={actual:?}\ttruth={expected:?}\trelative_error={:e}",
                            error / scale
                        );
                        assert!(
                            error / scale < 1e-10,
                            "{id} loses original selected full flux precision"
                        );
                    }
                    if id == "tiny_rectangle_compression" {
                        assert!(matches!(point.kind, PointFluxKind::Exterior(_)));
                        assert_eq!(frozen::components(&point.area)?, source_a);
                        assert_eq!(frozen::components(&point.discharge)?, source_q);
                    } else {
                        assert!(matches!(point.kind, PointFluxKind::Fan(_)));
                        let proposal =
                            frozen::components(&point.discharge_proposal)?[0].map(f64::from_bits);
                        assert_eq!([proposal[0], proposal[1]], [0., 0.]);
                        assert_eq!(
                            point.discharge_proposal_contained.try_to_vec::<f64>()?,
                            [0.]
                        );
                        let physical = frozen::endpoints(&point.discharge_physical)?[0];
                        assert!(physical[1] < 0.);
                        let chosen = frozen::components(&point.discharge)?[0];
                        let change = frozen::components(&point.discharge_selection_change)?[0];
                        assert_eq!(&chosen[..2], &change[..2]);
                        let raw =
                            frozen::components(&point.discharge_evaluation)?[0].map(f64::from_bits);
                        assert!(raw[2] > 0.);
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }
    }

    #[test]
    fn configured_cpu_point_fluxes_select_coherent_conserved_states() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        basic::<CpuRuntime>(&client)?;
        corpus::run::<CpuRuntime>(&client, Coverage::Full)?;
        controls::run::<CpuRuntime>(&client)?;
        controller::run::<CpuRuntime>(&client)?;
        positive_support::run::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_fluxes_select_coherent_conserved_states() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        basic::<CudaRuntime>(&client)?;
        corpus::run::<CudaRuntime>(&client, Coverage::Sample)?;
        controller::run::<CudaRuntime>(&client)
    }

    #[test]
    fn configured_cpu_point_fluxes_preserve_film_and_source_precision() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        precision::run::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_fluxes_preserve_film_and_source_precision() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        precision::run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
