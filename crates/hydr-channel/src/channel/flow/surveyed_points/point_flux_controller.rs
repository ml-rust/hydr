use super::{
    point_flux_faces::{Continuation, Faces},
    point_flux_masks::equals,
    point_flux_types::{PointFluxStatus, ResolvedPointFluxes},
    point_root_session::{PointRootSession, invalid},
    point_wave_envelope,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps,
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
    /// Selects every face, then refines the possible star and fan candidates of each face
    /// over its flux budget. Each round advances the star rows once and each fan side once,
    /// then reselects the demanded faces. The rounds stop when every face is settled or no
    /// face advances, and never exceed `max_refinements`.
    /// Accepted star faces then sharpen their star bracket to the scale of their jump.
    /// Each face then takes the wave-speed envelope of its owning selection.
    pub(super) fn resolve_point_fluxes(
        &mut self,
        owners: &Tensor<R>,
    ) -> HydraulicResult<ResolvedPointFluxes<R>> {
        let work = self.sources.evaluation.work;
        let geometry = self.sources.geometry;
        let c = geometry.client();
        let points_before = self.callback_points();
        let residual_before = work.snapshot().residual_points;
        let initial = self.select_point_fluxes(owners)?;
        let count = self.invocation_rows.numel();
        let mut faces = checked::<R, _>(c, "point flux controller face state", |a| {
            Faces::new(a, count, initial)
        })?;
        let blocked = checked::<R, _>(c, "point flux controller positive-area faces", |a| {
            let pending = equals(
                a,
                &faces.status,
                PointFluxStatus::PendingPositiveArea as i64,
            )?;
            Ok(c.index_select(&faces.owner, 0, &c.nonzero(&pending)?.flatten()?)?)
        })?;
        let mut positive_support = None;
        if blocked.numel() != 0 {
            let support = self.advance_point_flux_positive_support(&blocked)?;
            self.invocation
                .check(&support.candidates.invocation_witness)?;
            let reselected = self.select_point_fluxes(&blocked)?;
            checked::<R, _>(c, "point flux controller positive-area reselection", |a| {
                faces.overwrite(a, count, reselected)
            })?;
            positive_support = Some(support);
        }
        let mut rounds = 0;
        for round in 1..=self.sources.evaluation.config.max_refinements {
            let demand = checked::<R, _>(c, "point flux controller demand", |a| faces.demand(a))?;
            if demand.all.numel() == 0 {
                break;
            }
            let mut advanced = 0;
            for continuation in Continuation::ALL {
                let owners = &demand.owners[continuation.index()];
                let moved = self.advance_continuation(&mut faces, continuation, owners)?;
                advanced = sum(advanced, moved)?;
            }
            if advanced == 0 {
                break;
            }
            work.begin_refinement(self.max_panels()?, round)?;
            rounds = round;
            let reselected = self.select_point_fluxes(&demand.all)?;
            checked::<R, _>(c, "point flux controller reselection", |a| {
                faces.overwrite(a, count, reselected)
            })?;
        }
        self.sharpen_star_faces(&mut faces, count)?;
        let last_level = self.last_level()?;
        checked::<R, _>(c, "point flux controller level ceiling", |a| {
            faces.close_ceiling(a, last_level)
        })?;
        let (envelope, status) = checked::<R, _>(c, "point flux controller final status", |a| {
            let status = faces.final_status(a)?;
            point_wave_envelope::resolve(a, count, &faces.owner, &faces.selections, &status)
        })?;
        let executed = work
            .snapshot()
            .residual_points
            .checked_sub(residual_before)
            .ok_or_else(|| invalid("point flux controller residual work decreased"))?;
        let recorded = self
            .callback_points()
            .checked_sub(points_before)
            .ok_or_else(|| invalid("point flux controller callback count decreased"))?;
        if executed != recorded {
            return Err(invalid(
                "point flux controller callback points differ from executed residual work",
            ));
        }
        let [star_outcome, _, _] = faces.outcome;
        let [star_level, _, _] = faces.level;
        Ok(ResolvedPointFluxes {
            invocation_owner: faces.owner,
            status,
            physical_valid: faces.physical_valid,
            joint_valid: faces.joint_valid,
            physical_water: faces.physical_water,
            physical_momentum: faces.physical_momentum,
            joint_water: faces.joint_water,
            joint_momentum: faces.joint_momentum,
            selection: faces.selection,
            selections: faces.selections,
            sonic_outcome: faces.sonic_outcome,
            sonic_level: faces.sonic_level,
            star_outcome,
            star_level,
            wave_speed_m_per_s: envelope.wave_speed_m_per_s,
            wave_valid: envelope.valid,
            positive_support,
            rounds,
            callback_points: self.callback_points(),
            point_limit: self.point_limit(),
        })
    }

    /// Tightens the depth tolerance of faces this continuation never tightened, then advances
    /// every demanded owner one level. Returns the count of faces that changed.
    /// A bracket that met a looser tolerance still has width to give, so the tightening
    /// lets each later level narrow it to the arithmetic floor.
    fn advance_continuation(
        &mut self,
        faces: &mut Faces<R>,
        continuation: Continuation,
        owners: &Tensor<R>,
    ) -> HydraulicResult<usize> {
        if owners.numel() == 0 {
            return Ok(0);
        }
        let c = self.sources.geometry.client();
        let fresh = checked::<R, _>(c, "point flux controller untightened faces", |a| {
            faces.untightened(a, continuation, owners)
        })?;
        let mut advanced = 0;
        if fresh.numel() != 0 {
            let progress = match continuation {
                Continuation::Star => self.tighten_wet_depth(&fresh)?,
                Continuation::Fan(side) => self.tighten_source_sonic(&fresh, side)?,
            };
            advanced = checked::<R, _>(c, "point flux controller tightening outcomes", |a| {
                faces.record_tightening(a, &progress)
            })?;
        }
        let progress = match continuation {
            Continuation::Star => self.advance_wet_star_quadrature(owners)?,
            Continuation::Fan(side) => self.advance_source_sonic_quadrature(owners, side)?,
        };
        let moved = checked::<R, _>(c, "point flux controller continuation outcomes", |a| {
            faces.record(a, &progress)
        })?;
        sum(advanced, moved)
    }

    /// The final admissible panel level under the refinement, panel and doubling ceilings.
    /// It reads only host config, so no device value.
    fn last_level(&self) -> HydraulicResult<usize> {
        let mut level = 0;
        while super::point_quadrature_capacity::next_level(self.sources, level)? {
            level = sum(level, 1)?;
        }
        Ok(level)
    }

    /// Largest Simpson panel count among the recorded sonic and star levels.
    /// Both level sets live on the host, so this reads no device value.
    fn max_panels(&self) -> HydraulicResult<usize> {
        let sonic = self.sources.panel_cache.iter().map(|(index, _)| *index);
        let star = self
            .wet
            .domains
            .iter()
            .flat_map(|domain| domain.levels.policies().iter().copied());
        match sonic.chain(star).max() {
            Some(index) => Ok(self.sources.panel_level(index)?.panels()),
            None => Ok(self.sources.evaluation.panels),
        }
    }
}

fn sum(left: usize, right: usize) -> HydraulicResult<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid("point flux controller advance count overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::super::point_flux::fixture::coherence;
    use super::super::{
        chart_fixture, source_fixture_support as frozen, wet_depth::fixture as support,
    };
    use frozen::{Result, hydraulic};

    #[test]
    fn configured_cpu_point_flux_controller_stops_at_its_ceilings() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = CpuClient::new(CpuDevice::new()).with_parallelism(config);
        controller::run::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_flux_controller_stops_at_its_ceilings() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        controller::run::<CudaRuntime>(&client)
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
}
