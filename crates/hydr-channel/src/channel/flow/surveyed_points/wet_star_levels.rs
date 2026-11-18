use crate::channel::flow::surveyed::error::HydraulicResult;
use numr::{
    dtype::DType,
    ops::{CompareOps, IndexingOps, TensorOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Per-row Simpson panel levels of wet star rows and every level they use.
pub(super) struct WetStarLevels<R: Runtime<DType = DType>> {
    levels: Tensor<R>,
    policies: Vec<usize>,
}

impl<R: Runtime<DType = DType>> Clone for WetStarLevels<R> {
    fn clone(&self) -> Self {
        Self {
            levels: self.levels.clone(),
            policies: self.policies.clone(),
        }
    }
}

impl<R> WetStarLevels<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    /// Every row starts at level 0, the initial Simpson panels.
    pub(super) fn initial(c: &R::Client, count: usize) -> HydraulicResult<Self> {
        Ok(Self {
            levels: Tensor::zeros(&[count], DType::I64, c.device())?,
            policies: vec![0],
        })
    }

    pub(super) fn levels(&self) -> &Tensor<R> {
        &self.levels
    }

    pub(super) fn policies(&self) -> &[usize] {
        &self.policies
    }

    /// Selected rows keep every policy, so grouping skips the empty ones.
    pub(super) fn selected(&self, c: &R::Client, rows: &Tensor<R>) -> HydraulicResult<Self> {
        if let [index] = self.policies() {
            return Ok(Self {
                levels: Tensor::full_scalar(rows.shape(), DType::I64, *index as f64, c.device())?,
                policies: vec![*index],
            });
        }
        Ok(Self {
            levels: c.index_select(&self.levels, 0, rows)?,
            policies: self.policies.clone(),
        })
    }

    /// Local rows at one panel level.
    pub(super) fn rows_at(&self, c: &R::Client, index: usize) -> HydraulicResult<Tensor<R>> {
        super::source_sonic_policy::policy_rows::<R>(c, &self.levels, index)
    }

    /// Moves local rows to one admitted panel level. Policies keep only the levels in use.
    pub(super) fn with_rows_at(
        &self,
        c: &R::Client,
        rows: &Tensor<R>,
        index: usize,
    ) -> HydraulicResult<Self> {
        use numr::ops::SortingOps;
        let level = Tensor::full_scalar(rows.shape(), DType::I64, index as f64, c.device())?;
        let levels = c.index_put(&self.levels, 0, rows, &level)?;
        let mut policies = Vec::with_capacity(self.policies.len() + 1);
        for policy in &self.policies {
            let value = Tensor::full_scalar(&[], DType::I64, *policy as f64, c.device())?;
            if c.nonzero(&c.eq(&levels, &value)?)?.numel() != 0 {
                policies.push(*policy);
            }
        }
        if !policies.contains(&index) {
            policies.push(index);
        }
        Ok(Self { levels, policies })
    }
}

#[cfg(test)]
pub(in crate::channel::flow::surveyed_points) mod tests {
    use super::super::{
        chart_test_support, paired_residual,
        point_candidate_types::PointCandidate,
        point_candidates::tests::{fixtures, flux_fixtures},
        point_flux_types::{PointFluxCertificates, PointFluxKind, SelectedPointFlux},
        point_root_session::PointRootSession,
        source_tests::support as frozen,
        wet_depth::tests::support,
    };
    use crate::channel::{
        flow::surveyed::{PrismaticRiemannConfig, error::checked},
        section::{
            CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PhysicalEndpoint,
            PrismaticSectionBatch,
        },
    };
    use frozen::{Result, hydraulic};
    use numr::{
        dtype::DType,
        ops::UtilityOps,
        runtime::{Runtime, RuntimeClient},
        tensor::Tensor,
    };

    /// Raised level for the test rows. Level 4 uses 64 panels.
    const LEVEL: usize = 4;

    /// A trapezoid has no analytic potential, so its star rows integrate numerically.
    pub(in crate::channel::flow::surveyed_points) fn trapezoid<R>(
        client: &R::Client,
    ) -> Result<PrismaticSectionBatch<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: frozen::Client<R>,
    {
        section::<R>(
            client,
            "level_trapezoid",
            [(0.0, 4.0), (4.0, 0.0), (8.0, 0.0), (12.0, 4.0)],
        )
    }

    /// The integer trapezoid of the surveyed reach tests: a 2 m bed with 1:1 banks.
    pub(in crate::channel::flow::surveyed_points) fn integer_trapezoid<R>(
        client: &R::Client,
    ) -> Result<PrismaticSectionBatch<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: frozen::Client<R>,
    {
        section::<R>(
            client,
            "integer_trapezoid",
            [(-5.0, 4.0), (-1.0, 0.0), (1.0, 0.0), (5.0, 4.0)],
        )
    }

    fn section<R>(
        client: &R::Client,
        id: &str,
        points: [(f64, f64); 4],
    ) -> Result<PrismaticSectionBatch<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: frozen::Client<R>,
    {
        let definition = CrossSectionDefinition {
            id: id.into(),
            vertical_datum: "Exact trapezoid datum".into(),
            points: points
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
            maximum_stage_m: 4.0,
            profile_provenance: "Independent dyadic proof".into(),
        };
        Ok(CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?)
    }

    /// One moving cell between walls: the left face sees its mirror state.
    /// The double rarefaction has the exact star discharge 0.
    pub(in crate::channel::flow::surveyed_points) const MIRROR: [[f64; 4]; 1] =
        [[12.0, -36.0, 12.0, 36.0]];

    /// Exact mirror faces a wall builds on `integer_trapezoid` with 4 m cells.
    /// Area is V/4 and discharge is M/4, both exact.
    /// One moving cell (V=12, M=36) between walls has a rarefying left face and a colliding
    /// right face. A fast shallow cell (V=2, M=40) meets the right wall.
    pub(in crate::channel::flow::surveyed_points) const WALLS: [(&str, [f64; 4]); 3] = [
        ("single cell left wall", [3.0, -9.0, 3.0, 9.0]),
        ("single cell right wall", [3.0, 9.0, 3.0, -9.0]),
        ("sources right wall", [0.5, 10.0, 0.5, -10.0]),
    ];

    fn config() -> PrismaticRiemannConfig {
        PrismaticRiemannConfig::default()
    }

    fn raise<R>(session: &mut PointRootSession<'_, '_, R>, index: usize) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let c = session.sources.geometry.client();
        for domain in &mut session.wet.domains {
            let rows = c.arange(0., domain.parent_rows.numel() as f64, 1., DType::I64)?;
            domain.levels = hydraulic(domain.levels.with_rows_at(c, &rows, index))?;
            assert_eq!(domain.levels.policies(), [index]);
            assert!(
                domain
                    .levels
                    .levels()
                    .try_to_vec::<i64>()?
                    .iter()
                    .all(|level| *level == index as i64)
            );
        }
        Ok(())
    }

    struct Star {
        level: usize,
        discharge: [f64; 2],
        sides: [[f64; 2]; 2],
    }

    fn star<R>(result: &PointFluxCertificates<R>) -> Result<Star>
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        for candidate in &result.candidates.candidates {
            if let PointCandidate::Star(s) = candidate {
                let wave = &result.candidates.waves[s.wave];
                let row = usize::try_from(s.rows.try_to_vec::<i64>()?[0])?;
                return Ok(Star {
                    level: wave.star.level.index(),
                    discharge: frozen::endpoints(&wave.star.discharge)?[row],
                    sides: [
                        frozen::endpoints(&wave.star.side_discharge[0])?[row],
                        frozen::endpoints(&wave.star.side_discharge[1])?[row],
                    ],
                });
            }
        }
        Err("Result lacks a star candidate".into())
    }

    fn selected<R: Runtime<DType = DType>>(
        result: &PointFluxCertificates<R>,
    ) -> Result<&SelectedPointFlux<R>> {
        result
            .selected
            .iter()
            .find(|point| point.kind == PointFluxKind::Star)
            .ok_or_else(|| "Result lacks a selected star point".into())
    }

    fn width(value: [f64; 2]) -> f64 {
        value[1] - value[0]
    }

    fn contains(outer: [f64; 2], inner: [f64; 2], label: &str) {
        assert!(
            outer[0] <= inner[0] && inner[1] <= outer[1],
            "{label}: {outer:?} excludes {inner:?}"
        );
    }

    /// The old selection: the left value at the selected area and its invariant product.
    fn left_only<R>(client: &R::Client, point: &SelectedPointFlux<R>) -> Result<[f64; 2]>
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let evaluation = hydraulic(checked::<R, _>(client, "left-only selection", |a| {
            Ok(a.expansion_interval(&point.discharge_evaluation)?)
        }))?;
        let evaluation = frozen::endpoints(&evaluation)?[0];
        let product = frozen::endpoints(&point.discharge_product)?[0];
        Ok([evaluation[0].max(product[0]), evaluation[1].min(product[1])])
    }

    fn mirror<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let section = trapezoid::<R>(client)?;
        let config = config();
        support::with_section::<R, _>(client, &section, &MIRROR, &config, |pair, _| {
            let owners = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            let base = hydraulic(session.select_point_fluxes(&owners))?;
            let initial = star(&base)?;
            assert_eq!(initial.level, 0);
            raise(&mut session, LEVEL)?;
            let refined = hydraulic(session.select_point_fluxes(&owners))?;
            let raised = star(&refined)?;
            let point = selected(&refined)?;
            let physical = frozen::endpoints(&point.discharge_physical)?[0];
            eprintln!(
                "WET_STAR_LEVEL\tmirror\tlevel0={:?}\tlevel{LEVEL}={:?}\tselected={physical:?}",
                initial.discharge, raised.discharge
            );
            assert_eq!(raised.level, LEVEL);
            assert!(width(raised.discharge) < width(initial.discharge));
            contains(raised.discharge, [0.0, 0.0], "raised mirror star");
            contains(initial.discharge, [0.0, 0.0], "initial mirror star");
            contains(physical, [0.0, 0.0], "raised mirror selection");
            Ok(())
        })
    }

    fn corpus<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let executions = fixtures::executions();
        for id in [
            "scalar_trapezoid_weak_depth_jump",
            "scalar_three_bands_second_knot_neighbors",
        ] {
            let case = executions
                .iter()
                .find(|case| case.id == id)
                .ok_or("Missing frozen star execution")?;
            let truth = flux_fixtures::REFERENCES
                .iter()
                .find(|reference| reference.id == id)
                .ok_or("Missing frozen star truth")?
                .water
                .ok_or("Star truth lacks water")?
                .map(f64::from_bits);
            let config = fixtures::CONFIGS[case.config].build();
            chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, _| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let base = hydraulic(session.select_point_fluxes(&owners))?;
                let initial = star(&base)?;
                let point = selected(&base)?;
                let physical = frozen::endpoints(&point.discharge_physical)?[0];
                let left = left_only::<R>(client, point)?;
                eprintln!(
                    "WET_STAR_TWO_SIDED\t{id}\tsides={:?}\tstar={:?}\tselected={physical:?}\tleft_only={left:?}",
                    initial.sides, initial.discharge
                );
                for side in initial.sides {
                    contains(side, physical, id);
                }
                assert_ne!(initial.sides[0], initial.sides[1], "{id} sides agree");
                assert!(width(physical) < width(initial.sides[0]), "{id}");
                contains(physical, truth, id);
                // The left value at the selected area excludes Q*, so it is no Q* enclosure.
                assert!(left[1] < truth[0] || truth[1] < left[0], "{id}");
                raise(&mut session, LEVEL)?;
                let refined = hydraulic(session.select_point_fluxes(&owners))?;
                let raised = star(&refined)?;
                eprintln!(
                    "WET_STAR_LEVEL\t{id}\tlevel0={:?}\tlevel{LEVEL}={:?}\ttruth={truth:?}",
                    initial.discharge, raised.discharge
                );
                assert_eq!(raised.level, LEVEL);
                contains(raised.discharge, truth, id);
                contains(
                    frozen::endpoints(&selected(&refined)?.discharge_physical)?[0],
                    truth,
                    id,
                );
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Mixed levels evaluate in groups and match each level evaluated alone.
    fn grouped<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let section = trapezoid::<R>(client)?;
        let config = config();
        let states = [
            MIRROR[0],
            [12.0, -30.0, 12.0, 36.0],
            [12.0, -36.0, 12.0, 30.0],
        ];
        support::with_section::<R, _>(client, &section, &states, &config, |pair, _| {
            let owners = support::rows::<R>(client, &[0, 1, 2])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            hydraulic(session.select_point_fluxes(&owners))?;
            let domain = session.wet.domains.first().ok_or("Missing wet domain")?;
            let rows = Tensor::<R>::from_slice(
                &domain.parent_rows.try_to_vec::<i64>()?,
                domain.parent_rows.shape(),
                client.device(),
            )?;
            let n = rows.numel();
            let query = hydraulic(checked::<R, _>(client, "grouped query copy", |a| {
                let all = client.arange(0., n as f64, 1., DType::I64)?;
                domain.query.select(a, &all)
            }))?;
            let raised_rows = support::rows::<R>(client, &[1])?;
            let mixed = hydraulic(domain.levels.with_rows_at(client, &raised_rows, LEVEL))?;
            assert_eq!(mixed.policies(), [0, LEVEL]);
            let positions = client.arange(0., n as f64, 1., DType::I64)?;
            let sources = &mut *session.sources;
            let grouped = hydraulic(paired_residual::endpoint_at_levels(
                sources,
                &rows,
                &query,
                PhysicalEndpoint::Lower,
                &mixed,
                &positions,
            ))?;
            let initial =
                hydraulic(sources.endpoint_on_rows(&rows, &query, PhysicalEndpoint::Lower))?;
            let level = hydraulic(sources.panel_level(LEVEL))?;
            let raised = hydraulic(paired_residual::endpoint_at_level(
                sources,
                &rows,
                &query,
                PhysicalEndpoint::Lower,
                &level,
            ))?;
            let grouped_bits = frozen::components(&grouped.retained)?;
            let initial_bits = frozen::components(&initial.retained)?;
            let raised_bits = frozen::components(&raised.retained)?;
            for row in 0..n {
                let expected = if row == 1 {
                    &raised_bits
                } else {
                    &initial_bits
                };
                assert_eq!(grouped_bits[row], expected[row], "grouped row {row}");
            }
            let initial = frozen::endpoints(&initial.physical)?;
            let raised = frozen::endpoints(&raised.physical)?;
            for row in 0..n {
                eprintln!(
                    "WET_STAR_RESIDUAL\trow={row}\tlevel0={:?}\tlevel{LEVEL}={:?}",
                    initial[row], raised[row]
                );
                assert!(width(raised[row]) < width(initial[row]));
            }
            Ok(())
        })
    }

    fn run<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        mirror::<R>(client)?;
        corpus::<R>(client)?;
        grouped::<R>(client)
    }

    #[test]
    fn configured_cpu_wet_star_levels_narrow_certified_star_rows() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        run::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_wet_star_levels_narrow_certified_star_rows() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
