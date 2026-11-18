use super::{
    point_flux_association,
    point_flux_types::{PointFluxKind, ResolvedPointFluxes},
    point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
    point_riemann_rejection,
    point_root_session::{PointRootSession, invalid},
    point_sources::PointSourceParts,
    source_sonic_types::SonicSide,
};
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig,
        error::{Arithmetic, HydraulicResult, checked},
        work::WorkLedger,
    },
    section::PrismaticSectionBatch,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::{CertifiedExpansion, Interval},
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

/// Resolves every face from four exact conserved points, or rejects the whole call.
pub(crate) fn solve<R>(
    section: &PrismaticSectionBatch<R>,
    owners: &Tensor<R>,
    inputs: [&CertifiedExpansion<R>; 4],
    config: &PrismaticRiemannConfig,
    work: &WorkLedger,
) -> HydraulicResult<PrismaticPointRiemannResult<R>>
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
    config.check()?;
    let parts = PointSourceParts::capture(section, owners, inputs, config)?;
    let c = parts.geometry.client();
    let count = parts.count();
    let fluxes = if count == 0 {
        checked::<R, _>(c, "point Riemann empty faces", Faces::empty)?
    } else {
        let rows = parts.sequential_rows()?;
        let mut pair = parts.pair(config, work, &rows)?;
        let mut session = PointRootSession::new(&mut pair, &rows, config.root.max_points)?;
        let resolved = session.resolve_point_fluxes(&rows)?;
        point_riemann_rejection::check(&session, &resolved)?;
        checked::<R, _>(c, "point Riemann selected traces", |a| {
            Faces::scatter(a, count, &resolved)
        })?
    };
    let input = parts.velocity.input();
    checked::<R, _>(c, "point Riemann result", |a| {
        let copy = |a: &mut Arithmetic<'_, R>, value: &CertifiedExpansion<R>| {
            a.expansion_bounds(value.high(), value.low(), value.abs_error())
        };
        let owners = Tensor::empty(input.owners.shape(), DType::I64, c.device())?;
        c.copy_into(&owners, &input.owners)?;
        Ok(PrismaticPointRiemannResult {
            owners,
            left_area: copy(a, &input.left_area)?,
            left_discharge: copy(a, &input.left_discharge)?,
            right_area: copy(a, &input.right_area)?,
            right_discharge: copy(a, &input.right_discharge)?,
            water_flux: fluxes.water,
            momentum_flux: fluxes.momentum,
            trace_area: fluxes.area,
            trace_discharge: fluxes.discharge,
            trace_kind: fluxes.kind,
            physical_water_flux: fluxes.physical_water,
            physical_momentum_flux: fluxes.physical_momentum,
            water_flux_bound: fluxes.joint_water,
            momentum_flux_bound: fluxes.joint_momentum,
            wave_speed: fluxes.wave_speed,
            work: work.snapshot(),
        })
    })
}

/// Query-order fields of the selected trace and the face bounds.
struct Faces<R: Runtime<DType = DType>> {
    water: CertifiedExpansion<R>,
    momentum: CertifiedExpansion<R>,
    area: CertifiedExpansion<R>,
    discharge: CertifiedExpansion<R>,
    kind: Tensor<R>,
    physical_water: Interval<R>,
    physical_momentum: Interval<R>,
    joint_water: Interval<R>,
    joint_momentum: Interval<R>,
    wave_speed: Interval<R>,
}

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
    fn empty(a: &mut Arithmetic<'_, R>) -> HydraulicResult<Self> {
        let c = a.client();
        let zero = Tensor::zeros(&[0], DType::F64, c.device())?;
        let point = |a: &mut Arithmetic<'_, R>| a.expansion_bounds(&zero, &zero, &zero);
        let interval = |a: &mut Arithmetic<'_, R>| a.bounds(&zero, &zero);
        Ok(Self {
            water: point(a)?,
            momentum: point(a)?,
            area: point(a)?,
            discharge: point(a)?,
            kind: Tensor::zeros(&[0], DType::I64, c.device())?,
            physical_water: interval(a)?,
            physical_momentum: interval(a)?,
            joint_water: interval(a)?,
            joint_momentum: interval(a)?,
            wave_speed: interval(a)?,
        })
    }

    /// Writes each face's owning selected point at its invocation owner.
    fn scatter(
        a: &mut Arithmetic<'_, R>,
        count: usize,
        resolved: &ResolvedPointFluxes<R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let association = point_flux_association::inverse(a, count, &resolved.invocation_owner)?;
        let zeros = || Tensor::zeros(&[count], DType::F64, c.device());
        // Components of the water, momentum, trace area and trace discharge points.
        let mut points = Vec::with_capacity(4);
        for _ in 0..4 {
            points.push([zeros()?, zeros()?, zeros()?]);
        }
        let mut kind = Tensor::full_scalar(&[count], DType::I64, -1., c.device())?;
        let mut filled = zeros()?;
        for (index, selection) in resolved.selections.iter().enumerate() {
            let owning = Tensor::full_scalar(&[], DType::I64, index as f64, c.device())?;
            for point in &selection.selected {
                let positions =
                    point_flux_association::positions(a, &association, &point.invocation_owner)?;
                let owned = c.eq(
                    &c.index_select(&resolved.selection, 0, &positions)?,
                    &owning,
                )?;
                let rows = c.nonzero(&owned)?.flatten()?;
                if rows.numel() == 0 {
                    continue;
                }
                let target = c.index_select(&point.invocation_owner, 0, &rows)?;
                let values = [
                    &point.discharge,
                    &point.momentum,
                    &point.area,
                    &point.discharge,
                ];
                for (components, value) in points.iter_mut().zip(values) {
                    for (component, source) in
                        components
                            .iter_mut()
                            .zip([value.high(), value.low(), value.abs_error()])
                    {
                        let selected = c.index_select(source, 0, &rows)?;
                        *component = c.index_put(component, 0, &target, &selected)?;
                    }
                }
                let code = Tensor::full_scalar(
                    target.shape(),
                    DType::I64,
                    trace_code(point.kind) as i64 as f64,
                    c.device(),
                )?;
                kind = c.index_put(&kind, 0, &target, &code)?;
                let ones = Tensor::ones(target.shape(), DType::F64, c.device())?;
                filled = c.index_put(&filled, 0, &target, &ones)?;
            }
        }
        let zero = Tensor::zeros(&[], DType::F64, c.device())?;
        if c.nonzero(&c.eq(&filled, &zero)?)?.numel() != 0 {
            return Err(invalid("accepted face lacks its owning selected trace"));
        }
        let owner = &resolved.invocation_owner;
        let interval = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| {
            let lower = c.index_put(&zeros()?, 0, owner, value.lower())?;
            let upper = c.index_put(&zeros()?, 0, owner, value.upper())?;
            HydraulicResult::Ok(a.bounds(&lower, &upper)?)
        };
        let physical_water = interval(a, &resolved.physical_water)?;
        let physical_momentum = interval(a, &resolved.physical_momentum)?;
        let joint_water = interval(a, &resolved.joint_water)?;
        let joint_momentum = interval(a, &resolved.joint_momentum)?;
        let wave_speed = interval(a, &resolved.wave_speed_m_per_s)?;
        let mut expansions = Vec::with_capacity(4);
        for components in &points {
            expansions.push(a.expansion_bounds(&components[0], &components[1], &components[2])?);
        }
        let [water, momentum, area, discharge]: [CertifiedExpansion<R>; 4] = expansions
            .try_into()
            .map_err(|_| invalid("selected trace expansion count differs from four"))?;
        Ok(Self {
            water,
            momentum,
            area,
            discharge,
            kind,
            physical_water,
            physical_momentum,
            joint_water,
            joint_momentum,
            wave_speed,
        })
    }
}

fn trace_code(kind: PointFluxKind) -> PrismaticPointTraceKind {
    match kind {
        PointFluxKind::Exterior(SonicSide::Left) => PrismaticPointTraceKind::LeftState,
        PointFluxKind::Fan(SonicSide::Left) => PrismaticPointTraceKind::LeftFan,
        PointFluxKind::Star => PrismaticPointTraceKind::Star,
        PointFluxKind::Fan(SonicSide::Right) => PrismaticPointTraceKind::RightFan,
        PointFluxKind::Exterior(SonicSide::Right) => PrismaticPointTraceKind::RightState,
        PointFluxKind::Dry => PrismaticPointTraceKind::Dry,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        corpus_coverage::Coverage,
        point_candidates::fixture::{fixtures, flux_fixtures},
        point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
        source_fixture_support::{self as frozen, Result, hydraulic},
        wet_depth::fixture as support,
    };
    use crate::channel::{
        flow::surveyed::{
            PrismaticRiemannConfig, PrismaticRiemannError, PrismaticRiemannRejection,
            error::checked,
        },
        section::PrismaticSectionBatch,
    };
    use numr::{
        dtype::DType,
        runtime::{Runtime, RuntimeClient},
        tensor::Tensor,
    };
    use solvr::common::arithmetic::CertifiedExpansion;
    use std::collections::BTreeMap;

    /// Runs one batch of frozen source cases through the public point entry.
    fn call<R>(
        client: &R::Client,
        section: &PrismaticSectionBatch<R>,
        ids: &[&str],
        config: &PrismaticRiemannConfig,
    ) -> Result<std::result::Result<PrismaticPointRiemannResult<R>, PrismaticRiemannError>>
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let cases = frozen::cases();
        let selected = ids
            .iter()
            .map(|id| {
                cases
                    .iter()
                    .find(|case| case.id == *id)
                    .ok_or("Missing frozen point source")
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let owners = Tensor::from_slice(
            &selected
                .iter()
                .map(|case| i64::try_from(case.owner))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            &[ids.len()],
            client.device(),
        )?;
        let inputs = (0..4)
            .map(|field| {
                frozen::expansion::<R>(
                    client,
                    &selected
                        .iter()
                        .map(|case| case.inputs[field])
                        .collect::<Vec<_>>(),
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(section.riemann_flux_points_tensor(
            &owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3], config,
        ))
    }

    fn point<R>(client: &R::Client, value: &CertifiedExpansion<R>) -> Result<Vec<[f64; 2]>>
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let interval = hydraulic(checked::<R, _>(client, "selected point bounds", |a| {
            Ok(a.expansion_interval(value)?)
        }))?;
        frozen::endpoints(&interval)
    }

    fn contains(actual: [f64; 2], truth: [f64; 2], label: &str) {
        assert!(
            actual[0] <= truth[0] && truth[1] <= actual[1],
            "{label}: {actual:?} excludes {truth:?}"
        );
    }

    fn run<R>(client: &R::Client, coverage: Coverage) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let executions = coverage.select(fixtures::executions(), |row| (row.outcome, row.physical));
        let references = flux_fixtures::REFERENCES
            .iter()
            .map(|r| (r.id, r))
            .collect::<BTreeMap<_, _>>();
        let section = frozen::batch::<R>(client)?;
        let mut counts = [0_usize; 3];
        for (config_index, original) in fixtures::CONFIGS.iter().enumerate() {
            let config = original.build();
            let rows = executions
                .iter()
                .filter(|row| row.config == config_index)
                .collect::<Vec<_>>();
            let mut accepted = Vec::new();
            for row in rows {
                let reference = references
                    .get(row.id)
                    .ok_or("Execution lacks independent full flux truth")?;
                let expected = if row.outcome == "WorkLimit" {
                    Some(PrismaticRiemannRejection::WorkLimit)
                } else if reference.water.is_none() || reference.momentum.is_none() {
                    Some(PrismaticRiemannRejection::CapExceeded)
                } else {
                    None
                };
                let Some(expected) = expected else {
                    accepted.push((row.id, *reference));
                    continue;
                };
                match call::<R>(client, &section, &[row.id], &config)? {
                    Err(PrismaticRiemannError::Rejected {
                        reason,
                        query,
                        section_owner,
                        ..
                    }) => {
                        assert_eq!(reason, expected, "{}", row.id);
                        if reason == PrismaticRiemannRejection::CapExceeded {
                            let owner = frozen::cases()
                                .iter()
                                .find(|case| case.id == row.id)
                                .ok_or("Missing frozen point source")?
                                .owner;
                            assert_eq!(query, Some(0), "{}", row.id);
                            assert_eq!(section_owner, Some(owner), "{}", row.id);
                            counts[1] += 1;
                        } else {
                            counts[2] += 1;
                        }
                    }
                    other => {
                        return Err(format!(
                            "{} returns {:?} instead of {expected:?}",
                            row.id,
                            other.err()
                        )
                        .into());
                    }
                }
            }
            if accepted.is_empty() {
                continue;
            }
            let ids = accepted.iter().map(|(id, _)| *id).collect::<Vec<_>>();
            let result = call::<R>(client, &section, &ids, &config)??;
            let physical = [
                frozen::endpoints(result.physical_water_flux_m3_per_s())?,
                frozen::endpoints(result.physical_momentum_flux_m4_per_s2())?,
            ];
            let bound = [
                frozen::endpoints(result.water_flux_bound_m3_per_s())?,
                frozen::endpoints(result.momentum_flux_bound_m4_per_s2())?,
            ];
            let selected = [
                point::<R>(client, result.water_flux_m3_per_s())?,
                point::<R>(client, result.momentum_flux_m4_per_s2())?,
            ];
            let discharge = point::<R>(client, result.trace_discharge_m3_per_s())?;
            let kinds = result.trace_kind().try_to_vec::<i64>()?;
            let speeds = frozen::endpoints(result.wave_speed_m_per_s())?;
            assert_eq!(result.owners().numel(), ids.len());
            for (i, (id, reference)) in accepted.iter().enumerate() {
                let truth = [
                    reference.water.ok_or("Flux water truth absent")?,
                    reference.momentum.ok_or("Flux momentum truth absent")?,
                ]
                .map(|bits| bits.map(f64::from_bits));
                let mirror = frozen::exact_mirror(id)?;
                for water in [physical[0][i], bound[0][i]] {
                    frozen::water_enclosed(water, truth[0], mirror, id);
                }
                if mirror {
                    assert_eq!(
                        selected[0][i].map(f64::to_bits),
                        [0, 0],
                        "{id} selected water"
                    );
                }
                contains(bound[0][i], selected[0][i], id);
                contains(physical[1][i], truth[1], id);
                contains(bound[1][i], truth[1], id);
                contains(bound[1][i], selected[1][i], id);
                assert_eq!(
                    discharge[i], selected[0][i],
                    "{id} water flux differs from its trace"
                );
                assert!(
                    PrismaticPointTraceKind::from_code(kinds[i]).is_some(),
                    "{id} carries trace code {}",
                    kinds[i]
                );
                assert!(
                    speeds[i][0] <= speeds[i][1],
                    "{id} wave speeds {:?}",
                    speeds[i]
                );
                counts[0] += 1;
            }
        }
        if coverage.is_full() {
            assert_eq!(counts, [66, 3, 1]);
        }
        Ok(())
    }

    mod exterior {
        use super::super::super::{
            point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
            source_fixture_support::{self as frozen, Result},
            wet_depth::fixture as support,
        };
        use crate::channel::{
            flow::surveyed::PrismaticRiemannConfig,
            section::{
                CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticSectionBatch,
            },
        };
        use numr::{dtype::DType, runtime::Runtime};

        /// Below-cap offsets of the two face areas.
        const OFFSETS: [f64; 2] = [2.5e-17, 8e-17];

        /// The rounded wet-fan section: three bands up to a 3 m cap of 14 m² and 8 m top width.
        /// At the cap the hydrostatic integral is 17 m³.
        fn rounded_wet_fan<R>(client: &R::Client) -> Result<PrismaticSectionBatch<R>>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let points = [
                (-4.0, 3.0),
                (-2.5, 2.0),
                (-2.0, 1.0),
                (-1.0, 0.0),
                (1.0, 0.0),
                (2.0, 1.0),
                (2.5, 2.0),
                (4.0, 3.0),
            ];
            let definition = CrossSectionDefinition {
                id: "rounded_wet_fan".into(),
                vertical_datum: "Exact dyadic datum".into(),
                points: points
                    .into_iter()
                    .map(|(station_m, elevation_m)| CrossSectionPoint {
                        station_m,
                        elevation_m,
                    })
                    .collect(),
                maximum_stage_m: 3.0,
                profile_provenance: "Independent dyadic proof".into(),
            };
            Ok(
                CrossSectionBatch::<R>::new(client.clone(), vec![definition])?
                    .certify_prismatic()?,
            )
        }

        /// Selected momentum components of equal-state faces, one face per offset, in that order.
        ///
        /// Each face holds A = 14 - d and Q = 2A on both sides. The velocity 2 m/s exceeds the
        /// celerity √1.75 m/s, so both waves move right and the trace is the left state.
        fn momentum<R>(
            client: &R::Client,
            section: &PrismaticSectionBatch<R>,
            offsets: &[f64],
        ) -> Result<Vec<[u64; 3]>>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let owners = support::rows::<R>(client, &vec![0; offsets.len()])?;
            let area = offsets
                .iter()
                .map(|d| [14.0_f64.to_bits(), (-d).to_bits(), 0])
                .collect::<Vec<_>>();
            let discharge = offsets
                .iter()
                .map(|d| [28.0_f64.to_bits(), (-2.0 * d).to_bits(), 0])
                .collect::<Vec<_>>();
            let area = frozen::expansion::<R>(client, &area, true)?;
            let discharge = frozen::expansion::<R>(client, &discharge, true)?;
            let config = PrismaticRiemannConfig {
                gravity_m_per_s2: 1.0,
                ..PrismaticRiemannConfig::default()
            };
            let result = section
                .riemann_flux_points_tensor(&owners, &area, &discharge, &area, &discharge, &config)
                .map_err(|error| format!("supercritical wet-fan faces reject: {error:?}"))?;
            let kinds = result.trace_kind().try_to_vec::<i64>()?;
            assert_eq!(
                kinds,
                vec![PrismaticPointTraceKind::LeftState as i64; offsets.len()]
            );
            frozen::components(result.momentum_flux_m4_per_s2())
        }

        fn left_state<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let section = rounded_wet_fan::<R>(client)?;
            let forward = momentum::<R>(client, &section, &OFFSETS)?;
            let reverse = momentum::<R>(client, &section, &[OFFSETS[1], OFFSETS[0]])?;
            for (index, d) in OFFSETS.into_iter().enumerate() {
                let [high, low, _] = forward[index].map(f64::from_bits);
                // Q²/A = 4A = 56 - 4d exactly. I1(14 - d) = 17 - 1.75d + O(d²), and g = 1.
                // The high part lies within one ulp of 73, so high - 73 is exact.
                let difference = (high - 73.0) + low + 4.0 * d;
                let expected = -1.75 * d;
                assert!(
                    (difference - expected).abs() <= 1e-30,
                    "d={d:e}, difference={difference:e}, expected={expected:e}"
                );
                // Equal faces give equal bits in either batch order.
                assert_eq!(forward[index], reverse[1 - index], "d={d:e}");
            }
            Ok(())
        }

        /// One rest face with the two-part areas 14 - dL and 14 - dR and zero discharge.
        fn rest_face<R>(
            client: &R::Client,
            section: &PrismaticSectionBatch<R>,
            [left, right]: [f64; 2],
        ) -> Result<PrismaticPointRiemannResult<R>>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let owners = support::rows::<R>(client, &[0])?;
            let area = |d: f64| {
                frozen::expansion::<R>(client, &[[14.0_f64.to_bits(), (-d).to_bits(), 0]], true)
            };
            let zero = frozen::expansion::<R>(client, &[[0, 0, 0]], true)?;
            let config = PrismaticRiemannConfig {
                gravity_m_per_s2: 1.0,
                ..PrismaticRiemannConfig::default()
            };
            section
                .riemann_flux_points_tensor(
                    &owners,
                    &area(left)?,
                    &zero,
                    &area(right)?,
                    &zero,
                    &config,
                )
                .map_err(|error| {
                    format!("rest star face dL={left:e} dR={right:e} rejects: {error:?}").into()
                })
        }

        /// Rest faces between two cap deficits select the star state to second order in the jump.
        ///
        /// The deeper side sends a rarefaction and the shallower side a shock, so water flows
        /// toward the shallower side. With c = √1.75 the linearized star state is
        /// A* = 14 - (dL + dR)/2, Q* = c(dR - dL)/2, and M* = 17 - 1.75(dL + dR)/2.
        /// The dropped terms are below 1e-33.
        fn rest_star<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let section = rounded_wet_fan::<R>(client)?;
            let celerity = 1.75_f64.sqrt();
            for [left, right] in [OFFSETS, [OFFSETS[1], OFFSETS[0]]] {
                let result = rest_face::<R>(client, &section, [left, right])?;
                let kinds = result.trace_kind().try_to_vec::<i64>()?;
                assert_eq!(kinds, vec![PrismaticPointTraceKind::Star as i64]);
                let [mh, ml, _] =
                    frozen::components(result.momentum_flux_m4_per_s2())?[0].map(f64::from_bits);
                let [qh, ql, _] =
                    frozen::components(result.water_flux_m3_per_s())?[0].map(f64::from_bits);
                // The high part lies within one ulp of 17, so mh - 17 is exact.
                let momentum = ((mh - 17.0) + ml) + 1.75 * (left + right) / 2.0;
                let discharge = (qh - celerity * (right - left) / 2.0) + ql;
                let work = result.work();
                eprintln!(
                    "POINT_RIEMANN_REST_STAR\tdL={left:e}\tdR={right:e}\tmomentum_error={momentum:e}\tdischarge_error={discharge:e}\trefinements={}\troot_iterations={}\tresidual_points={}",
                    work.refinements, work.root_iterations, work.residual_points
                );
                assert!(
                    momentum.abs() <= 1e-20,
                    "dL={left:e} dR={right:e} momentum error {momentum:e}"
                );
                assert!(
                    discharge.abs() <= 1e-20,
                    "dL={left:e} dR={right:e} discharge error {discharge:e}"
                );
            }
            Ok(())
        }

        #[test]
        fn configured_cpu_public_point_rest_star_selects_the_jump_scaled_state() -> Result {
            use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
            rest_star::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn actual_cuda_public_point_rest_star_selects_the_jump_scaled_state() -> Result {
            use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
            rest_star::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn configured_cpu_public_point_left_state_momentum_uses_two_part_pressure() -> Result {
            use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
            left_state::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn actual_cuda_public_point_left_state_momentum_uses_two_part_pressure() -> Result {
            use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
            left_state::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
        }
    }

    mod sharpening {
        use super::super::super::{
            point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
            source_fixture_support::{self as frozen, Result},
            wet_depth::fixture as support,
        };
        use crate::channel::{
            flow::surveyed::PrismaticRiemannConfig,
            section::{
                CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticSectionBatch,
            },
        };
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedBisectionOptions;

        /// The low part of the right area at a rest face just inside a dam-break rarefaction head.
        /// The left area is exactly 12 m², so the jump between the sources is near 4.6e-97 m².
        const RIGHT_LOW: u64 = 0xabef84666e2af06a;

        /// The multiband section: three bands up to a 4 m cap.
        fn multiband<R>(client: &R::Client) -> Result<PrismaticSectionBatch<R>>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let points = [
                (0.0, 4.0),
                (1.0, 2.0),
                (3.0, 1.0),
                (4.0, 0.0),
                (5.0, 0.0),
                (6.0, 1.0),
                (8.0, 2.0),
                (9.0, 4.0),
            ];
            let definition = CrossSectionDefinition {
                id: "multiband".into(),
                vertical_datum: "Exact dyadic datum".into(),
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
            Ok(
                CrossSectionBatch::<R>::new(client.clone(), vec![definition])?
                    .certify_prismatic()?,
            )
        }

        /// The reach Riemann controls.
        fn config() -> PrismaticRiemannConfig {
            PrismaticRiemannConfig {
                water_flux_atol_m3_per_s: 1e-12,
                momentum_flux_atol_m4_per_s2: 1e-12,
                flux_rtol: 1e-12,
                root: CertifiedBisectionOptions {
                    atol: 1e-15,
                    rtol: 1e-15,
                    ..CertifiedBisectionOptions::default()
                },
                ..PrismaticRiemannConfig::default()
            }
        }

        /// One rest face with left area 12 and right area 12 plus `low`, both at zero discharge.
        fn rest_face<R>(
            client: &R::Client,
            section: &PrismaticSectionBatch<R>,
            low: u64,
        ) -> Result<PrismaticPointRiemannResult<R>>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let owners = support::rows::<R>(client, &[0])?;
            let left = frozen::expansion::<R>(client, &[[12.0_f64.to_bits(), 0, 0]], true)?;
            let right = frozen::expansion::<R>(client, &[[12.0_f64.to_bits(), low, 0]], true)?;
            let zero = frozen::expansion::<R>(client, &[[0, 0, 0]], true)?;
            section
                .riemann_flux_points_tensor(&owners, &left, &zero, &right, &zero, &config())
                .map_err(|error| format!("rest face low={low:#x} rejects: {error:?}").into())
        }

        /// Root rounds a face may take when its bracket halves from the area scale down to the
        /// two-part rounding of that area.
        const AREA_ROUNDS: usize = 128;

        /// A rest face whose jump lies far below the two-part rounding of its area stops sharpening
        /// at that rounding, so its rounds stay within `AREA_ROUNDS`.
        fn tiny_jump<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let section = multiband::<R>(client)?;
            let result = rest_face::<R>(client, &section, RIGHT_LOW)?;
            let kinds = result.trace_kind().try_to_vec::<i64>()?;
            assert_eq!(kinds, vec![PrismaticPointTraceKind::Star as i64]);
            let work = result.work();
            eprintln!(
                "POINT_RIEMANN_TINY_JUMP\troot_iterations={}\tmaximum_root_iterations={}\tresidual_points={}",
                work.root_iterations, work.maximum_root_iterations, work.residual_points
            );
            assert!(
                work.maximum_root_iterations <= AREA_ROUNDS,
                "rounds={}",
                work.maximum_root_iterations
            );
            Ok(())
        }

        #[test]
        fn configured_cpu_public_point_tiny_jump_rest_face_sharpens_to_the_area_rounding() -> Result
        {
            use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
            tiny_jump::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn actual_cuda_public_point_tiny_jump_rest_face_sharpens_to_the_area_rounding() -> Result {
            use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
            tiny_jump::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
        }
    }

    mod star {
        use super::super::super::{
            corpus_coverage::Coverage,
            point_output::{PrismaticPointRiemannResult, PrismaticPointTraceKind},
            source_fixture_support::{self as frozen, Result, hydraulic},
            wet_depth::fixture as support,
            wet_star_levels::tests::{MIRROR, WALLS, integer_trapezoid, trapezoid},
        };
        use crate::channel::{
            flow::surveyed::{PrismaticRiemannConfig, PrismaticRiemannError, error::checked},
            section::PrismaticSectionBatch,
        };
        use numr::{dtype::DType, runtime::Runtime};

        /// The reach allowances: water and momentum atol and the relative allowance are all 1e-12.
        const ALLOWANCE: f64 = 1e-12;

        /// A gate-like star face: deep flow upstream of a shallow reach with the same discharge.
        const GATE: [f64; 4] = [24.0, 12.0, 12.0, 12.0];

        /// A dam break onto a shallow pool. The left rarefaction straddles the interface.
        const FAN: [f64; 4] = [24.0, 0.0, 1.0, 0.0];

        type Outcome<R> =
            std::result::Result<PrismaticPointRiemannResult<R>, PrismaticRiemannError>;

        fn config() -> PrismaticRiemannConfig {
            PrismaticRiemannConfig {
                water_flux_atol_m3_per_s: ALLOWANCE,
                momentum_flux_atol_m4_per_s2: ALLOWANCE,
                flux_rtol: ALLOWANCE,
                ..PrismaticRiemannConfig::default()
            }
        }

        /// One public call with every face on section 0.
        fn call<R>(
            client: &R::Client,
            section: &PrismaticSectionBatch<R>,
            states: &[[f64; 4]],
        ) -> Result<Outcome<R>>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let owners = support::rows::<R>(client, &vec![0; states.len()])?;
            let inputs = (0..4)
                .map(|field| {
                    frozen::expansion::<R>(
                        client,
                        &states
                            .iter()
                            .map(|state| [state[field].to_bits(), 0, 0])
                            .collect::<Vec<_>>(),
                        true,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(section.riemann_flux_points_tensor(
                &owners,
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &inputs[3],
                &config(),
            ))
        }

        fn within(value: [f64; 2]) -> bool {
            value[1] - value[0] <= ALLOWANCE + ALLOWANCE * value[0].abs().max(value[1].abs())
        }

        /// Accepted faces carry the expected trace kinds and flux bounds within the allowances.
        fn accepted<R>(
            client: &R::Client,
            section: &PrismaticSectionBatch<R>,
            label: &str,
            states: &[[f64; 4]],
            kinds: &[PrismaticPointTraceKind],
        ) -> Result<PrismaticPointRiemannResult<R>>
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let result = call::<R>(client, section, states)?
                .map_err(|error| format!("{label} rejects at the public entry: {error:?}"))?;
            let codes = result.trace_kind().try_to_vec::<i64>()?;
            let actual = codes
                .iter()
                .map(|code| PrismaticPointTraceKind::from_code(*code))
                .collect::<Vec<_>>();
            let expected = kinds.iter().copied().map(Some).collect::<Vec<_>>();
            assert_eq!(actual, expected, "{label} trace kinds");
            let water = frozen::endpoints(result.water_flux_bound_m3_per_s())?;
            let momentum = frozen::endpoints(result.momentum_flux_bound_m4_per_s2())?;
            for face in 0..states.len() {
                assert!(
                    within(water[face]),
                    "{label} face {face} water {:?}",
                    water[face]
                );
                assert!(
                    within(momentum[face]),
                    "{label} face {face} momentum {:?}",
                    momentum[face]
                );
            }
            let work = result.work();
            eprintln!(
                "POINT_RIEMANN_STAR\t{label}\trefinements={}\tpanels={}\tresidual_points={}",
                work.refinements, work.maximum_simpson_panels, work.residual_points
            );
            assert!(work.refinements <= config().max_refinements);
            assert!(work.maximum_simpson_panels <= config().max_simpson_panels);
            Ok(result)
        }

        /// The gate star face is over budget at the initial panels and accepted after star rounds.
        fn gate<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let section = trapezoid::<R>(client)?;
            let result = accepted::<R>(
                client,
                &section,
                "gate",
                &[GATE],
                &[PrismaticPointTraceKind::Star],
            )?;
            assert!(result.work().refinements >= 1);
            Ok(())
        }

        /// A star face and a sonic-fan face share their rounds, so the batch needs no more rounds
        /// than the slower face alone.
        fn shared_rounds<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let section = trapezoid::<R>(client)?;
            let star = accepted::<R>(
                client,
                &section,
                "gate",
                &[GATE],
                &[PrismaticPointTraceKind::Star],
            )?;
            let fan = accepted::<R>(
                client,
                &section,
                "fan",
                &[FAN],
                &[PrismaticPointTraceKind::LeftFan],
            )?;
            let both = accepted::<R>(
                client,
                &section,
                "gate and fan",
                &[GATE, FAN],
                &[
                    PrismaticPointTraceKind::Star,
                    PrismaticPointTraceKind::LeftFan,
                ],
            )?;
            let [star, fan] = [star.work().refinements, fan.work().refinements];
            assert!(star >= 1 && fan >= 1, "gate={star} fan={fan}");
            assert_eq!(both.work().refinements, star.max(fan));
            Ok(())
        }

        /// Exact mirror faces take the reflection identity at the public entry: zero water in every
        /// component, and a selected momentum inside its bound.
        fn mirror<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let level = trapezoid::<R>(client)?;
            let integer = integer_trapezoid::<R>(client)?;
            let faces = std::iter::once((&level, "mirror", MIRROR[0])).chain(
                WALLS
                    .into_iter()
                    .map(|(label, state)| (&integer, label, state)),
            );
            for (section, label, state) in faces {
                let result = accepted::<R>(
                    client,
                    section,
                    label,
                    &[state],
                    &[PrismaticPointTraceKind::Star],
                )?;
                for value in [
                    result.water_flux_m3_per_s(),
                    result.trace_discharge_m3_per_s(),
                ] {
                    assert_eq!(frozen::components(value)?, [[0; 3]], "{label}");
                }
                for water in [
                    result.water_flux_bound_m3_per_s(),
                    result.physical_water_flux_m3_per_s(),
                ] {
                    for value in [water.lower(), water.upper()] {
                        let bits = value.try_to_vec::<f64>()?.into_iter().map(f64::to_bits);
                        assert_eq!(bits.collect::<Vec<_>>(), [0], "{label}");
                    }
                }
                let bound = frozen::endpoints(result.momentum_flux_bound_m4_per_s2())?[0];
                let selected = hydraulic(checked::<R, _>(client, "mirror momentum point", |a| {
                    Ok(a.expansion_interval(result.momentum_flux_m4_per_s2())?)
                }))?;
                let selected = frozen::endpoints(&selected)?[0];
                assert!(
                    bound[0] <= selected[0] && selected[1] <= bound[1],
                    "{label} momentum {selected:?} outside {bound:?}"
                );
            }
            Ok(())
        }

        fn run<R>(client: &R::Client, coverage: Coverage) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            gate::<R>(client)?;
            if coverage.is_full() {
                shared_rounds::<R>(client)?;
                mirror::<R>(client)?;
            }
            Ok(())
        }

        #[test]
        fn configured_cpu_public_point_fluxes_refine_star_and_fan_faces() -> Result {
            use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
            run::<CpuRuntime>(&CpuClient::new(CpuDevice::new()), Coverage::Full)
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn actual_cuda_public_point_fluxes_refine_star_and_fan_faces() -> Result {
            use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
            run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
        }
    }

    #[test]
    fn configured_cpu_public_point_fluxes_enclose_frozen_references() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        let client = CpuClient::new(CpuDevice::new());
        run::<CpuRuntime>(&client, Coverage::Full)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_public_point_fluxes_enclose_frozen_references() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        run::<CudaRuntime>(&client, Coverage::Sample)
    }
}
