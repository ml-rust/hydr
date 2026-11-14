use super::super::super::{
    point_candidates::tests::{fixtures, flux_fixtures},
    point_flux_types::PointFluxStatus,
    point_root_session::PointRootSession,
};
use super::{chart_test_support, frozen, hydraulic, support};
use crate::channel::flow::surveyed::error::{HydraulicError, PrismaticRiemannRejection};
use frozen::Result;
use numr::{dtype::DType, runtime::Runtime};
use std::collections::BTreeMap;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let executions = fixtures::executions();
    let references = flux_fixtures::REFERENCES
        .iter()
        .map(|r| (r.id, r))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(executions.len(), 70);
    assert_eq!(references.len(), 70);
    let mut outcomes = [0_usize; 4];
    let mut evaluated = 0;
    let mut caps = 0;
    let mut work_limits = 0;
    for (config_index, original) in fixtures::CONFIGS.iter().enumerate() {
        let rows = executions
            .iter()
            .filter(|e| e.config == config_index)
            .collect::<Vec<_>>();
        let ids = rows.iter().map(|e| e.id).collect::<Vec<_>>();
        let config = original.build();
        chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
            let parents = (0..rows.len())
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let parents = support::rows::<R>(client, &parents)?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            let result = session.select_point_fluxes(&parents);
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
            let mut result = hydraulic(result)?;
            let initial_status = result.status.try_to_vec::<i64>()?;
            let initial_owners = result.candidates.invocation_owner.try_to_vec::<i64>()?;
            let blocked = initial_owners
                .iter()
                .zip(&initial_status)
                .filter_map(|(owner, status)| {
                    (*status == PointFluxStatus::PendingPositiveArea as i64).then_some(*owner)
                })
                .collect::<Vec<_>>();
            if !blocked.is_empty() {
                let requested = support::rows::<R>(client, &blocked)?;
                let support = hydraulic(session.advance_point_flux_positive_support(&requested))?;
                hydraulic(
                    session
                        .invocation
                        .check(&support.candidates.invocation_witness),
                )?;
                eprintln!(
                    "POINT_FLUX_POSITIVE_SUPPORT\towners={blocked:?}\toutcomes={:?}\tprogress={:?}\tattempts={:?}\tadditional_points={}",
                    support.status.try_to_vec::<i64>()?,
                    support.progressed.try_to_vec::<f64>()?,
                    support.attempts.try_to_vec::<i64>()?,
                    support.additional_points
                );
                assert_eq!(support.callback_points, session.callback_points());
                assert_eq!(support.point_limit, config.root.max_points);
                result = hydraulic(session.select_point_fluxes(&parents))?;
            }
            super::diagnostics::check(&session, &result)?;
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
                    "POINT_FLUX_ROW\t{}\tstatus={}\tphysical_q={:?}\tphysical_j={:?}\tjoint_q={:?}\tjoint_j={:?}\tpoints={}",
                    row.id,
                    status[i],
                    water[i],
                    momentum[i],
                    joint_q[i],
                    joint_j[i],
                    result.callback_points
                );
                if reference.water.is_none() || reference.momentum.is_none() {
                    assert!(
                        row.outcome == "CapExceeded" || row.outcome == "physical_cap_rejection"
                    );
                    assert_eq!(status[i], PointFluxStatus::PendingCandidates as i64);
                    caps += 1;
                    continue;
                }
                assert_ne!(
                    status[i],
                    PointFluxStatus::PendingCandidates as i64,
                    "{} loses existing candidate coverage",
                    row.id
                );
                if status[i] == PointFluxStatus::PendingPositiveArea as i64 {
                    let owner = i64::try_from(i)?;
                    for candidate in &result.candidates.candidates {
                        if let super::super::super::point_candidate_types::PointCandidate::Fan(
                            fan,
                        ) = candidate
                        {
                            let certificate = &fan.certificate;
                            let owners = certificate.invocation_owner.try_to_vec::<i64>()?;
                            if let Some(position) = owners.iter().position(|value| *value == owner)
                            {
                                let area = frozen::endpoints(&certificate.query.area)?;
                                let parameter =
                                    frozen::endpoints(&certificate.query.query.parameter)?;
                                let scale = frozen::components(&certificate.query.query.scale)?;
                                let depth = frozen::endpoints(&certificate.depth)?;
                                let normalized = frozen::endpoints(&certificate.normalized_depth)?;
                                let physical_status = certificate.status.try_to_vec::<i64>()?;
                                let depth_status = certificate.depth_status.try_to_vec::<i64>()?;
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
                contains(
                    water[i],
                    reference.water.ok_or("Flux water truth absent")?,
                    row.id,
                );
                contains(
                    momentum[i],
                    reference.momentum.ok_or("Flux momentum truth absent")?,
                    row.id,
                );
                if reference.positive_water_required {
                    let owner = i64::try_from(i)?;
                    let mut positive = false;
                    for point in &result.selected {
                        let owners = point.invocation_owner.try_to_vec::<i64>()?;
                        let bounds =
                            hydraulic(crate::channel::flow::surveyed::error::checked::<R, _>(
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
                        for point in &result.selected {
                            let owners = point.invocation_owner.try_to_vec::<i64>()?;
                            if let Some(position) = owners.iter().position(|value| *value == owner)
                            {
                                let pressure = point
                                    .pressure
                                    .as_ref()
                                    .ok_or("Positive quantum water has no pressure certificate")?;
                                pressure_bounds =
                                    Some(frozen::endpoints(&pressure.evaluation)?[position]);
                            }
                        }
                        pressure_bounds
                            .ok_or("Declared quantum pressure has no selected certificate")?
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
                if status[i] == PointFluxStatus::FluxBudgetExceeded as i64 {
                    eprintln!(
                        "POINT_FLUX_NEEDS_CONTINUATION\t{}\tjoint_q_width={:e}\tjoint_j_width={:e}",
                        row.id,
                        joint_q[i][1] - joint_q[i][0],
                        joint_j[i][1] - joint_j[i][0]
                    );
                }
                outcomes[usize::try_from(status[i])?] += 1;
                evaluated += 1;
            }
            assert_eq!(session.callback_points(), work.snapshot().residual_points);
            assert!(session.callback_points() <= config.root.max_points);
            Ok(())
        })?;
    }
    assert_eq!([evaluated, caps, work_limits], [66, 3, 1]);
    eprintln!(
        "POINT_FLUX_SUMMARY\tevaluated={evaluated}\tcap={caps}\twork={work_limits}\tstatuses={outcomes:?}"
    );
    Ok(())
}

pub(super) fn contains(actual: [f64; 2], expected: [u64; 2], label: &str) {
    let truth = expected.map(f64::from_bits);
    assert!(
        actual[0] <= truth[0] && truth[1] <= actual[1],
        "{label}: {actual:?} excludes {truth:?}"
    );
}
