use super::super::super::{
    point_candidate_types::PointCandidate,
    point_flux_types::{PointFluxCertificates, PointFluxKind, PointFluxStatus},
    point_root_session::PointRootSession,
};
use super::{frozen, hydraulic, support};
use crate::channel::flow::surveyed::error::checked;
use frozen::Result;
use numr::{dtype::DType, runtime::Runtime};
use std::collections::BTreeMap;

pub(super) fn check<R>(
    session: &PointRootSession<'_, '_, R>,
    result: &PointFluxCertificates<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    hydraulic(
        session
            .invocation
            .check(&result.candidates.invocation_witness),
    )?;
    assert_eq!(result.callback_points, session.callback_points());
    selection(session, result)
}

/// Checks one selection, which can be older than the latest session call.
pub(super) fn selection<R>(
    session: &PointRootSession<'_, '_, R>,
    result: &PointFluxCertificates<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    hydraulic(
        session
            .invocation
            .check(&result.candidates.invocation_witness),
    )?;
    assert!(result.callback_points <= session.callback_points());
    assert_eq!(result.point_limit, session.point_limit());
    let requested = result.candidates.invocation_owner.try_to_vec::<i64>()?;
    let parents = result.candidates.parent_rows.try_to_vec::<i64>()?;
    let position = requested
        .iter()
        .enumerate()
        .map(|(i, owner)| (*owner, i))
        .collect::<BTreeMap<_, _>>();
    let original = session.sources.original_rows.try_to_vec::<i64>()?;
    let status = result.status.try_to_vec::<i64>()?;
    let physical_valid = result.physical_valid.try_to_vec::<f64>()?;
    let joint_valid = result.joint_valid.try_to_vec::<f64>()?;
    let physical_q = frozen::endpoints(&result.physical_water)?;
    let physical_j = frozen::endpoints(&result.physical_momentum)?;
    let joint_q = frozen::endpoints(&result.joint_water)?;
    let joint_j = frozen::endpoints(&result.joint_momentum)?;
    assert_eq!(status.len(), requested.len());
    for row in 0..requested.len() {
        let complete = status[row] == PointFluxStatus::Accepted as i64
            || status[row] == PointFluxStatus::FluxBudgetExceeded as i64;
        assert_eq!(physical_valid[row], if complete { 1. } else { 0. });
        assert_eq!(joint_valid[row], if complete { 1. } else { 0. });
        encloses(joint_q[row], physical_q[row]);
        encloses(joint_j[row], physical_j[row]);
    }
    let mut selected_counts = vec![0_usize; requested.len()];
    for point in &result.selected {
        let candidate = result
            .candidates
            .candidates
            .get(point.candidate)
            .ok_or("Selected candidate identity is absent")?;
        match (point.kind, candidate) {
            (PointFluxKind::Exterior(side), PointCandidate::Exterior(e)) => {
                assert_eq!(side, e.side)
            }
            (PointFluxKind::Star, PointCandidate::Star(_))
            | (PointFluxKind::Dry, PointCandidate::Dry(_)) => {}
            (PointFluxKind::Fan(side), PointCandidate::Fan(f)) => {
                assert_eq!(side, f.certificate.side)
            }
            _ => return Err("Selected point mixes candidate identities".into()),
        }
        let owners = point.invocation_owner.try_to_vec::<i64>()?;
        let selected_parent = point.parent_rows.try_to_vec::<i64>()?;
        let tags = point.query.query.original_rows.try_to_vec::<i64>()?;
        let area = frozen::components(&point.area)?;
        let discharge = frozen::components(&point.discharge)?;
        let velocity = frozen::components(&point.velocity)?;
        let momentum = frozen::components(&point.momentum)?;
        let area_evaluation = frozen::components(&point.area_evaluation)?;
        let q_evaluation = frozen::components(&point.discharge_evaluation)?;
        let q_proposal = frozen::components(&point.discharge_proposal)?;
        let q_contained = point.discharge_proposal_contained.try_to_vec::<f64>()?;
        let q_change = frozen::components(&point.discharge_selection_change)?;
        let invariant = frozen::components(&point.discharge_invariant_velocity)?;
        let q_physical = frozen::endpoints(&point.discharge_physical)?;
        let q_product = frozen::endpoints(&point.discharge_product)?;
        let u_arithmetic = frozen::components(&point.velocity_arithmetic)?;
        let j_arithmetic = frozen::components(&point.momentum_arithmetic)?;
        let (q_bounds, j_bounds, area_bounds, q_eval, j_pure) = hydraulic(checked::<R, _>(
            session.sources.geometry.client(),
            "selected flux diagnostic projections",
            |a| {
                Ok((
                    a.expansion_interval(&point.discharge)?,
                    a.expansion_interval(&point.momentum)?,
                    a.expansion_interval(&point.area)?,
                    a.expansion_interval(&point.discharge_evaluation)?,
                    a.expansion_interval(&point.momentum_arithmetic)?,
                ))
            },
        ))?;
        let q_bounds = frozen::endpoints(&q_bounds)?;
        let j_bounds = frozen::endpoints(&j_bounds)?;
        let area_bounds = frozen::endpoints(&area_bounds)?;
        let q_eval = frozen::endpoints(&q_eval)?;
        let j_pure = frozen::endpoints(&j_pure)?;
        let j_eval = frozen::endpoints(&point.momentum_evaluation)?;
        let depth = frozen::endpoints(&point.depth)?;
        for value in [
            &point.area_discrepancy,
            &point.area_projection,
            &point.discharge_discrepancy,
            &point.momentum_projection,
        ] {
            for bounds in frozen::endpoints(value)? {
                ordered(bounds);
            }
        }
        for row in 0..owners.len() {
            let output = *position
                .get(&owners[row])
                .ok_or("Selected owner is outside demand")?;
            selected_counts[output] += 1;
            assert_eq!(selected_parent[row], parents[output]);
            assert_eq!(tags[row], original[usize::try_from(selected_parent[row])?]);
            for value in [area[row], discharge[row], velocity[row], momentum[row]] {
                assert_eq!(value[2], 0);
            }
            assert_eq!(&area[row][..2], &area_evaluation[row][..2]);
            assert_eq!(&q_proposal[row][..2], &q_evaluation[row][..2]);
            assert_eq!(q_proposal[row][2], 0);
            if q_contained[row] == 1. {
                assert_eq!(discharge[row], q_proposal[row]);
            } else {
                assert_eq!(q_contained[row], 0.);
            }
            assert!(
                q_change[row]
                    .iter()
                    .all(|bits| f64::from_bits(*bits).is_finite())
            );
            assert!(
                invariant[row]
                    .iter()
                    .all(|bits| f64::from_bits(*bits).is_finite())
            );
            ordered(q_product[row]);
            encloses(q_eval[row], q_physical[row]);
            assert_eq!(&velocity[row][..2], &u_arithmetic[row][..2]);
            assert_eq!(&momentum[row][..2], &j_arithmetic[row][..2]);
            encloses(joint_q[output], q_bounds[row]);
            encloses(joint_q[output], q_physical[row]);
            encloses(joint_j[output], j_bounds[row]);
            encloses(joint_j[output], j_pure[row]);
            encloses(joint_j[output], j_eval[row]);
            if point.kind == PointFluxKind::Dry {
                assert_eq!(area_bounds[row], [0., 0.]);
                assert_eq!(depth[row], [0., 0.]);
            } else {
                assert!(area_bounds[row][0] > 0. && depth[row][0] > 0.);
            }
        }
        if let Some(pressure) = &point.pressure {
            let anchor = frozen::components(&pressure.source_anchor)?;
            let selected_anchor = frozen::components(&pressure.selected_anchor)?;
            let increment = frozen::components(&pressure.increment.retained_increment)?;
            let selected_increment = frozen::components(&pressure.selected_increment)?;
            let sum = frozen::components(&pressure.proposal_arithmetic)?;
            let proposal = frozen::components(&pressure.proposal)?;
            let contained = pressure.proposal_contained.try_to_vec::<f64>()?;
            let chosen = frozen::components(&pressure.chosen)?;
            let selection = frozen::components(&pressure.selection_change)?;
            let delta = frozen::components(&pressure.increment.delta_area)?;
            let physical = frozen::endpoints(&pressure.increment.physical_increment)?;
            let secant = frozen::endpoints(&pressure.increment.secant)?;
            for row in 0..owners.len() {
                assert_eq!(&anchor[row][..2], &selected_anchor[row][..2]);
                assert_eq!(&increment[row][..2], &selected_increment[row][..2]);
                assert_eq!(selected_anchor[row][2], 0);
                assert_eq!(selected_increment[row][2], 0);
                assert_eq!(proposal[row][2], 0);
                assert_eq!(chosen[row][2], 0);
                if contained[row] == 1. {
                    assert_eq!(proposal[row], chosen[row]);
                } else {
                    assert_eq!(contained[row], 0.);
                }
                assert!(
                    selection[row]
                        .iter()
                        .all(|bits| f64::from_bits(*bits).is_finite())
                );
                assert!(f64::from_bits(sum[row][2]) >= 0.);
                assert!(f64::from_bits(delta[row][2]) >= 0.);
                ordered(physical[row]);
                ordered(secant[row]);
                assert!(secant[row][1] >= 0.);
            }
            for bounds in frozen::endpoints(&pressure.evaluation)? {
                ordered(bounds);
            }
            for bounds in frozen::endpoints(&pressure.evaluation_discrepancy)? {
                ordered(bounds);
            }
        } else {
            assert_eq!(point.kind, PointFluxKind::Dry);
        }
    }
    for (row, count) in selected_counts.iter().enumerate() {
        assert!(
            *count <= 1,
            "Owner {} selects multiple candidates",
            requested[row]
        );
        if status[row] == PointFluxStatus::Accepted as i64 {
            assert_eq!(*count, 1);
        }
    }
    Ok(())
}

fn ordered(value: [f64; 2]) {
    assert!(value[0].is_finite() && value[1].is_finite() && value[0] <= value[1]);
}
fn encloses(outer: [f64; 2], inner: [f64; 2]) {
    assert!(
        outer[0] <= inner[0] && inner[1] <= outer[1],
        "{outer:?} excludes {inner:?}"
    );
}
