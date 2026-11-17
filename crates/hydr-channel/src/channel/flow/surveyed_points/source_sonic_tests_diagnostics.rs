use super::{Result, frozen, support};
use crate::channel::flow::surveyed_points::source_sonic_types::{
    SonicFanMembership, SourceSonicCertificate,
};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn check<R>(domain: &SourceSonicCertificate<R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    assert_eq!(domain.fan_membership, SonicFanMembership::NotChecked);
    let n = domain.invocation_owner.numel();
    assert_eq!(domain.parent_rows.numel(), n);
    assert_eq!(domain.query.query.original_rows.numel(), n);
    assert_eq!(domain.depth_status.numel(), n);
    for value in [
        &domain.normalized_depth,
        &domain.sound,
        &domain.velocity,
        &domain.discharge,
        &domain.residual,
    ] {
        for [lower, upper] in frozen::endpoints(value)? {
            assert!(lower.is_finite() && upper.is_finite() && lower <= upper);
        }
    }
    for [lower, upper] in frozen::endpoints(&domain.residual)? {
        assert!(lower <= 0.0 && upper >= 0.0);
    }
    for value in [
        &domain.source_discharge,
        &domain.area_difference,
        &domain.velocity_change,
        &domain.discharge_change,
        &domain.anchored_discharge,
    ] {
        assert!(
            value
                .abs_error()
                .try_to_vec::<f64>()?
                .iter()
                .all(|&x| x.is_finite() && x >= 0.0)
        );
    }
    assert!(domain.callback_points >= domain.iterations);
    assert!(domain.mapped_points >= domain.iterations);
    Ok(())
}

pub(super) fn evidence<R: Runtime<DType = DType>>(
    result: &crate::channel::flow::surveyed_points::source_sonic_types::SourceSonicCertificates<R>,
) -> Result {
    assert!(result.callback_points <= result.point_limit);
    let mut reported = result
        .domains
        .iter()
        .map(|d| d.callback_points)
        .sum::<usize>();
    for item in &result.evidence {
        let owners = item.invocation_owner.try_to_vec::<i64>()?;
        assert_eq!(item.parent_rows.numel(), owners.len());
        assert_eq!(item.status.numel(), owners.len());
        assert_eq!(item.dry_residual.available.numel(), owners.len());
        assert_eq!(item.source_residual.available.numel(), owners.len());
        let global = result.status[item.side.index()].try_to_vec::<i64>()?;
        let status = item.status.try_to_vec::<i64>()?;
        for (index, owner) in owners.into_iter().enumerate() {
            assert_eq!(global[usize::try_from(owner)?], status[index]);
        }
        assert!(item.bracket_expansions <= item.bracket_points);
        reported += item.endpoint_points + item.bracket_points;
    }
    assert!(reported <= result.callback_points);
    Ok(())
}
