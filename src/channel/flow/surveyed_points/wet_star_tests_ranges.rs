use super::{
    Result, WetStarStatus, chart_test_support, frozen, hydraulic, locate_wet_roots,
    refine_wet_depth, support,
};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn full_domain_discharge<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    use crate::channel::flow::surveyed_points::wet_depth_types::WetDepthStatus;
    use numr::{runtime::RuntimeClient, tensor::Tensor};
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        8.0,
        &[[4.0, -4.0, 4.0, 12.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let mut session = hydraulic(refine_wet_depth(pair, located))?;
            // Widen the supported bracket around the independently known root A*=1.
            let query = chart_test_support::chart(
                session.sources(),
                chart_test_support::Anchor::Zero,
                [1.0_f64.to_bits(), 0],
                &[[0.25, 2.25]],
            )?;
            session.domains[0].query = query;
            session.domains[0].status = Tensor::full_scalar(
                &[1],
                DType::I64,
                WetDepthStatus::WorkLimit as i64 as f64,
                client.device(),
            )?;
            let before = work.snapshot();
            let certificates = hydraulic(session.certify_wet_star())?;
            let star = &certificates[0];
            assert_eq!(
                star.status.try_to_vec::<i64>()?,
                [WetStarStatus::Certified as i64]
            );
            let range = frozen::endpoints(&star.side_discharge[0])?[0];
            assert!(range[0] <= 1.0 && range[1] >= 1.0);
            // QL(A)=A*(3-2*sqrt(A)) peaks at the root, above both endpoint values.
            let lower = frozen::endpoints(&frozen::bounds(
                client,
                &star.endpoints[0].sides[0].discharge,
            )?)?[0];
            let upper = frozen::endpoints(&frozen::bounds(
                client,
                &star.endpoints[1].sides[0].discharge,
            )?)?[0];
            assert!(lower[1] < 1.0 && upper[1] < 1.0);
            assert_eq!(
                session.domains()[0].status.try_to_vec::<i64>()?,
                [WetDepthStatus::WorkLimit as i64]
            );
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}

pub(super) fn empty_intersections<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    use crate::channel::flow::surveyed::error::checked;
    use numr::{runtime::RuntimeClient, tensor::Tensor};
    hydraulic(checked::<R, _>(
        client,
        "wet star unresolved and unavailable bounds",
        |a| {
            let zero = Tensor::zeros(&[1], a.dtype(), client.device())?;
            let one = Tensor::ones(&[1], a.dtype(), client.device())?;
            let first = a.point(&zero)?;
            let second = a.point(&one)?;
            let (value, valid) = super::super::intersection(a, &first, &second)?;
            assert_eq!(valid.try_to_vec::<f64>()?, [0.0]);
            assert_eq!(value.lower().try_to_vec::<f64>()?, [0.0]);
            let (value, valid) = super::super::ordered_bounds(a, &one, &zero, &zero)?;
            assert_eq!(valid.try_to_vec::<f64>()?, [0.0]);
            assert_eq!(value.upper().try_to_vec::<f64>()?, [0.0]);
            Ok(())
        },
    ))
}
