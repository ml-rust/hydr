use crate::channel::flow::{
    surveyed::error::{HydraulicError, PrismaticRiemannRejection},
    surveyed_points::{
        chart_test_support::{self, cache},
        source_tests::support::{Client, Result, hydraulic},
        wet_root_types::WetRootStatus,
        wet_roots::locate_wet_roots,
    },
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = chart_test_support::config();
    chart_test_support::with_pair::<R, _>(
        client,
        &["scalar_rectangle_uniform_rest"],
        &config,
        |pair, work| {
            let parents = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let before = (cache(pair)?, work.snapshot());
            for points in [0, config.root.max_points + 1] {
                assert!(matches!(
                    locate_wet_roots(pair, &parents, points),
                    Err(HydraulicError::Rejected {
                        reason: PrismaticRiemannRejection::WorkLimit,
                        ..
                    })
                ));
                assert_eq!((cache(pair)?, work.snapshot()), before);
            }
            for invalid in [
                Tensor::from_slice(&[-1_i64], &[1], client.device())?,
                Tensor::from_slice(&[1_i64], &[1], client.device())?,
                Tensor::from_slice(&[0.0_f64], &[1], client.device())?,
                Tensor::from_slice(&[0_i64], &[], client.device())?,
            ] {
                assert!(locate_wet_roots(pair, &invalid, config.root.max_points).is_err());
                assert_eq!((cache(pair)?, work.snapshot()), before);
            }
            let empty = Tensor::from_slice::<i64>(&[], &[0], client.device())?;
            let result = hydraulic(locate_wet_roots(pair, &empty, 0))?;
            assert_eq!(result.status.shape(), [0]);
            assert_eq!(result.invocation_rows.shape(), [0]);
            assert!(result.domains.is_empty() && result.evidence.is_empty());
            assert_eq!(result.callback_points, 0);
            assert_eq!((cache(pair)?, work.snapshot()), before);
            let result = hydraulic(locate_wet_roots(pair, &parents, 1))?;
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [WetRootStatus::Located as i64]
            );
            assert_eq!(result.callback_points, 1);
            assert_eq!(work.snapshot().residual_points, 1);
            super::certificates::check(pair, &result)
        },
    )?;
    let mut zero = config.clone();
    zero.max_native_query_points = 0;
    chart_test_support::with_pair::<R, _>(
        client,
        &["scalar_rectangle_uniform_rest"],
        &zero,
        |pair, work| {
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let before = cache(pair)?;
            assert!(matches!(
                locate_wet_roots(pair, &rows, zero.root.max_points),
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                })
            ));
            assert_eq!(work.snapshot().residual_points, 0);
            assert_eq!(cache(pair)?, before);
            Ok(())
        },
    )?;
    chart_test_support::with_pair::<R, _>(
        client,
        &["scalar_rectangle_left_dry_sonic"],
        &config,
        |pair, work| {
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let before = (cache(pair)?, work.snapshot());
            assert!(matches!(
                locate_wet_roots(pair, &rows, config.root.max_points),
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::GeometryUnresolved,
                    ..
                })
            ));
            assert_eq!((cache(pair)?, work.snapshot()), before);
            Ok(())
        },
    )
}
