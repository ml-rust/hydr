use super::{super::solve_chart, support};
use crate::channel::{
    flow::surveyed_points::source_tests::support::{Client, Result, endpoints, hydraulic},
    section::PhysicalEndpoint,
};
use numr::{
    dtype::DType,
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedBisectionAlgorithms, CertifiedRootStatus},
};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let config = support::config();
    support::uncertain_pair::<R, _>(client, &config, |pair, work| {
        let query = support::chart(
            pair,
            support::Anchor::Zero,
            [1.0_f64.to_bits(), 0],
            &[[1.0, 1.0]],
        )?;
        let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        let residual = hydraulic(pair.endpoint_on_rows(&rows, &query, PhysicalEndpoint::Lower))?;
        assert_eq!(residual.available.try_to_vec::<f64>()?, [1.0]);
        assert!(residual.retained.abs_error().item::<f64>()? > 0.0);
        let physical = endpoints(&residual.physical)?;
        assert_eq!(physical.len(), 1);
        assert!(physical[0][0] <= 0.0 && physical[0][1] >= 0.0 && physical[0][0] < physical[0][1]);
        let before = work.snapshot();
        let roots = hydraulic(solve_chart(pair, &rows, &query, config.root.max_points))?;
        assert_eq!(roots.invocation_rows.try_to_vec::<i64>()?, [0]);
        assert_eq!(
            roots.result.status().try_to_vec::<i64>()?,
            [CertifiedRootStatus::UnresolvedBracket as i64]
        );
        for value in [
            &roots.root_rows,
            roots.result.root_owner(),
            &roots.query.query.owners,
            &roots.query.query.original_rows,
        ] {
            assert_eq!(value.shape(), &[0]);
            assert_eq!(value.dtype(), DType::I64);
            assert!(value.device().is_same(client.device()));
        }
        for value in [
            roots.result.root_lower(),
            roots.result.root_upper(),
            roots.query.query.parameter.lower(),
            roots.query.query.parameter.upper(),
        ] {
            assert_eq!(value.shape(), &[0]);
            assert_eq!(value.dtype(), DType::F64);
            assert!(value.device().is_same(client.device()));
        }
        assert_eq!(
            work.snapshot().residual_points - before.residual_points,
            roots.result.evaluated_points()
        );
        Ok(())
    })?;
    support::with_pair::<R, _>(
        client,
        &["scalar_rectangle_uniform_rest"],
        &config,
        |pair, work| {
            let full = support::chart(
                pair,
                support::Anchor::Source,
                [1.0_f64.to_bits(), 0],
                &[[-0.25, 0.5]],
            )?;
            let empty = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
            let query = support::select(pair, &full, &empty)?;
            let before = work.snapshot();
            let roots = hydraulic(solve_chart(pair, &empty, &query, 0))?;
            for value in [
                &roots.invocation_rows,
                &roots.root_rows,
                roots.result.status(),
                roots.result.root_owner(),
                &roots.query.query.owners,
                &roots.query.query.original_rows,
            ] {
                assert_eq!(value.shape(), &[0]);
                assert_eq!(value.dtype(), DType::I64);
                assert!(value.device().is_same(client.device()));
            }
            for value in [
                roots.result.root_lower(),
                roots.result.root_upper(),
                roots.query.query.parameter.lower(),
                roots.query.query.parameter.upper(),
                roots.query.query.scale.high(),
                roots.query.query.scale.low(),
                roots.query.query.scale.abs_error(),
            ] {
                assert_eq!(value.shape(), &[0]);
                assert_eq!(value.dtype(), DType::F64);
                assert!(value.device().is_same(client.device()));
            }
            assert_eq!(roots.result.iterations(), 0);
            assert_eq!(roots.result.evaluated_points(), 0);
            let after = work.snapshot();
            assert_eq!(after.root_solves, before.root_solves + 1);
            assert_eq!(after.residual_points, before.residual_points);
            assert_eq!(after.simpson_value_points, before.simpson_value_points);
            assert_eq!(
                after.simpson_derivative_points,
                before.simpson_derivative_points
            );
            Ok(())
        },
    )
}
