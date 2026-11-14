use super::*;
use crate::channel::flow::surveyed_points::analytic_bottom_tests::fixtures;

pub(super) fn truth_and_dedup<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let ids = [
        "scalar_trapezoid_depth_jump",
        "scalar_three_bands_depth_jump",
    ];
    let config = chart_test_support::config();
    chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
        let rows = Tensor::from_slice(&[1_i64, 0, 1], &[3], client.device())?;
        let unique = Tensor::from_slice(&[0_i64, 1], &[2], client.device())?;
        let empty = Tensor::empty(&[0], DType::I64, client.device())?;
        let coarse = hydraulic(pair.panel_level(0))?;
        let fine = hydraulic(pair.panel_level(4))?;
        let low = hydraulic(pair.source_values_at_level(&coarse, [&rows, &rows]))?;
        let before = work.snapshot();
        let high = hydraulic(pair.source_values_at_level(&fine, [&rows, &rows]))?;
        let after = work.snapshot();
        assert!(after.simpson_value_points > before.simpson_value_points);
        assert!(after.simpson_derivative_points > before.simpson_derivative_points);
        assert_eq!(after.residual_points, before.residual_points);
        for side in 0..2 {
            let lo = endpoints(&low.values[side].physical)?;
            let hi = endpoints(&high.values[side].physical)?;
            let retained = endpoints(
                &crate::channel::flow::surveyed_points::source_tests::support::bounds::<R>(
                    client,
                    &high.values[side].retained,
                )?,
            )?;
            for (position, parent) in [1usize, 0, 1].into_iter().enumerate() {
                let truth = fixtures::CASES
                    .iter()
                    .find(|case| case.id == ids[parent])
                    .ok_or("Missing frozen panel source truth")?;
                assert_eq!(truth.gravity_bits, config.gravity_m_per_s2.to_bits());
                let expected = truth.sides[side].phi.bounds.map(f64::from_bits);
                for actual in [lo[position], hi[position], retained[position]] {
                    assert!(
                        actual[0] <= expected[0] && expected[1] <= actual[1],
                        "{} side={side}: {actual:?} excludes {expected:?}",
                        ids[parent]
                    );
                }
                assert!(hi[position][1] - hi[position][0] < lo[position][1] - lo[position][0]);
            }
            assert_eq!(hi[0], hi[2]);
        }
        let high_before = components(&high.values[0].retained)?;
        let baseline = chart_test_support::cache(pair)?;
        let cached = hydraulic(pair.source_values_at_level(&fine, [&unique, &unique]))?;
        assert_eq!(work.snapshot(), after);
        assert_eq!(components(&high.values[0].retained)?, high_before);
        assert_eq!(
            endpoints(&cached.values[0].physical)?[1],
            endpoints(&high.values[0].physical)?[0]
        );
        assert_eq!(chart_test_support::cache(pair)?, baseline);
        let caches = pair.panel_cache.len();
        let unused = hydraulic(pair.panel_level(5))?;
        hydraulic(pair.source_values_at_level(&unused, [&empty, &empty]))?;
        assert_eq!(pair.panel_cache.len(), caches);
        assert_eq!(work.snapshot(), after);
        Ok(())
    })?;
    let mut totals = Vec::new();
    for values in [&[1_i64, 0, 1][..], &[0_i64, 1][..]] {
        totals.push(chart_test_support::with_pair::<R, _>(
            client,
            &ids,
            &config,
            |pair, work| {
                let rows = Tensor::from_slice(values, &[values.len()], client.device())?;
                let level = hydraulic(pair.panel_level(1))?;
                hydraulic(pair.source_values_at_level(&level, [&rows, &rows]))?;
                Ok(work.snapshot())
            },
        )?);
    }
    assert_eq!(totals[0], totals[1]);
    Ok(())
}

pub(super) fn work_error<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let mut config = chart_test_support::config();
    let id = ["scalar_trapezoid_depth_jump"];
    let charge = chart_test_support::with_pair::<R, _>(client, &id, &config, |pair, work| {
        let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        let empty = Tensor::empty(&[0], DType::I64, client.device())?;
        let level = hydraulic(pair.panel_level(1))?;
        hydraulic(pair.source_values_at_level(&level, [&rows, &empty]))?;
        let work = work.snapshot();
        Ok(work.simpson_value_points + work.simpson_derivative_points)
    })?;
    assert!(charge > 0);
    config.max_native_query_points = charge;
    chart_test_support::with_pair::<R, _>(client, &id, &config, |pair, work| {
        let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        let empty = Tensor::empty(&[0], DType::I64, client.device())?;
        let level = hydraulic(pair.panel_level(1))?;
        let result = pair.source_values_at_level(&level, [&rows, &rows]);
        assert!(matches!(
            result,
            Err(HydraulicError::Rejected {
                reason: crate::channel::flow::surveyed::PrismaticRiemannRejection::WorkLimit,
                ..
            })
        ));
        assert_eq!(pair.panel_cache.len(), 1);
        let caches = &pair.panel_cache[0].1;
        assert_eq!(caches[0].ready.try_to_vec::<f64>()?, [1.0]);
        assert_eq!(caches[1].ready.try_to_vec::<f64>()?, [0.0]);
        let before = work.snapshot();
        assert_eq!(
            before.simpson_value_points + before.simpson_derivative_points,
            charge
        );
        let checked = hydraulic(pair.source_values_at_level(&level, [&rows, &empty]))?;
        assert_eq!(checked.values[0].available.try_to_vec::<f64>()?, [1.0]);
        assert_eq!(work.snapshot(), before);
        assert!(pair.source_values_at_level(&level, [&rows, &rows]).is_err());
        assert_eq!(work.snapshot(), before);
        Ok(())
    })
}
