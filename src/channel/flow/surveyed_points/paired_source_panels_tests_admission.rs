use super::*;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let config = chart_test_support::config();
    let id = ["scalar_trapezoid_depth_jump"];
    let foreign = chart_test_support::with_pair::<R, _>(client, &id, &config, |pair, _| {
        hydraulic(pair.panel_level(1))
    })?;
    chart_test_support::with_pair::<R, _>(client, &id, &config, |pair, work| {
        let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        let empty = Tensor::empty(&[0], DType::I64, client.device())?;
        let level = hydraulic(pair.panel_level(1))?;
        let before = work.snapshot();
        let coarse = chart_test_support::cache(pair)?;
        assert!(
            pair.source_values_at_level(&foreign, [&rows, &rows])
                .is_err()
        );
        let rejected_level = config.max_refinements + 1;
        let HydraulicError::Numerical { context: error } =
            pair.panel_level(rejected_level)
                .err()
                .ok_or("Out-of-limit panel level accepted")?
        else {
            return Err("Out-of-limit panel level returned a non-numerical error".into());
        };
        assert!(error.contains(&format!("level={rejected_level}")));
        assert!(error.contains(&format!("max_refinements={}", config.max_refinements)));
        assert!(error.contains(&format!("max_panels={}", config.max_simpson_panels)));
        assert!(pair.panel_level(usize::MAX).is_err());
        let panels = pair.evaluation.panels;
        for changed in [0, panels * 2] {
            pair.evaluation.panels = changed;
            assert!(pair.source_values_at_level(&level, [&rows, &rows]).is_err());
            assert!(pair.panel_level(0).is_err());
        }
        pair.evaluation.panels = panels;
        hydraulic(pair.source_values_at_level(&level, [&empty, &empty]))?;
        for (position, invalid) in [
            Tensor::from_slice(&[-1_i64], &[1], client.device())?,
            Tensor::from_slice(&[1_i64], &[1], client.device())?,
            Tensor::from_slice(&[i64::MAX], &[1], client.device())?,
            rows.reshape(&[1, 1])?,
            Tensor::from_slice(&[0_f64], &[1], client.device())?,
        ]
        .into_iter()
        .enumerate()
        {
            let HydraulicError::Numerical { context: error } = pair
                .source_values_at_level(&level, [&rows, &invalid])
                .err()
                .ok_or("Invalid right source panel request accepted")?
            else {
                return Err(
                    "Invalid right source panel request returned a non-numerical error".into(),
                );
            };
            assert!(error.contains("side=right"));
            if position < 3 {
                let parent = [-1_i64, 1, i64::MAX][position];
                assert!(error.contains(&format!("parent={parent}")));
                assert!(error.contains("captured_count=1"));
                assert!(error.contains("[0, 1)"));
            } else {
                assert!(error.contains("rank="));
                assert!(error.contains("dtype="));
                assert!(error.contains("device="));
            }
            let base = hydraulic(pair.panel_level(0))?;
            assert!(
                pair.source_values_at_level(&base, [&rows, &invalid])
                    .is_err()
            );
        }
        assert!(pair.panel_cache.is_empty());
        assert_eq!(work.snapshot(), before);
        assert_eq!(chart_test_support::cache(pair)?, coarse);
        Ok(())
    })?;
    let huge = crate::channel::flow::surveyed::PrismaticRiemannConfig {
        max_refinements: usize::MAX,
        max_simpson_panels: 1usize << (usize::BITS - 1),
        ..config
    };
    chart_test_support::with_pair::<R, _>(client, &id, &huge, |pair, work| {
        let before = work.snapshot();
        assert!(pair.panel_level(usize::MAX).is_err());
        assert!(pair.panel_level(usize::BITS as usize).is_err());
        assert!(pair.panel_cache.is_empty());
        assert_eq!(work.snapshot(), before);
        Ok(())
    })?;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(super) fn foreign_device(client: &numr::runtime::cuda::CudaClient) -> Result {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};
    let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::I64, &CudaDevice::new(1))?;
    chart_test_support::with_pair::<CudaRuntime, _>(
        client,
        &["scalar_trapezoid_depth_jump"],
        &chart_test_support::config(),
        |pair, work| {
            let before = work.snapshot();
            let empty = Tensor::empty(&[0], DType::I64, client.device())?;
            let level = hydraulic(pair.panel_level(1))?;
            assert!(
                pair.source_values_at_level(&level, [&empty, &foreign])
                    .is_err()
            );
            assert!(pair.panel_cache.is_empty());
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}
