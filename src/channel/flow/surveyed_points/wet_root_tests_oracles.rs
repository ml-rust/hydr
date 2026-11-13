use super::{certificates, support};
use crate::channel::flow::surveyed_points::{
    chart_test_support,
    source_tests::support::{Client, Result, components, hydraulic},
    wet_root_types::{WetRootEvidenceKind, WetRootStatus},
    wet_roots::locate_wet_roots,
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
    support::uncertain_dry_limit::<R, _>(client, &config, |pair, work| {
        let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        let result = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [WetRootStatus::Unresolved as i64]
        );
        assert!(result.domains.is_empty());
        certificates::check(pair, &result)?;
        let evidence = result
            .evidence
            .iter()
            .find(|evidence| evidence.kind == WetRootEvidenceKind::DryLimit)
            .ok_or("Unresolved dry limit lacks its original residual evidence")?;
        // Exact du=10/3 and PhiL+PhiR=10/3 establish mathematical touching.
        // Independent thirds certificates cannot establish the required structural zero.
        assert!(evidence.value.physical.lower().item::<f64>()? <= 0.0);
        assert!(evidence.value.physical.upper().item::<f64>()? >= 0.0);
        assert!(
            components(&evidence.value.retained)?
                .iter()
                .all(|point| f64::from_bits(point[2]) > 0.0)
        );
        assert!(result.callback_points() > 0);
        assert!(result.callback_points() <= config.root.max_points);
        assert_eq!(work.snapshot().residual_points, result.callback_points());
        Ok(())
    })
}
