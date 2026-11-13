use super::{cases, certificates, support};
use crate::channel::flow::{
    surveyed::PrismaticRiemannConfig,
    surveyed_points::{
        chart_roots::solve_chart,
        chart_test_support,
        paired_sources::PairedSources,
        source_tests::support::{Client, Result, components, endpoints, hydraulic},
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
    optimize::{
        CertifiedBisectionAlgorithms, CertifiedRootStatus, CertifiedUpperBracketAlgorithms,
    },
};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    for id in [
        "retained_vacuum_inside_film",
        "uniform_cap_retained_deficit_dyadic_velocity",
        "scalar_rectangle_stationary_shock",
        "tiny_rectangle_compression",
        "scalar_rectangle_weak_depth_jump",
        "retained_knot_2_crossing",
        "retained_knot_7_crossing",
    ] {
        let (config, _) = cases::declared(id);
        chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
            chain(pair, &config, id, cases::star(id))?;
            assert!(work.snapshot().residual_points <= config.root.max_points);
            Ok(())
        })?;
    }
    run_pointed::<R>(client)
}

pub(super) fn run_pointed<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let mut config = chart_test_support::config();
    config.gravity_m_per_s2 = 2.0;
    support::pointed_film::<R, _>(client, &config, |pair, _| {
        // A=h², Phi(A)=4*A^(1/4), F(A)=8*A^(1/4)-2^-80.
        chain(
            pair,
            &config,
            "analytic_pointed_film",
            Some([0x2b30000000000000; 2]),
        )
    })
}

fn chain<R>(
    pair: &mut PairedSources<'_, R>,
    config: &PrismaticRiemannConfig,
    id: &str,
    star: Option<[u64; 2]>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let client = pair.geometry.client();
    let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
    assert_eq!(
        located.status.try_to_vec::<i64>()?,
        [WetRootStatus::Located as i64],
        "{id}: callbacks={}, work={:?}, evidence={:?}",
        located.callback_points,
        pair.evaluation.work.snapshot(),
        located
            .evidence
            .iter()
            .map(|evidence| Ok((
                evidence.kind,
                endpoints(&evidence.value.physical)?,
                components(&evidence.value.retained)?,
            )))
            .collect::<Result<Vec<_>>>()?
    );
    certificates::check(pair, &located)?;
    assert!(!located.domains.is_empty(), "{id}");
    let mut remaining = config
        .root
        .max_points
        .checked_sub(located.callback_points)
        .ok_or("Locator exceeds cumulative root-point budget")?;
    let automatic_film = matches!(id, "retained_vacuum_inside_film" | "analytic_pointed_film");
    for domain in &located.domains {
        if automatic_film {
            let physical = support::physical_bounds(pair, &domain.parent_rows, &domain.query)?;
            assert!(
                physical
                    .iter()
                    .all(|bounds| bounds[1] > 0.0 && bounds[1] < 1e-40),
                "{id}: automatic film domain {physical:?}"
            );
        }
        let before = pair.evaluation.work.snapshot().residual_points;
        let root = hydraulic(solve_chart(
            pair,
            &domain.parent_rows,
            &domain.query,
            remaining,
        ))?;
        let after = pair.evaluation.work.snapshot().residual_points;
        assert_eq!(after - before, root.result.evaluated_points(), "{id}");
        remaining = remaining
            .checked_sub(root.result.evaluated_points())
            .ok_or("Chart refinement exceeds cumulative root-point budget")?;
        assert_eq!(root.root_rows.try_to_vec::<i64>()?, [0], "{id}");
        assert!(
            root.result
                .status()
                .try_to_vec::<i64>()?
                .iter()
                .all(|&code| code == CertifiedRootStatus::Converged as i64
                    || code == CertifiedRootStatus::AccuracyFloor as i64),
            "{id}: root status {:?}",
            root.result.status().try_to_vec::<i64>()?
        );
        let physical = support::physical_bounds(pair, &root.root_rows, &root.query)?;
        if let Some(reference) = star {
            for &bounds in &physical {
                support::star_contains(bounds, reference, id);
            }
        }
        if id == "retained_vacuum_inside_film" {
            let star = f64::from_bits(0x35b0000000000000);
            let gate = f64::from_bits(0x3ddb7cdfd9d7bdba);
            for [lower, upper] in physical {
                assert!(lower <= star && star <= upper);
                assert!(
                    (upper - lower) / star <= gate,
                    "film [{lower:e}, {upper:e}]"
                );
                assert!((star - lower).max(upper - star) / star <= gate);
            }
        }
        if id == "uniform_cap_retained_deficit_dyadic_velocity" {
            support::cap_deficit_contains(pair, &root.root_rows, &root.query)?;
        }
    }
    Ok(())
}
