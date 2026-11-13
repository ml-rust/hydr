use super::{
    Result, chart_test_support, frozen, hydraulic, locate_wet_roots, refine_wet_depth, support,
};
use crate::channel::{
    flow::surveyed::error::checked,
    section::{AreaAnchor, AreaDirection, AreaQuery},
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    broad_uniform::<R>(client)?;
    source_crossing::<R>(client, false)?;
    source_crossing::<R>(client, true)
}

fn broad_uniform<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.0,
        8.0,
        &[[1.0, 1.0, 1.0, 1.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let mut session = hydraulic(refine_wet_depth(pair, located))?;
            // Exact uniform identity proves A*=1 independently of this wider supported bracket.
            session.domains[0].query = chart_test_support::chart(
                session.sources(),
                chart_test_support::Anchor::Zero,
                [1.0_f64.to_bits(), 0],
                &[[0.25, 2.25]],
            )?;
            let before = work.snapshot();
            let waves = hydraulic(session.certify_wet_waves())?;
            let wave = &waves[0];
            super::diagnostics::check(wave)?;
            assert_eq!(wave.regions.left_exterior.try_to_vec::<f64>()?, [1.0]);
            assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.0]);
            assert_eq!(wave.regions.left_fan.try_to_vec::<f64>()?, [0.0]);
            assert_eq!(wave.regions.right_fan.try_to_vec::<f64>()?, [0.0]);
            // The independent exact characteristic is zero. Certified uncertainty preserves both adjacent candidates.
            let head = frozen::endpoints(&wave.sides[0].head.physical)?[0];
            assert!(head[0] <= 0.0 && head[1] >= 0.0);
            for side in &wave.sides {
                assert_eq!(side.equality_possible.try_to_vec::<f64>()?, [1.0]);
                assert_eq!(side.shock_possible.try_to_vec::<f64>()?, [0.0]);
                assert_eq!(side.rarefaction_possible.try_to_vec::<f64>()?, [0.0]);
            }
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}

fn source_crossing<R>(client: &R::Client, uncertain_endpoint: bool) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    // Rectangle Phi(A)=2*sqrt(A). These distinct sources give the exact root A*=1,u*=1.
    support::with_rectangle::<R, _>(
        client,
        1.0,
        8.0,
        &[[1.0, 1.0, 4.0, 12.0]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[0])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let mut session = hydraulic(refine_wet_depth(pair, located))?;
            let query = if uncertain_endpoint {
                hydraulic(checked::<R, _>(
                    client,
                    "supported uncertain source-crossing wave control",
                    |a| {
                        let pair = session.sources();
                        let zero = Tensor::zeros(&[1], a.dtype(), client.device())?;
                        let two = Tensor::full_scalar(&[1], a.dtype(), 2.0, client.device())?;
                        let low = Tensor::full_scalar(
                            &[1],
                            a.dtype(),
                            -f64::from_bits(1),
                            client.device(),
                        )?;
                        let half = Tensor::full_scalar(&[1], a.dtype(), 0.5, client.device())?;
                        let upper = Tensor::full_scalar(&[1], a.dtype(), 1.125, client.device())?;
                        // Exact scale times one half gives 1-2^-1075. Only the endpoint transform acquires uncertainty.
                        let scale = a.expansion_bounds(&two, &low, &zero)?;
                        let parameter = a.bounds(&half, &upper)?;
                        let query = AreaQuery::band_affine(
                            a,
                            pair.bands,
                            AreaAnchor::Zero,
                            AreaDirection::Add,
                            &scale,
                            &parameter,
                        )?;
                        client.copy_into(&query.original_rows, &pair.original_rows)?;
                        query.prove_support(a, pair.bands)
                    },
                ))?
            } else {
                chart_test_support::chart(
                    session.sources(),
                    chart_test_support::Anchor::Zero,
                    [1.0_f64.to_bits(), 0],
                    &[[0.25, 2.25]],
                )?
            };
            session.domains[0].query = query;
            let before = work.snapshot();
            let waves = hydraulic(session.certify_wet_waves())?;
            let wave = &waves[0];
            super::diagnostics::check(wave)?;
            let left = &wave.sides[0];
            assert_eq!(left.shock_possible.try_to_vec::<f64>()?, [1.0]);
            assert_eq!(left.shock_actual.try_to_vec::<f64>()?, [0.0]);
            assert_eq!(left.shock_conditional.try_to_vec::<f64>()?, [1.0]);
            assert_eq!(left.rarefaction_possible.try_to_vec::<f64>()?, [1.0]);
            assert_eq!(left.equality_possible.try_to_vec::<f64>()?, [1.0]);
            // Independent source sound equals one. The retained source certificate supplies this lower limit.
            let source_sound = frozen::endpoints(&left.source_sound.physical)?[0];
            assert!(source_sound[0] <= 1.0 && source_sound[1] >= 1.0);
            let retained_sound =
                frozen::endpoints(&frozen::bounds(client, &left.source_sound.retained)?)?[0];
            assert!(retained_sound[0] <= 1.0 && retained_sound[1] >= 1.0);
            let offset = frozen::endpoints(&left.shock_offset.physical)?[0];
            assert_eq!(offset[0], retained_sound[0]);
            assert!(offset[0] <= 1.0 && offset[1] > 1.0);
            let shock = frozen::endpoints(&left.shock_speed.physical)?[0];
            assert!(shock[0] < 0.0 && shock[1] >= 0.0);
            assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.0]);
            if uncertain_endpoint {
                assert_eq!(
                    wave.star
                        .query
                        .query
                        .scale
                        .abs_error()
                        .try_to_vec::<f64>()?,
                    [0.0]
                );
                assert!(
                    wave.star.endpoints[0]
                        .area
                        .abs_error()
                        .try_to_vec::<f64>()?[0]
                        > 0.0
                );
                let evidence = wave.star.endpoints[0].waves.local[0]
                    .as_ref()
                    .ok_or("Missing conditional lower endpoint")?;
                assert_eq!(
                    evidence
                        .endpoint
                        .actual_shock
                        .value
                        .available
                        .try_to_vec::<f64>()?,
                    [0.0]
                );
                assert_eq!(
                    evidence
                        .endpoint
                        .conditional_shock
                        .value
                        .available
                        .try_to_vec::<f64>()?,
                    [1.0]
                );
            }
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            assert_eq!(work.snapshot().root_iterations, before.root_iterations);
            Ok(())
        },
    )
}
