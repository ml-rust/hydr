use super::{
    frozen::{Result, hydraulic},
    support,
};
use crate::channel::flow::surveyed_points::{
    chart_test_support, wet_depth::refine_wet_depth, wet_roots::locate_wet_roots,
};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    unchanged_retained_source::<R>(client)?;
    let foreign = support::with_rectangle::<R, _>(
        client,
        1.0,
        8.0,
        &[[1.0, 0.0, 1.0, 0.0]],
        &config,
        |pair, _| hydraulic(pair.root_witness()),
    )?;
    for stale in [false, true] {
        support::with_rectangle::<R, _>(
            client,
            1.0,
            8.0,
            &[[1.0, 0.0, 1.0, 0.0]],
            &config,
            |pair, work| {
                let rows = support::rows::<R>(client, &[0])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                let mut session = hydraulic(refine_wet_depth(pair, located))?;
                let before = work.snapshot();
                let cache = chart_test_support::cache(session.sources())?;
                if stale {
                    session.sources.evaluation.panels *= 2;
                } else {
                    session.witness = foreign.clone();
                }
                assert!(session.certify_wet_star().is_err());
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(session.sources())?, cache);
                Ok(())
            },
        )?;
    }
    support::with_rectangle::<R, _>(client, 1.0, 8.0, &[], &config, |pair, work| {
        let rows = support::rows::<R>(client, &[])?;
        let located = hydraulic(locate_wet_roots(pair, &rows, 0))?;
        let mut session = hydraulic(refine_wet_depth(pair, located))?;
        let before = work.snapshot();
        assert!(hydraulic(session.certify_wet_star())?.is_empty());
        assert_eq!(work.snapshot(), before);
        Ok(())
    })
}

fn unchanged_retained_source<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    use super::frozen::components;
    use crate::channel::{
        flow::surveyed_points::{potential_range::source_query, wet_star_endpoint},
        section::PhysicalEndpoint,
    };
    let config = chart_test_support::config();
    chart_test_support::with_pair::<R, _>(
        client,
        &["uniform_cap_retained_deficit_dyadic_velocity"],
        &config,
        |pair, _| {
            let rows = support::rows::<R>(client, &[0])?;
            let query = hydraulic(source_query(
                pair.geometry,
                pair.bands,
                pair.sources[0],
                &pair.original_rows,
            ))?;
            let endpoint = hydraulic(wet_star_endpoint::evaluate(
                pair,
                &rows,
                &query,
                PhysicalEndpoint::Lower,
            ))?;
            for side in &endpoint.sides {
                assert_eq!(side.unchanged.try_to_vec::<f64>()?, [1.0]);
                assert_eq!(
                    components(&side.discharge)?,
                    components(&pair.velocity.input().left_discharge)?
                );
                for point in components(&side.area_difference)?
                    .into_iter()
                    .chain(components(&side.velocity_change)?)
                {
                    assert!(point.map(f64::from_bits).iter().all(|&x| x == 0.0));
                }
            }
            Ok(())
        },
    )
}
