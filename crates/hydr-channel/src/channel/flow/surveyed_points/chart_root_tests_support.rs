use crate::channel::flow::surveyed_points::{
    chart_roots::ChartRoots,
    chart_test_support,
    paired_sources::PairedSources,
    source_tests::support::{Client, Result},
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::common::arithmetic::CertifiedExpansion;

pub(super) use chart_test_support::{
    Anchor, chart, config, rectangle, select, uncertain_pair, with_pair,
};

pub(super) fn physical_endpoints<R>(
    pair: &PairedSources<'_, R>,
    roots: &ChartRoots<R>,
) -> Result<[CertifiedExpansion<R>; 2]>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(
        roots.query.query.parameter.lower().try_to_vec::<f64>()?,
        roots.result.root_lower().try_to_vec::<f64>()?
    );
    assert_eq!(
        roots.query.query.parameter.upper().try_to_vec::<f64>()?,
        roots.result.root_upper().try_to_vec::<f64>()?
    );
    chart_test_support::endpoint_areas(pair, &roots.root_rows, &roots.query)
}
