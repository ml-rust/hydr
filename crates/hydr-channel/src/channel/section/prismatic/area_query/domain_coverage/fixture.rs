use super::super::test_support::{Client, Result};
use crate::channel::section::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticSectionBatch,
};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn rectangles<R>(
    client: &R::Client,
    dimensions: &[(f64, f64)],
) -> Result<PrismaticSectionBatch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definitions = dimensions
        .iter()
        .copied()
        .enumerate()
        .map(|(index, (width, depth))| CrossSectionDefinition {
            id: format!("coverage_rectangle_{index}"),
            vertical_datum: "Exact rectangle datum".into(),
            points: [(0.0, depth), (0.0, 0.0), (width, 0.0), (width, depth)]
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
            maximum_stage_m: depth,
            profile_provenance: "Independent power-of-two capacity".into(),
        })
        .collect();
    Ok(CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?)
}
