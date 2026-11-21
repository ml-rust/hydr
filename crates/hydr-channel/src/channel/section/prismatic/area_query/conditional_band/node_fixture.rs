//! Ordinary wet conditional segments on the shared private profiles, for callers outside the section module.

use super::ConditionalBandSegments;
use crate::channel::flow::surveyed::error;
use crate::channel::section::prismatic::area_query::{
    AreaDirection, AreaQuery, test_support as support,
};
use crate::channel::section::{HydraulicGeometry, PrismaticRecoveryConfig};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) use support::{Client, Result, hydraulic};

/// Segments of two faces at area 4 on owner 1, displaced by `distance` in opposite directions.
pub(crate) fn segments<R>(client: &R::Client, distance: f64) -> Result<ConditionalBandSegments<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = support::batch::<R>(client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let owners = Tensor::<R>::from_slice(&[1_i64; 2], &[2], client.device())?;
    let areas = support::expansion::<R>(client, &[[4.0_f64.to_bits(), 0, 0]; 2], false)?;
    let source =
        section.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;
    support::hydraulic(error::checked::<R, _>(
        client,
        "ordinary node fixture segments",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let scale =
                a.expansion_point(&Tensor::from_slice(&[distance; 2], &[2], client.device())?)?;
            let parameter = a.point(&Tensor::from_slice(&[1.0, -1.0], &[2], client.device())?)?;
            let endpoint = AreaQuery::source_affine(
                a,
                &bands,
                &source,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?
            .prove_support(a, &bands)?;
            ConditionalBandSegments::prepare(a, &bands, &source, &endpoint, &boundaries)
        },
    ))
}
