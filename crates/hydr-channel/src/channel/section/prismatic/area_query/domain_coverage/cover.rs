use super::super::super::{boundary_area::BoundaryAreas, hydraulic_geometry::GatheredBands};
use super::super::chart::{AreaAnchor, AreaDirection, AreaQuery};
#[cfg(test)]
use super::super::selection;
use super::{CoveredDomain, DomainCoverage, admit::admit, align::align, proof::proof_candidates};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Prove coverage with overlapping lower and cap-deficit charts, without diagnosing physical capacity rejection.
/// Results remain provisional until the caller's existing checked context completes.
pub(crate) fn cover_domain<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    boundaries: &BoundaryAreas<R>,
    original_rows: &Tensor<R>,
) -> HydraulicResult<DomainCoverage<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let diagnostics = admit(a, bands, boundaries, original_rows)?;
    let c = a.client();
    #[cfg(test)]
    let count = bands.section_owner.numel();
    #[cfg(test)]
    let zero = Tensor::zeros(&[count], a.dtype(), c.device())?;
    let capacity = boundaries.boundary(a, boundaries.cap_indices())?;
    let candidates = proof_candidates(a, &capacity)?;
    if candidates.owner.numel() == 0 {
        return Ok(DomainCoverage {
            #[cfg(test)]
            proved: zero,
            owner: candidates.owner,
            pair: None,
        });
    }

    let selected_bands = bands.select(a, &candidates.owner)?;
    let diagnostics = c.index_select(&diagnostics, 0, &candidates.owner)?;
    let span = a.expansion_interval(&candidates.span)?;
    let parameter = a.bounds(
        &Tensor::zeros(span.lower().shape(), a.dtype(), c.device())?,
        span.upper(),
    )?;
    let scale = a.expansion_point(&Tensor::ones(span.lower().shape(), a.dtype(), c.device())?)?;
    let lower = AreaQuery::band_affine(
        a,
        &selected_bands,
        AreaAnchor::Zero,
        AreaDirection::Add,
        &scale,
        &parameter,
    )?;
    let cap = AreaQuery::cap_deficit(a, &selected_bands, &scale, &parameter)?;
    c.copy_into(&lower.original_rows, &diagnostics)?;
    c.copy_into(&cap.original_rows, &diagnostics)?;
    let lower = lower.select_supported(a, &selected_bands)?;
    let cap = cap.select_supported(a, &selected_bands)?;
    let common = c.mul(&candidates.proved, &c.mul(&lower.proved, &cap.proved)?)?;
    let common_owner = c.nonzero(&common)?.flatten()?;
    let owner = c.index_select(&candidates.owner, 0, &common_owner)?;
    if common_owner.numel() == 0 {
        return Ok(DomainCoverage {
            #[cfg(test)]
            proved: zero,
            owner,
            pair: None,
        });
    }

    let lower = align(a, lower, &common, &common_owner, "lower")?;
    let cap = align(a, cap, &common, &common_owner, "cap")?;
    #[cfg(test)]
    let cap_minus_lower = selection::expansion(a, &candidates.cap_minus_lower, &common_owner)?;
    #[cfg(test)]
    let cap_minus_deficit = selection::expansion(a, &candidates.cap_minus_deficit, &common_owner)?;
    #[cfg(test)]
    let overlap_gap = selection::expansion(a, &candidates.overlap_gap, &common_owner)?;
    #[cfg(test)]
    let proved = c.index_put(
        &zero,
        0,
        &owner,
        &Tensor::ones(owner.shape(), a.dtype(), c.device())?,
    )?;
    Ok(DomainCoverage {
        #[cfg(test)]
        proved,
        owner,
        pair: Some(CoveredDomain {
            lower,
            cap,
            #[cfg(test)]
            cap_minus_lower,
            #[cfg(test)]
            cap_minus_deficit,
            #[cfg(test)]
            overlap_gap,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::test_support as support;
    use super::super::fixture::rectangles;
    use super::super::proof::tests as certificates;
    use super::super::{CoveredDomain, DomainCoverage};
    use super::*;
    use crate::channel::section::PrismaticSectionBatch;
    use crate::channel::{flow::surveyed::error, section::HydraulicGeometry};
    use numr::runtime::Device;
    use support::{Client, Result, hydraulic};

    mod controls {
        use super::super::super::super::test_support as support;
        use super::super::super::admit::tests::admission;
        use super::super::super::fixture::rectangles;
        use super::super::super::proof::tests::wide_arithmetic;
        use super::super::cover_domain;
        use crate::channel::{
            flow::surveyed::error,
            section::{CrossSectionError, HydraulicGeometry, PrismaticSectionBatch},
        };
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use support::{Client, Result, hydraulic};

        pub(super) fn run<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>,
        {
            rejected_pressure::<R>(client)?;
            empty::<R>(client, section)?;
            admission::<R>(client, section)?;
            wide_arithmetic::<R>(client)
        }

        fn rejected_pressure<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>,
        {
            let error = rectangles::<R>(client, &[(2.0_f64.powi(-500), 2.0_f64.powi(-573))])
                .err()
                .ok_or(
                    "Pressure-underflow rectangle unexpectedly passes ordinary geometry admission",
                )?;
            assert!(
                matches!(error.downcast_ref::<CrossSectionError>(), Some(CrossSectionError::Numerical { context, .. })
                if context.contains("hydrostatic pressure integral=0") && context.contains("rescale profile coordinates"))
            );
            Ok(())
        }

        fn empty<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>,
        {
            let geometry = HydraulicGeometry::from_batch(section);
            let rows = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
            let coverage = hydraulic(error::checked::<R, _>(
                client,
                "empty geometry domain coverage",
                |a| {
                    let bands = geometry.gather(a, &rows)?;
                    let boundaries = geometry.boundary_areas(a)?.gather(a, &rows)?;
                    cover_domain(a, &bands, &boundaries, &rows)
                },
            ))?;
            assert_eq!(coverage.proved.shape(), &[0]);
            assert_eq!(coverage.proved.dtype(), DType::F64);
            assert_eq!(coverage.owner.shape(), &[0]);
            assert_eq!(coverage.owner.dtype(), DType::I64);
            assert!(coverage.pair.is_none());
            Ok(())
        }
    }

    fn witnesses<R>(client: &R::Client, pair: &CoveredDomain<R>, count: usize) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        hydraulic(error::checked::<R, _>(
            client,
            "retained domain coverage inequalities",
            |a| {
                for (value, nonnegative) in [
                    (&pair.cap_minus_lower, true),
                    (&pair.cap_minus_deficit, true),
                    (&pair.overlap_gap, false),
                ] {
                    for component in [value.high(), value.low(), value.abs_error()] {
                        assert_eq!(component.shape(), &[count]);
                        assert_eq!(component.dtype(), DType::F64);
                        assert!(component.device().is_same(client.device()));
                    }
                    let interval = a.expansion_interval(value)?;
                    if nonnegative {
                        assert!(
                            interval
                                .lower()
                                .try_to_vec::<f64>()?
                                .into_iter()
                                .all(|value| value >= 0.0)
                        );
                    } else {
                        assert!(
                            interval
                                .upper()
                                .try_to_vec::<f64>()?
                                .into_iter()
                                .all(|value| value <= 0.0)
                        );
                    }
                }
                Ok(())
            },
        ))
    }

    fn exercise<R>(
        client: &R::Client,
        section: &PrismaticSectionBatch<R>,
        owners: &[i64],
        normals: &[bool],
        capacity_truth: Option<&[f64]>,
    ) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        assert_eq!(owners.len(), normals.len());
        let geometry = HydraulicGeometry::from_batch(section);
        let owner_input = Tensor::from_slice(owners, &[owners.len()], client.device())?;
        let identities = (0..owners.len())
            .map(|index| i64::try_from(index).map(|index| 701 + index))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let diagnostics = Tensor::from_slice(&identities, &[owners.len()], client.device())?;
        let coverage: DomainCoverage<R> = hydraulic(error::checked::<R, _>(
            client,
            "real compiled geometry domain coverage",
            |a| {
                let bands = geometry.gather(a, &owner_input)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owner_input)?;
                if let Some(truth) = capacity_truth {
                    assert_eq!(truth.len(), owners.len());
                    let capacity = boundaries.boundary(a, boundaries.cap_indices())?;
                    let interval = a.expansion_interval(&capacity)?;
                    for ((lower, upper), &truth) in interval
                        .lower()
                        .try_to_vec::<f64>()?
                        .into_iter()
                        .zip(interval.upper().try_to_vec::<f64>()?)
                        .zip(truth)
                    {
                        assert!(lower <= truth && truth <= upper);
                    }
                }
                cover_domain(a, &bands, &boundaries, &diagnostics)
            },
        ))?;
        assert_eq!(coverage.proved.shape(), &[owners.len()]);
        assert_eq!(coverage.proved.dtype(), DType::F64);
        assert_eq!(coverage.owner.dtype(), DType::I64);
        let proved = coverage.proved.try_to_vec::<f64>()?;
        assert!(proved.iter().all(|&value| value == 0.0 || value == 1.0));
        for (&normal, &proved) in normals.iter().zip(&proved) {
            if normal {
                assert_eq!(proved, 1.0);
            }
        }
        let selected = proved
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value == 1.0).then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(
            coverage.owner.try_to_vec::<i64>()?,
            selected
                .iter()
                .map(|&index| i64::try_from(index))
                .collect::<std::result::Result<Vec<_>, _>>()?
        );
        if selected.is_empty() {
            assert!(coverage.pair.is_none());
            return Ok(());
        }
        let pair = coverage
            .pair
            .ok_or("Proved domain rows have no covered pair")?;
        witnesses::<R>(client, &pair, selected.len())?;
        for query in [&pair.lower, &pair.cap] {
            assert_eq!(
                query.query.owners.try_to_vec::<i64>()?,
                selected
                    .iter()
                    .map(|&index| owners[index])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                query.query.original_rows.try_to_vec::<i64>()?,
                selected
                    .iter()
                    .map(|&index| identities[index])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                query.query.parameter.lower().try_to_vec::<f64>()?,
                vec![0.0; selected.len()]
            );
            assert_eq!(
                query.query.scale.high().try_to_vec::<f64>()?,
                vec![1.0; selected.len()]
            );
            assert_eq!(
                query.query.scale.low().try_to_vec::<f64>()?,
                vec![0.0; selected.len()]
            );
            assert_eq!(
                query.query.scale.abs_error().try_to_vec::<f64>()?,
                vec![0.0; selected.len()]
            );
        }
        assert_eq!(
            pair.lower.query.parameter.upper().try_to_vec::<f64>()?,
            pair.cap.query.parameter.upper().try_to_vec::<f64>()?
        );
        assert!(matches!(&pair.lower.query.anchor, super::AreaAnchor::Zero));
        assert!(matches!(&pair.cap.query.anchor, super::AreaAnchor::Cap));
        assert!(matches!(
            pair.lower.query.direction,
            super::AreaDirection::Add
        ));
        assert!(matches!(
            pair.cap.query.direction,
            super::AreaDirection::Subtract
        ));
        certificates::check::<R>(
            client,
            &geometry,
            &owner_input,
            &diagnostics,
            &coverage.owner,
            &coverage.proved,
            &pair,
        )?;
        Ok(())
    }

    fn matrix<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let normal = support::batch::<R>(client)?;
        exercise::<R>(client, &normal, &[4, 2, 4, 1, 0, 3], &[true; 6], None)?;
        let rectangles = rectangles::<R>(
            client,
            &[
                (2.0, 2.0),
                (2.0_f64.powi(500), 2.0_f64.powi(250)),
                (2.0_f64.powi(-500), 2.0_f64.powi(-250)),
                (f64::from_bits(4), 1.0),
            ],
        )?;
        let thin_owner = Tensor::from_slice(&[3_i64], &[1], client.device())?;
        let thin_depth = Tensor::from_slice(&[1.0_f64], &[1], client.device())?;
        let thin = rectangles.evaluate_depths_tensor(&thin_owner, &thin_depth)?;
        assert!(thin.area_m2().lower().item::<f64>()? <= f64::from_bits(4));
        assert!(thin.area_m2().upper().item::<f64>()? >= f64::from_bits(4));
        assert!(thin.hydrostatic_integral_m3().lower().item::<f64>()? <= f64::from_bits(2));
        assert!(thin.hydrostatic_integral_m3().upper().item::<f64>()? >= f64::from_bits(2));
        exercise::<R>(
            client,
            &rectangles,
            &[0, 3, 2, 1, 3],
            &[true, false, true, true, false],
            Some(&[
                4.0,
                f64::from_bits(4),
                2.0_f64.powi(-750),
                2.0_f64.powi(750),
                f64::from_bits(4),
            ]),
        )?;
        controls::run::<R>(client, &normal)
    }

    #[test]
    fn configured_cpu_domain_coverage_preserves_geometry_and_retained_inequalities() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let base = CpuClient::new(CpuDevice::new());
        let before = base.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = base.with_parallelism(config);
        matrix::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(base.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_domain_coverage_preserves_geometry_and_retained_inequalities() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
