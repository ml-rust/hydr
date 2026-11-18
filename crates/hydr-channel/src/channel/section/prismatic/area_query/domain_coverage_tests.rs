use super::test_support as support;
use super::{CoveredDomain, DomainCoverage, cover_domain};
use crate::channel::{
    flow::surveyed::error,
    section::{
        CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, HydraulicGeometry,
        PrismaticSectionBatch,
    },
};
use numr::{
    dtype::DType,
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use support::{Client, Result, hydraulic};

#[path = "domain_coverage_tests_certificates.rs"]
mod certificates;
#[path = "domain_coverage_tests_controls.rs"]
mod controls;

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
