use super::tests::{
    fixtures::{AnchorKind, CASES, Chart},
    parameter_rebinding_tests::{float_fields, integer_fields, snapshot},
    test_support as support,
};
use crate::channel::{flow::surveyed::error, section::HydraulicGeometry};
use numr::{
    dtype::DType,
    ops::{BinaryOps, UtilityOps},
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use std::{collections::BTreeMap, sync::Arc};
use support::{Client, Result, hydraulic};

#[path = "support_selection_tests_controls.rs"]
mod controls;

fn shape<R: Runtime<DType = DType>>(proved: &Tensor<R>, owner: &Tensor<R>, count: usize) {
    assert_eq!(proved.shape(), &[count]);
    assert_eq!(proved.dtype(), DType::F64);
    assert_eq!(owner.ndim(), 1);
    assert_eq!(owner.dtype(), DType::I64);
    assert!(proved.device().is_same(owner.device()));
}

fn matrix<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = support::batch::<R>(&client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let mut groups = BTreeMap::new();
    for case in CASES.iter().rev() {
        groups
            .entry((case.chart.kind, case.chart.add))
            .or_insert_with(Vec::new)
            .push(case);
    }
    let mut exercised = 0;
    for cases in groups.values() {
        let charts = cases.iter().map(|case| case.chart).collect::<Vec<_>>();
        for strided in [false, true] {
            let input = support::input::<R>(&client, &section, &charts, strided, false)?;
            let (strict, selection) = hydraulic(error::checked::<R, _>(
                &client,
                "all-supported selection equivalence",
                |a| {
                    let bands = geometry.gather(a, &input.owners)?;
                    let raw = support::query(a, &bands, &input)?;
                    let identities = Tensor::from_slice(
                        &(0..charts.len())
                            .map(|index| i64::try_from(index).map(|index| 1001 + index))
                            .collect::<std::result::Result<Vec<_>, _>>()
                            .map_err(|error| {
                                crate::channel::flow::surveyed::error::HydraulicError::Numerical {
                                    context: error.to_string(),
                                }
                            })?,
                        &[charts.len()],
                        client.device(),
                    )?;
                    client.copy_into(&raw.original_rows, &identities)?;
                    let rows = client.arange(0.0, charts.len() as f64, 1.0, DType::I64)?;
                    let strict = raw.select(a, &rows)?.prove_support(a, &bands)?;
                    Ok((strict, raw.select_supported(a, &bands)?))
                },
            ))?;
            let selection: super::SupportSelection<R> = selection;
            shape(&selection.proved, &selection.owner, charts.len());
            assert_eq!(
                selection.proved.try_to_vec::<f64>()?,
                vec![1.0; charts.len()]
            );
            assert_eq!(
                selection.owner.try_to_vec::<i64>()?,
                (0..charts.len())
                    .map(i64::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?
            );
            let selected = selection
                .query
                .ok_or("All-supported selection has no query")?;
            assert!(Arc::ptr_eq(&strict.query.model, &selected.query.model));
            let expected = snapshot(&strict)?;
            assert_eq!(snapshot(&selected)?, expected);
            client.copy_into(
                &input.owners,
                &Tensor::zeros(input.owners.shape(), DType::I64, client.device())?,
            )?;
            client.copy_into(
                &selection.proved,
                &Tensor::zeros(selection.proved.shape(), DType::F64, client.device())?,
            )?;
            client.copy_into(
                &selection.owner,
                &Tensor::zeros(selection.owner.shape(), DType::I64, client.device())?,
            )?;
            assert_eq!(snapshot(&selected)?, expected);
            assert_eq!(snapshot(&strict)?, expected);
            assert!(
                float_fields(&selected)
                    .iter()
                    .all(|field| field.is_contiguous())
            );
            assert!(
                integer_fields(&selected)
                    .iter()
                    .all(|field| field.is_contiguous())
            );
        }
        exercised += charts.len();
    }
    assert_eq!(exercised, 17);
    mixed::<R>(&client, &section)?;
    controls::run::<R>(&client, &section)
}

pub(super) fn rectangle_chart(kind: AnchorKind, add: bool, lower: f64, upper: f64) -> Chart {
    Chart {
        owner: 1,
        kind,
        band: 0,
        source: [2.0_f64.to_bits(), 0, 0],
        add,
        scale: [1.0_f64.to_bits(), 0, 0],
        parameter: [lower.to_bits(), upper.to_bits()],
    }
}

fn mixed<R>(
    client: &R::Client,
    section: &crate::channel::section::PrismaticSectionBatch<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    // The rectangle has width2 and cap depth3, so exact support is 0<=A<=6.
    let cases = [
        (
            AnchorKind::Zero,
            true,
            [(1.0, 2.0), (-1.0, -1.0), (7.0, 7.0)],
        ),
        (
            AnchorKind::Source,
            true,
            [(-0.5, 0.5), (-3.0, -3.0), (5.0, 5.0)],
        ),
        (
            AnchorKind::BandStart,
            true,
            [(1.0, 2.0), (-1.0, -1.0), (7.0, 7.0)],
        ),
        (
            AnchorKind::BandEnd,
            true,
            [(-2.0, -1.0), (-7.0, -7.0), (1.0, 1.0)],
        ),
        (
            AnchorKind::Cap,
            true,
            [(-2.0, -1.0), (-7.0, -7.0), (1.0, 1.0)],
        ),
        (
            AnchorKind::Zero,
            false,
            [(-2.0, -1.0), (1.0, 1.0), (-7.0, -7.0)],
        ),
        (
            AnchorKind::Source,
            false,
            [(-0.5, 0.5), (3.0, 3.0), (-5.0, -5.0)],
        ),
        (
            AnchorKind::BandEnd,
            false,
            [(1.0, 2.0), (7.0, 7.0), (-1.0, -1.0)],
        ),
    ];
    let geometry = HydraulicGeometry::from_batch(section);
    for (kind, add, bounds) in cases {
        let charts = bounds.map(|(lo, hi)| rectangle_chart(kind, add, lo, hi));
        let input = support::input::<R>(client, section, &charts, true, false)?;
        let (independent, selection) = hydraulic(error::checked::<R, _>(
            client,
            "mixed support selection",
            |a| {
                let bands = geometry.gather(a, &input.owners)?;
                let raw = support::query(a, &bands, &input)?;
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let selected_bands = bands.select(a, &rows)?;
                let independent = raw.select(a, &rows)?.prove_support(a, &selected_bands)?;
                Ok((independent, raw.select_supported(a, &bands)?))
            },
        ))?;
        shape(&selection.proved, &selection.owner, 3);
        assert_eq!(selection.proved.try_to_vec::<f64>()?, [1.0, 0.0, 0.0]);
        assert_eq!(selection.owner.try_to_vec::<i64>()?, [0]);
        let selected = selection
            .query
            .ok_or("Mixed support loses its admitted row")?;
        assert_eq!(snapshot(&selected)?, snapshot(&independent)?);
        assert!(Arc::ptr_eq(&selected.query.model, &independent.query.model));
    }
    Ok(())
}

#[test]
fn configured_cpu_support_selection_preserves_complete_certificates() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let base = CpuClient::new(CpuDevice::new());
    let before = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    matrix::<CpuRuntime>(client.clone())?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(base.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_support_selection_preserves_complete_certificates() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
