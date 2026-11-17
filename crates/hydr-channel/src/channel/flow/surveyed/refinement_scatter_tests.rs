use super::super::controller::resolve_width_error;
use super::super::error::{
    Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection, checked, into_public,
};
use super::super::refinement::{Refinement, select_solution};
use super::super::sample_merge::{SampledSolution, Samples};
use super::super::types::CapturedInput;
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn marked<R>(a: &mut Arithmetic<'_, R>, markers: &[f64]) -> HydraulicResult<SampledSolution<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let mut result = Samples::new(a, markers.len())?.solution;
    for (field, interval) in [
        &mut result.water_flux,
        &mut result.momentum_flux,
        &mut result.trace_area,
        &mut result.trace_discharge,
        &mut result.trace_depth,
        &mut result.wave_speed,
    ]
    .into_iter()
    .enumerate()
    {
        let lower: Vec<f64> = markers
            .iter()
            .map(|value| value + field as f64 * 1000.0)
            .collect();
        let upper: Vec<f64> = lower.iter().map(|value| value + 0.5).collect();
        *interval = a.bounds(
            &Tensor::from_slice(&lower, &[markers.len()], a.client().device())?,
            &Tensor::from_slice(&upper, &[markers.len()], a.client().device())?,
        )?;
    }
    Ok(result)
}

fn rejected(row: usize) -> HydraulicError {
    HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::FluxBudgetExceeded,
        query: Some(row),
        section_owner: Some(7),
        context: "retained width rejection".into(),
    }
}

fn contracts<R>(client: &R::Client) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(
        client,
        "pending rows preserve every completed field and original identity",
        |a| {
            let tensor =
                |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], client.device());
            let mut first = marked(a, &[10.0, 20.0, 30.0, 40.0])?;
            first.trace_dry_possible = tensor(&[0.0, 1.0, 1.0, 0.0])?;
            first.trace_wet_possible = tensor(&[1.0, 0.0, 0.0, 1.0])?;
            first.root_work_limited_rows = tensor(&[1.0, 0.0, 0.0, 1.0])?;
            first.root_work_limited = true;
            let mut retained = Refinement::new(a, 4)?;
            let accepted = tensor(&[1.0, 0.0, 1.0, 0.0])?;
            assert_eq!(
                retained.retain(a, &first, &accepted)?.try_to_vec::<i64>()?,
                [1, 3]
            );
            assert_eq!(retained.pending().try_to_vec::<i64>()?, [1, 3]);
            let mapped = retained.map_error(rejected(1))?;
            assert!(matches!(
                mapped,
                HydraulicError::Rejected {
                    query: Some(3),
                    section_owner: Some(7),
                    ..
                }
            ));
            assert!(matches!(
                retained.map_error(rejected(2)),
                Err(HydraulicError::Numerical { .. })
            ));
            let input = CapturedInput {
                owners: Tensor::from_slice(&[2_i64, 7, 2, 1], &[4], client.device())?,
                left_area: tensor(&[10.0, 20.0, 30.0, 40.0])?,
                left_discharge: tensor(&[11.0, 21.0, 31.0, 41.0])?,
                right_area: tensor(&[12.0, 22.0, 32.0, 42.0])?,
                right_discharge: tensor(&[13.0, 23.0, 33.0, 43.0])?,
            };
            let compact = retained.input(a, &input)?;
            assert_eq!(compact.owners.try_to_vec::<i64>()?, [7, 1]);
            assert_eq!(compact.left_area.try_to_vec::<f64>()?, [20.0, 40.0]);
            assert_eq!(compact.left_discharge.try_to_vec::<f64>()?, [21.0, 41.0]);
            assert_eq!(compact.right_area.try_to_vec::<f64>()?, [22.0, 42.0]);
            assert_eq!(compact.right_discharge.try_to_vec::<f64>()?, [23.0, 43.0]);
            let rows = Tensor::from_slice(&[1_i64, 3], &[2], client.device())?;
            let selected = select_solution(a, &first, &rows)?;
            assert_eq!(
                selected.root_work_limited_rows.try_to_vec::<f64>()?,
                [0.0, 1.0]
            );
            let unrelated = resolve_width_error(rejected(0), &selected)?;
            assert!(matches!(
                unrelated,
                HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::FluxBudgetExceeded,
                    query: Some(0),
                    ..
                }
            ));
            let limited = resolve_width_error(rejected(1), &selected)?;
            assert!(matches!(
                limited,
                HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                }
            ));
            let mut last = marked(a, &[120.0, 140.0])?;
            last.trace_dry_possible = tensor(&[1.0, 0.0])?;
            last.trace_wet_possible = tensor(&[0.0, 1.0])?;
            last.root_work_limited_rows = tensor(&[0.0, 1.0])?;
            last.root_work_limited = true;
            retained.retain(a, &last, &tensor(&[1.0; 2])?)?;
            let finished = retained.finish()?;
            for (field, interval) in [
                &finished.water_flux,
                &finished.momentum_flux,
                &finished.trace_area,
                &finished.trace_discharge,
                &finished.trace_depth,
                &finished.wave_speed,
            ]
            .into_iter()
            .enumerate()
            {
                let expected =
                    [10.0, 120.0, 30.0, 140.0].map(|value| value + field as f64 * 1000.0);
                assert_eq!(interval.lower().try_to_vec::<f64>()?, expected);
                assert_eq!(
                    interval.upper().try_to_vec::<f64>()?,
                    expected.map(|value| value + 0.5)
                );
            }
            assert_eq!(
                finished.trace_dry_possible.try_to_vec::<f64>()?,
                [0.0, 1.0, 1.0, 0.0]
            );
            assert_eq!(
                finished.trace_wet_possible.try_to_vec::<f64>()?,
                [1.0, 0.0, 0.0, 1.0]
            );
            assert_eq!(
                finished.root_work_limited_rows.try_to_vec::<f64>()?,
                [1.0, 0.0, 0.0, 1.0]
            );
            assert!(finished.root_work_limited);
            let mut partial = Refinement::new(a, 4)?;
            partial.retain(a, &first, &accepted)?;
            assert!(matches!(
                partial.finish(),
                Err(HydraulicError::Numerical { .. })
            ));
            Ok(())
        },
    )
}

#[test]
fn compaction_scatter_and_rejections_retain_original_rows_cpu()
-> Result<(), super::super::PrismaticRiemannError> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    contracts::<CpuRuntime>(&client).map_err(|error| into_public(error, Default::default()))
}

#[cfg(feature = "cuda")]
#[test]
fn compaction_scatter_and_rejections_retain_original_rows_cuda()
-> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    contracts::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
        .map_err(|error| into_public(error, Default::default()))?;
    Ok(())
}
