use super::{
    equation_ops::select_interval,
    error::{Arithmetic, HydraulicError, HydraulicResult},
    sample_merge::{SampledSolution, Samples},
    types::CapturedInput,
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps, TypeConversionOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

pub(crate) struct Refinement<R: Runtime<DType = DType>> {
    pending: Tensor<R>,
    completed: SampledSolution<R>,
}

impl<R> Refinement<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn new(a: &mut Arithmetic<'_, R>, count: usize) -> HydraulicResult<Self> {
        Ok(Self {
            pending: a.client().arange(0.0, count as f64, 1.0, DType::I64)?,
            completed: Samples::new(a, count)?.solution,
        })
    }

    pub(crate) fn pending(&self) -> &Tensor<R> {
        &self.pending
    }

    pub(crate) fn input(
        &self,
        a: &mut Arithmetic<'_, R>,
        input: &CapturedInput<R>,
    ) -> HydraulicResult<CapturedInput<R>> {
        select_input(a, input, &self.pending)
    }

    pub(crate) fn retain(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        solution: &SampledSolution<R>,
        accepted: &Tensor<R>,
    ) -> HydraulicResult<Tensor<R>> {
        let c = a.client();
        let rows = c.nonzero(accepted)?.flatten()?;
        if rows.numel() != 0 {
            let original = c.index_select(&self.pending, 0, &rows)?;
            let scatter = |old: &Tensor<R>, new: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
                Ok(c.index_put(old, 0, &original, &c.index_select(new, 0, &rows)?)?)
            };
            let interval = |a: &mut Arithmetic<'_, R>, old: &Interval<R>, new: &Interval<R>| {
                a.bounds(
                    &scatter(old.lower(), new.lower())?,
                    &scatter(old.upper(), new.upper())?,
                )
                .map_err(HydraulicError::from)
            };
            self.completed.water_flux =
                interval(a, &self.completed.water_flux, &solution.water_flux)?;
            self.completed.momentum_flux =
                interval(a, &self.completed.momentum_flux, &solution.momentum_flux)?;
            self.completed.trace_area =
                interval(a, &self.completed.trace_area, &solution.trace_area)?;
            self.completed.trace_discharge = interval(
                a,
                &self.completed.trace_discharge,
                &solution.trace_discharge,
            )?;
            self.completed.trace_depth =
                interval(a, &self.completed.trace_depth, &solution.trace_depth)?;
            self.completed.wave_speed =
                interval(a, &self.completed.wave_speed, &solution.wave_speed)?;
            self.completed.trace_dry_possible = scatter(
                &self.completed.trace_dry_possible,
                &solution.trace_dry_possible,
            )?;
            self.completed.trace_wet_possible = scatter(
                &self.completed.trace_wet_possible,
                &solution.trace_wet_possible,
            )?;
            self.completed.root_work_limited_rows = scatter(
                &self.completed.root_work_limited_rows,
                &solution.root_work_limited_rows,
            )?;
            self.completed.root_work_limited = c
                .cast(
                    &c.any(&self.completed.root_work_limited_rows, &[], false)?,
                    DType::F64,
                )?
                .item::<f64>()?
                != 0.0;
        }
        let zero = Tensor::zeros(&[], DType::F64, c.device())?;
        let rejected = c.nonzero(&c.eq(accepted, &zero)?)?.flatten()?;
        self.pending = c.index_select(&self.pending, 0, &rejected)?;
        Ok(rejected)
    }

    pub(crate) fn finish(self) -> HydraulicResult<SampledSolution<R>> {
        if self.pending.numel() != 0 {
            return Err(HydraulicError::Numerical {
                context: "Refinement retains unresolved rows. Complete flux acceptance before returning the solution".into(),
            });
        }
        Ok(self.completed)
    }

    pub(crate) fn map_error(&self, mut error: HydraulicError) -> HydraulicResult<HydraulicError> {
        if let HydraulicError::Rejected {
            query: Some(row), ..
        } = &mut error
        {
            if *row >= self.pending.numel() {
                return Err(HydraulicError::Numerical {
                    context: format!(
                        "Rejected query row {row} exceeds the pending row count {}. Check rejection row identity",
                        self.pending.numel()
                    ),
                });
            }
            let original = self.pending.narrow(0, *row, 1)?.item::<i64>()?;
            *row = usize::try_from(original).map_err(|_| HydraulicError::Numerical {
                context: format!("Original query row {original} is invalid. Rebuild the query"),
            })?;
        }
        Ok(error)
    }
}

pub(crate) fn select_input<R>(
    a: &mut Arithmetic<'_, R>,
    input: &CapturedInput<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<CapturedInput<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    Ok(CapturedInput {
        owners: c.index_select(&input.owners, 0, rows)?,
        left_area: c.index_select(&input.left_area, 0, rows)?,
        left_discharge: c.index_select(&input.left_discharge, 0, rows)?,
        right_area: c.index_select(&input.right_area, 0, rows)?,
        right_discharge: c.index_select(&input.right_discharge, 0, rows)?,
    })
}

pub(crate) fn select_solution<R>(
    a: &mut Arithmetic<'_, R>,
    solution: &SampledSolution<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<SampledSolution<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let root_work_limited_rows = c.index_select(&solution.root_work_limited_rows, 0, rows)?;
    let root_work_limited = if rows.numel() == 0 {
        false
    } else {
        c.cast(&c.any(&root_work_limited_rows, &[], false)?, DType::F64)?
            .item::<f64>()?
            != 0.0
    };
    Ok(SampledSolution {
        root_work_limited,
        root_work_limited_rows,
        water_flux: select_interval(a, &solution.water_flux, rows)?,
        momentum_flux: select_interval(a, &solution.momentum_flux, rows)?,
        trace_area: select_interval(a, &solution.trace_area, rows)?,
        trace_discharge: select_interval(a, &solution.trace_discharge, rows)?,
        trace_depth: select_interval(a, &solution.trace_depth, rows)?,
        trace_dry_possible: c.index_select(&solution.trace_dry_possible, 0, rows)?,
        trace_wet_possible: c.index_select(&solution.trace_wet_possible, 0, rows)?,
        wave_speed: select_interval(a, &solution.wave_speed, rows)?,
    })
}

#[cfg(test)]
mod tests {
    use super::super::controller::resolve_width_error;
    use super::super::error::{PrismaticRiemannRejection, checked, into_public};
    use super::*;

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
                let tensor = |values: &[f64]| {
                    Tensor::<R>::from_slice(values, &[values.len()], client.device())
                };
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
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
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
}
