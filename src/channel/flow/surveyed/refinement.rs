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
