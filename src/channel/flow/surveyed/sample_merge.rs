use super::{
    error::{Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection},
    sample_candidates::Candidate,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ReduceOps,
        ScalarOps, TensorOps, TwoSumOps, TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

pub(crate) struct SampledSolution<R: Runtime<DType = DType>> {
    pub(crate) root_work_limited: bool,
    pub(crate) root_work_limited_rows: Tensor<R>,
    pub(crate) water_flux: Interval<R>,
    pub(crate) momentum_flux: Interval<R>,
    pub(crate) trace_area: Interval<R>,
    pub(crate) trace_discharge: Interval<R>,
    pub(crate) trace_depth: Interval<R>,
    pub(crate) trace_dry_possible: Tensor<R>,
    pub(crate) trace_wet_possible: Tensor<R>,
    pub(crate) wave_speed: Interval<R>,
}

pub(crate) struct Samples<R: Runtime<DType = DType>> {
    pub(crate) solution: SampledSolution<R>,
    present: Tensor<R>,
}

impl<R> Samples<R>
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
        let zero = Tensor::zeros(&[count], DType::F64, a.client().device())?;
        Ok(Self {
            solution: SampledSolution {
                root_work_limited: false,
                root_work_limited_rows: zero.clone(),
                water_flux: a.point(&zero)?,
                momentum_flux: a.point(&zero)?,
                trace_area: a.point(&zero)?,
                trace_discharge: a.point(&zero)?,
                trace_depth: a.point(&zero)?,
                trace_dry_possible: zero.clone(),
                trace_wet_possible: zero.clone(),
                wave_speed: a.point(&zero)?,
            },
            present: zero,
        })
    }

    pub(crate) fn root_status(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
        status: &Tensor<R>,
    ) -> HydraulicResult<()> {
        if status.numel() == 0 {
            return Ok(());
        }
        let c = a.client();
        let code = Tensor::from_slice(&[4_i64], &[], c.device())?;
        let limited_rows = c.cast(&c.eq(status, &code)?, DType::F64)?;
        let previous = c.index_select(&self.solution.root_work_limited_rows, 0, rows)?;
        self.solution.root_work_limited_rows = c.index_put(
            &self.solution.root_work_limited_rows,
            0,
            rows,
            &c.maximum(&previous, &limited_rows)?,
        )?;
        let limited = c
            .cast(&c.any(&limited_rows, &[], false)?, DType::F64)?
            .item::<f64>()?
            != 0.0;
        self.solution.root_work_limited |= limited;
        Ok(())
    }

    pub(crate) fn merge(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
        candidate: Candidate<R>,
    ) -> HydraulicResult<()> {
        let c = a.client();
        let present = c.index_select(&self.present, 0, rows)?;
        let choose =
            |old: &Tensor<R>, new: &Tensor<R>, lower: bool| -> HydraulicResult<Tensor<R>> {
                let old_rows = c.index_select(old, 0, rows)?;
                let union = if lower {
                    c.minimum(&old_rows, new)?
                } else {
                    c.maximum(&old_rows, new)?
                };
                let merged = c.where_cond(&present, &union, new)?;
                let selected = c.where_cond(&candidate.possible, &merged, &old_rows)?;
                Ok(c.index_put(old, 0, rows, &selected)?)
            };
        let merge_interval = |a: &mut Arithmetic<'_, R>,
                              old: &Interval<R>,
                              new: &Interval<R>|
         -> HydraulicResult<Interval<R>> {
            a.bounds(
                &choose(old.lower(), new.lower(), true)?,
                &choose(old.upper(), new.upper(), false)?,
            )
            .map_err(Into::into)
        };
        self.solution.water_flux =
            merge_interval(a, &self.solution.water_flux, &candidate.water_flux)?;
        self.solution.momentum_flux =
            merge_interval(a, &self.solution.momentum_flux, &candidate.momentum_flux)?;
        self.solution.trace_area = merge_interval(a, &self.solution.trace_area, &candidate.area)?;
        self.solution.trace_discharge =
            merge_interval(a, &self.solution.trace_discharge, &candidate.discharge)?;
        self.solution.trace_depth =
            merge_interval(a, &self.solution.trace_depth, &candidate.depth)?;
        for (old, new) in [
            (
                &mut self.solution.trace_dry_possible,
                candidate.dry_possible,
            ),
            (
                &mut self.solution.trace_wet_possible,
                candidate.wet_possible,
            ),
        ] {
            let gathered = c.index_select(old, 0, rows)?;
            let active = c.mul(&candidate.possible, &new)?;
            *old = c.index_put(old, 0, rows, &c.maximum(&gathered, &active)?)?;
        }
        self.present = c.index_put(
            &self.present,
            0,
            rows,
            &c.maximum(&present, &candidate.possible)?,
        )?;
        Ok(())
    }

    pub(crate) fn wave(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
        speed: &Interval<R>,
    ) -> HydraulicResult<()> {
        let c = a.client();
        let lower = c.minimum(
            &c.index_select(self.solution.wave_speed.lower(), 0, rows)?,
            speed.lower(),
        )?;
        let upper = c.maximum(
            &c.index_select(self.solution.wave_speed.upper(), 0, rows)?,
            speed.upper(),
        )?;
        self.solution.wave_speed = a.bounds(
            &c.index_put(self.solution.wave_speed.lower(), 0, rows, &lower)?,
            &c.index_put(self.solution.wave_speed.upper(), 0, rows, &upper)?,
        )?;
        Ok(())
    }

    pub(crate) fn finish(self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<SampledSolution<R>> {
        let c = a.client();
        if self.present.numel() != 0 {
            let zero = Tensor::zeros(&[], DType::F64, c.device())?;
            if c.cast(
                &c.any(&c.eq(&self.present, &zero)?, &[], false)?,
                DType::F64,
            )?
            .item::<f64>()?
                != 0.0
            {
                let reason = if self.solution.root_work_limited {
                    PrismaticRiemannRejection::WorkLimit
                } else {
                    PrismaticRiemannRejection::SamplingUnresolved
                };
                return Err(HydraulicError::Rejected { reason, query: None, section_owner: None, context: "Interface sample has no certified candidate. Review retained root status and wave enclosures".into() });
            }
        }
        Ok(self.solution)
    }
}
