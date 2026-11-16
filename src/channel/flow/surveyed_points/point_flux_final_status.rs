use super::{
    point_flux_faces::{Continuation, Faces},
    point_flux_masks::{code, equals, flag},
    point_flux_types::PointFluxStatus,
    point_quadrature_types::PointQuadratureOutcome as Outcome,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

impl<R> Faces<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Records `LevelLimit` for each over-budget face whose possible continuation last
    /// advanced to `last_level`, the final admissible level. No further advance can run there.
    pub(super) fn close_ceiling(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        last_level: usize,
    ) -> HydraulicResult<()> {
        let c = a.client();
        let shape = self.status.shape().to_vec();
        let budget = equals(a, &self.status, PointFluxStatus::FluxBudgetExceeded as i64)?;
        let limit = Tensor::full_scalar(
            &shape,
            DType::I64,
            Outcome::LevelLimit as i64 as f64,
            c.device(),
        )?;
        let zero = Tensor::zeros(&shape, a.dtype(), c.device())?;
        let last = i64::try_from(last_level)
            .map_err(|_| super::point_root_session::invalid("last panel level exceeds i64"))?;
        for continuation in Continuation::ALL {
            let index = continuation.index();
            let advanced = equals(a, &self.outcome[index], Outcome::Advanced as i64)?;
            let ceiling = flag(a, &c.ge(&self.level[index], &code(a, last)?)?)?;
            let mask = c.mul(
                &c.mul(&budget, &self.possible[index])?,
                &c.mul(&advanced, &ceiling)?,
            )?;
            let closed = c.gt(&mask, &zero)?;
            self.outcome[index] = c.where_cond(&closed, &limit, &self.outcome[index])?;
            self.terminal[index] = c.maximum(&self.terminal[index], &mask)?;
            if let Continuation::Fan(_) = continuation {
                self.sonic_outcome = c.where_cond(&closed, &limit, &self.sonic_outcome)?;
            }
        }
        Ok(())
    }

    /// Maps each face still over budget to the reason its possible continuations stopped.
    /// A work ceiling outranks an unresolved root. Every other stop stays `FluxBudgetExceeded`.
    pub(super) fn final_status(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<Tensor<R>> {
        let c = a.client();
        let shape = self.status.shape();
        let full = |value: PointFluxStatus| {
            Tensor::full_scalar(shape, DType::I64, value as i64 as f64, c.device())
        };
        let zero = Tensor::zeros(shape, a.dtype(), c.device())?;
        let (mut work, mut unresolved) = (zero.clone(), zero.clone());
        for continuation in Continuation::ALL {
            let index = continuation.index();
            let possible = &self.possible[index];
            let outcome = &self.outcome[index];
            let limited = c.mul(possible, &equals(a, outcome, Outcome::WorkLimit as i64)?)?;
            work = c.maximum(&work, &limited)?;
            let open = c.mul(possible, &equals(a, outcome, Outcome::Unresolved as i64)?)?;
            unresolved = c.maximum(&unresolved, &open)?;
        }
        let reason = c.where_cond(
            &c.gt(&work, &zero)?,
            &full(PointFluxStatus::WorkLimit)?,
            &c.where_cond(
                &c.gt(&unresolved, &zero)?,
                &full(PointFluxStatus::RootBracketUnresolved)?,
                &full(PointFluxStatus::FluxBudgetExceeded)?,
            )?,
        )?;
        Ok(c.where_cond(
            &c.eq(
                &self.status,
                &code(a, PointFluxStatus::FluxBudgetExceeded as i64)?,
            )?,
            &reason,
            &self.status,
        )?)
    }
}
