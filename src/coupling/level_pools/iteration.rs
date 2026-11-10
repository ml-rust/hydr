use super::{
    CoupledLevelPoolError, context::Context, exchange::Exchange, heads::Heads, topology::reject,
};
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, AndersonConvergence, AndersonOptions, RootOptions,
};
use std::cell::RefCell;

pub(super) struct Evaluation<R: Runtime<DType = DType>> {
    pub pools: LevelPoolBatch<R>,
    pub exchange: Exchange<R>,
    pub heads: Heads<R>,
    pub packet: Tensor<R>,
}
impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    pub fn evaluate(
        &self,
        pools: &LevelPoolBatch<R>,
        end: f64,
        x: &Tensor<R>,
    ) -> Result<Evaluation<R>, CoupledLevelPoolError> {
        self.check_packet(x)?;
        let n = self.pool_count();
        let e = self.edge_count();
        let start = pools.stages_tensor()?;
        let start_wet = self.wet(pools.volumes_tensor())?;
        let estimate = x.narrow(0, 0, n)?;
        let estimate_wet = x.narrow(0, n + e, n)?;
        let heads = self.midpoint_heads(&start, &estimate, &start_wet, &estimate_wet)?;
        let result = self.structures.evaluate_heads(
            &heads.a,
            &heads.b,
            &self.bed_a,
            &self.bed_b,
            &heads.wet_a,
            &heads.wet_b,
        )?;
        let exchange = self.exchange(
            self.structures.held_volumes(&result, end - pools.time())?,
            pools.volumes_tensor(),
        )?;
        let next = pools
            .trial_to_tensors(end, &exchange.incoming, &exchange.outgoing)?
            .into_next();
        let next_stage = next.stages_tensor()?;
        let next_wet = self.wet(next.volumes_tensor())?;
        let next_heads = self.midpoint_heads(&start, &next_stage, &start_wet, &next_wet)?;
        let packet = self.packet(&next_stage, &exchange.signed, &next_wet, &next_heads)?;
        // Packet support is exact. The evaluated heads remain the accepted law operands.
        Ok(Evaluation {
            pools: next,
            exchange,
            heads,
            packet,
        })
    }
    pub fn initial_packet(
        &self,
        pools: &LevelPoolBatch<R>,
    ) -> Result<Tensor<R>, CoupledLevelPoolError> {
        let stage = pools.stages_tensor()?;
        let wet = self.wet(pools.volumes_tensor())?;
        let heads = self.heads(&stage, &wet)?;
        let volumes = Tensor::<R>::zeros(&[self.edge_count()], DType::F64, self.client.device())?;
        self.packet(&stage, &volumes, &wet, &heads)
    }
    pub fn residuals(
        &self,
        x: &Tensor<R>,
        gx: &Tensor<R>,
    ) -> Result<(f64, f64, bool), CoupledLevelPoolError> {
        let n = self.pool_count();
        let e = self.edge_count();
        let c = &self.client;
        let residual = c.abs(&c.sub(gx, x)?)?;
        let head = c
            .max(&residual.narrow(0, 0, n)?, &[0], false)?
            .item::<f64>()?;
        let volume = c
            .max(&residual.narrow(0, n, e)?, &[0], false)?
            .item::<f64>()?;
        let flags = c
            .max(&residual.narrow(0, n + e, n + 2 * e)?, &[0], false)?
            .item::<f64>()?
            == 0.0;
        if !head.is_finite() || !volume.is_finite() {
            return Err(CoupledLevelPoolError::numerical(
                "iteration residual overflows. Rescale heads or storage",
            ));
        }
        Ok((head, volume, flags))
    }
    pub fn check_endpoint(&self, evaluation: &Evaluation<R>) -> Result<(), CoupledLevelPoolError> {
        let c = &self.client;
        let stage = evaluation.pools.stages_tensor()?;
        let wet = self.wet(evaluation.pools.volumes_tensor())?;
        let heads = self.heads(&stage, &wet)?;
        self.structures.evaluate_heads(
            &heads.a,
            &heads.b,
            &self.bed_a,
            &self.bed_b,
            &heads.wet_a,
            &heads.wet_b,
        )?;
        let signed = &evaluation.exchange.signed;
        let crossing_a = c.mul(&c.gt(signed, &self.zero)?, &c.le(&heads.a, &self.cutoff_a)?)?;
        let crossing_b = c.mul(&c.lt(signed, &self.zero)?, &c.le(&heads.b, &self.cutoff_b)?)?;
        let origin_a = c.index_select(&self.origin, 0, &self.a)?;
        let origin_b = c.index_select(&self.origin, 0, &self.b)?;
        let raised_a = c.gt(&self.cutoff_a, &origin_a)?;
        let raised_b = c.gt(&self.cutoff_b, &origin_b)?;
        reject(
            c,
            &c.maximum(
                &c.mul(&crossing_a, &raised_a)?,
                &c.mul(&crossing_b, &raised_b)?,
            )?,
            "active transfer crosses a raised local bed or law cutoff. Shorten the interval",
        )?;
        Ok(())
    }
    pub fn mass_check(
        &self,
        before: &LevelPoolBatch<R>,
        after: &LevelPoolBatch<R>,
        assembly: f64,
    ) -> Result<f64, CoupledLevelPoolError> {
        let old = before.totals()?;
        let new = after.totals()?;
        let interval = new.volume_m3 - old.volume_m3;
        let cumulative = new.volume_m3 - new.initial_volume_m3;
        let scale = old.physical_scale_m3.max(new.physical_scale_m3);
        let budget = self.setup.config.mass_absolute_m3 + self.setup.config.mass_relative * scale;
        let charge = |a, b| {
            crate::soil::admission::add_adjustment_charge(a, b)
                .map_err(|e| CoupledLevelPoolError::numerical(e.to_string()))
        };
        let interval_charge = charge(
            charge(
                charge(interval.abs(), old.reduction_bound_m3)?,
                new.reduction_bound_m3,
            )?,
            assembly,
        )?;
        let cumulative_charge =
            charge(charge(cumulative.abs(), new.reduction_bound_m3)?, assembly)?;
        if !budget.is_finite()
            || !interval_charge.is_finite()
            || !cumulative_charge.is_finite()
            || interval_charge > budget
            || cumulative_charge > budget
        {
            return Err(CoupledLevelPoolError::numerical(format!(
                "closed mass drift interval={interval}, cumulative={cumulative}, assembly={assembly}, budget={budget}: shorten the interval or inspect physical scales"
            )));
        }
        Ok(interval)
    }
}

pub(super) struct SolvedInterval<R: Runtime<DType = DType>> {
    pub input: Tensor<R>,
    pub evaluation: Evaluation<R>,
    pub iterations: usize,
    pub head_residual: f64,
    pub transfer_residual: f64,
}

pub(super) fn solve<R>(
    context: &Context<R>,
    pools: &LevelPoolBatch<R>,
    end: f64,
) -> Result<SolvedInterval<R>, CoupledLevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>,
{
    let error = RefCell::new(None);
    let map = |x: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        match context.evaluate(pools, end, x) {
            Ok(evaluation) => Ok(evaluation.packet),
            Err(e) => {
                let mut slot =
                    error
                        .try_borrow_mut()
                        .map_err(|_| numr::error::Error::InvalidArgument {
                            arg: "coupled error channel",
                            reason: "callback diagnostic channel is already borrowed".into(),
                        })?;
                if slot.is_none() {
                    *slot = Some(e);
                }
                Err(numr::error::Error::InvalidArgument {
                    arg: "coupled pool callback",
                    reason: "hydraulic map rejected the original candidate".into(),
                })
            }
        }
    };
    let config = &context.setup.config;
    let options = AndersonOptions {
        m: 0,
        alpha: 1.0,
        base: RootOptions {
            max_iter: config.iteration_limit,
            ..RootOptions::default()
        },
    };
    let initial = context.initial_packet(pools)?;
    let root = context
        .client
        .anderson_with_convergence(map, &initial, &options, |x, gx, _| {
            match context.residuals(x, gx) {
                Ok((h, v, flags)) => Ok(AndersonConvergence {
                    residual_norm: (h / config.head_absolute_m)
                        .max(v / config.transfer_absolute_m3),
                    converged: flags
                        && h <= config.head_absolute_m
                        && v <= config.transfer_absolute_m3,
                }),
                Err(e) => {
                    let mut slot = error.try_borrow_mut().map_err(|_| {
                        numr::error::Error::InvalidArgument {
                            arg: "coupled error channel",
                            reason: "callback diagnostic channel is already borrowed".into(),
                        }
                    })?;
                    if slot.is_none() {
                        *slot = Some(e);
                    }
                    Err(numr::error::Error::InvalidArgument {
                        arg: "coupled pool convergence",
                        reason: "dimensional convergence check rejected the candidate".into(),
                    })
                }
            }
        });
    if let Some(error) = error.into_inner() {
        return Err(error);
    }
    let root = root?;
    if !root.converged {
        return Err(CoupledLevelPoolError::numerical(format!(
            "fixed-point iteration exhausted {} updates. Shorten the interval",
            config.iteration_limit
        )));
    }
    let final_evaluation = context.evaluate(pools, end, &root.x)?;
    let (head, volume, flags) = context.residuals(&root.x, &final_evaluation.packet)?;
    if !flags || head > config.head_absolute_m || volume > config.transfer_absolute_m3 {
        return Err(CoupledLevelPoolError::numerical(
            "reevaluated final iterate exceeds dimensional budgets or changes physical support",
        ));
    }
    context.check_endpoint(&final_evaluation)?;
    Ok(SolvedInterval {
        input: root.x,
        evaluation: final_evaluation,
        iterations: root.iterations,
        head_residual: head,
        transfer_residual: volume,
    })
}
