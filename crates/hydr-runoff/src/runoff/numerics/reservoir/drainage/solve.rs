//! Shared immutable excess-depth solve through configured solvr bisection.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

use super::admission::admit_drainage;
use super::algebra;
use super::budget::DrainageBudget;
use super::receipt::{DrainageResult, DrainageRoot};
use crate::runoff::ReservoirError;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedBisectionOptions};

pub(crate) fn drain<R>(
    prepared: &DrainageBudget<'_, R>,
) -> Result<DrainageResult<R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let context = prepared.context();
    let dt = prepared.interval();
    let c = &context.client;
    let (lower, upper, iterations) = if context.any(prepared.active())? {
        // Check the original physical upper endpoint before entering generic root work.
        let upper_rate = algebra::law(context, prepared.excess(), prepared.active())?;
        let upper_loss = c.mul(prepared.active_duration(), &upper_rate)?;
        context.finite(&upper_loss, "implicit upper-endpoint interval loss")?;
        let upper_residual = algebra::residual(context, prepared.excess(), prepared)?;
        context.finite(&upper_residual, "implicit upper-endpoint residual")?;
        let tolerance = c.add_scalar(
            &c.mul_scalar(prepared.excess(), context.config.root_relative_depth)?,
            context.config.root_absolute_depth,
        )?;
        context.finite(&tolerance, "initial root width tolerance")?;
        context.reject_mask(
            &c.mul(prepared.active(), &c.le(&tolerance, &context.zero)?)?,
            "initial root tolerance",
            &[("tolerance", &tolerance), ("excess", prepared.excess())],
            &[],
            "Root width tolerance underflows. Select a representable tolerance or rescale depths",
        )?;
        let options = BatchedBisectionOptions {
            atol: context.config.root_absolute_depth,
            rtol: context.config.root_relative_depth,
            max_iter: context.config.root_iterations,
            max_expansions: 0,
            expansion_factor: 2.0,
        };
        let result = c.bisect_batched(|x| algebra::residual(context, x, prepared), &context.zero, prepared.excess(), &options)
            .map_err(|error| {
                let message = format!("solvr bisect_batched: dt={dt}, atol={}, rtol={}, max_iter={}, max_expansions=0: {error}. Increase root work, select an admitted tolerance, or rescale depths", options.atol, options.rtol, options.max_iter);
                if matches!(error, solvr::optimize::OptimizeError::NumrError(_)) {
                    ReservoirError::Runtime { element: None, context: message }
                } else {
                    ReservoirError::numerical(message)
                }
            })?;
        (result.lower, result.upper, result.iterations)
    } else {
        (context.zero.clone(), context.zero.clone(), 0)
    };
    let depth = algebra::candidate_depth(context, prepared, &upper)?;
    admit_drainage(
        prepared,
        DrainageRoot {
            lower,
            upper,
            depth,
            iterations,
        },
    )
}
