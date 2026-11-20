use super::state::{Accepted, BoundForcing, NativeEvaluation};
use super::{ChannelPoolError, context_types::Context};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, MidpointOps,
    ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{AndersonAlgorithms, BatchedBisectionAlgorithms};
use solvr::optimize::roots::{AndersonConvergence, AndersonOptions, RootOptions};
use std::cell::RefCell;

pub(super) struct SolvedInterval<R: Runtime<DType = DType>> {
    pub evaluation: NativeEvaluation<R>,
    pub iterations: usize,
}
pub(super) fn solve<R: Runtime<DType = DType>>(
    context: &Context<R>,
    accepted: &Accepted<R>,
    end: f64,
    forcing: &BoundForcing<R>,
) -> Result<SolvedInterval<R>, ChannelPoolError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + numr::ops::TensorOps<R>
        + numr::ops::NextRepresentableOps<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>
        + Interp1dAlgorithms<R>
        + BatchedBisectionAlgorithms<R>
        + AndersonAlgorithms<R>,
{
    let error = RefCell::new(None);
    let record = |e: ChannelPoolError| -> numr::error::Error {
        match error.try_borrow_mut() {
            Ok(mut slot) => {
                if slot.is_none() {
                    *slot = Some(e);
                }
            }
            Err(_) => {
                return numr::error::Error::InvalidArgument {
                    arg: "channel/pool diagnostic channel",
                    reason: "callback diagnostic channel is already borrowed".into(),
                };
            }
        }
        numr::error::Error::InvalidArgument {
            arg: "channel/pool fixed-point map",
            reason: "hydraulic map rejected the complete candidate".into(),
        }
    };
    // The map keeps its latest evaluation, whose input packet is its iterate. With no history
    // and unit mixing, the root is the last iterate, so its evaluation is the candidate.
    let latest: RefCell<Option<NativeEvaluation<R>>> = RefCell::new(None);
    let map = |x: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        let evaluation = context
            .evaluate(accepted, end, forcing, x)
            .map_err(record)?;
        let output = evaluation.interval.output_packet.clone();
        *latest.borrow_mut() = Some(evaluation);
        Ok(output)
    };
    let options = AndersonOptions {
        m: 0,
        alpha: 1.0,
        base: RootOptions {
            max_iter: context.setup.config.iteration_limit,
            ..RootOptions::default()
        },
    };
    let initial = context.initial_packet(&accepted.pools)?;
    let root = context
        .client
        .anderson_with_convergence(map, &initial, &options, |x, gx, _| {
            context
                .residuals(x, gx)
                .map(|residual| AndersonConvergence {
                    residual_norm: residual.norm,
                    converged: residual.converged,
                })
                .map_err(record)
        });
    if let Some(error) = error.into_inner() {
        return Err(error);
    }
    let root = root?;
    if !root.converged {
        return Err(ChannelPoolError::numerical(format!(
            "fixed-point iteration exhausted {} updates. Shorten the complete interval",
            context.setup.config.iteration_limit
        )));
    }
    // Reevaluation repeats the same calculation on the same operands, so it runs only when
    // the root is not the latest evaluated iterate.
    let evaluation = match latest.into_inner() {
        Some(evaluation)
            if evaluation
                .interval
                .input_packet
                .shares_storage_with(&root.x) =>
        {
            evaluation
        }
        _ => context.evaluate(accepted, end, forcing, &root.x)?,
    };
    if !context
        .residuals(&root.x, &evaluation.interval.output_packet)?
        .converged
    {
        return Err(ChannelPoolError::numerical(
            "final reevaluation exceeds head or transfer budgets, or changes support. Shorten the complete interval",
        ));
    }
    Ok(SolvedInterval {
        evaluation,
        iterations: root.iterations,
    })
}
