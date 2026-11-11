use super::{
    acceptance::accept,
    config::PrismaticRiemannConfig,
    curves::prepare_side,
    error::{HydraulicError, HydraulicResult, PrismaticRiemannRejection, checked},
    sample_merge::Samples,
    sampling::{SampledSolution, sample},
    types::{CapturedInput, Oracle},
    work::WorkLedger,
};
use crate::channel::section::HydraulicGeometry;
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};
use solvr::{integrate::CertifiedSimpsonAlgorithms, optimize::CertifiedBisectionAlgorithms};

pub(crate) fn solve<R>(
    geometry: &HydraulicGeometry<R>,
    input: &CapturedInput<R>,
    config: &PrismaticRiemannConfig,
    work: &WorkLedger,
) -> HydraulicResult<SampledSolution<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = geometry.client();
    if input.owners.numel() == 0 {
        return checked(c, "surveyed empty Riemann admission", |a| {
            Samples::new(a, 0)?.finish(a)
        });
    }
    let bands = checked(c, "surveyed gather", |a| geometry.gather(a, &input.owners))?;
    let mut panels = config.initial_simpson_panels;
    let mut refinement = 0;
    loop {
        work.begin_refinement(panels, refinement)?;
        let initial_work = work.snapshot();
        let attempt = (|| {
            let left = prepare_side(
                geometry,
                &bands,
                &input.left_area,
                &input.left_discharge,
                config,
                panels,
                work,
            )?;
            let right = prepare_side(
                geometry,
                &bands,
                &input.right_area,
                &input.right_discharge,
                config,
                panels,
                work,
            )?;
            let oracle = Oracle {
                geometry,
                bands: &bands,
                left: &left,
                right: &right,
                config,
                panels,
                work,
            };
            let solution = sample(&oracle)?;
            if let Err(error) = checked(c, "surveyed flux acceptance", |a| {
                accept(a, &solution, config)
            }) {
                return Err(resolve_acceptance(error, solution.root_work_limited));
            }
            Ok(solution)
        })();
        match attempt {
            Ok(solution) => return Ok(solution),
            Err(error) => {
                let retry = matches!(
                    &error,
                    HydraulicError::Rejected {
                        reason: PrismaticRiemannRejection::QuadratureUnresolved
                            | PrismaticRiemannRejection::RootBracketUnresolved
                            | PrismaticRiemannRejection::SamplingUnresolved
                            | PrismaticRiemannRejection::FluxBudgetExceeded,
                        ..
                    }
                );
                if !retry {
                    return Err(error);
                }
                let executed = work.snapshot();
                if executed.simpson_value_points == initial_work.simpson_value_points
                    && executed.simpson_derivative_points == initial_work.simpson_derivative_points
                {
                    return Err(error);
                }
                let next = panels.checked_mul(2);
                if refinement >= config.max_refinements
                    || next.is_none_or(|next| next > config.max_simpson_panels)
                {
                    if matches!(
                        &error,
                        HydraulicError::Rejected {
                            reason: PrismaticRiemannRejection::FluxBudgetExceeded,
                            ..
                        }
                    ) {
                        return Err(error);
                    }
                    return Err(HydraulicError::Rejected {
                        reason: PrismaticRiemannRejection::WorkLimit,
                        query: None,
                        section_owner: None,
                        context: format!(
                            "Hydraulic refinement reached panels={panels}, refinement={refinement}. Previous cause: {}. Review conditioning or resource ceilings",
                            describe(&error)
                        ),
                    });
                }
                panels = next.ok_or_else(|| HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    query: None,
                    section_owner: None,
                    context:
                        "Simpson panel doubling overflows usize. Reduce the initial panel count"
                            .into(),
                })?;
                refinement += 1;
            }
        }
    }
}

fn describe(error: &HydraulicError) -> String {
    match error {
        HydraulicError::Rejected {
            reason, context, ..
        } => format!("{reason:?}: {context}"),
        HydraulicError::Arithmetic(error) => error.to_string(),
        HydraulicError::Runtime(error) => error.to_string(),
        HydraulicError::Geometry(error) => error.to_string(),
        HydraulicError::Numerical { context } => context.clone(),
    }
}

pub(crate) fn resolve_acceptance(error: HydraulicError, root_work_limited: bool) -> HydraulicError {
    if root_work_limited
        && matches!(
            &error,
            HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::FluxBudgetExceeded,
                ..
            }
        )
    {
        return HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::WorkLimit,
            query: None,
            section_owner: None,
            context: format!(
                "Certified root WorkLimit prevents accepted flux accuracy. Previous cause: {}. Review root work ceilings",
                describe(&error)
            ),
        };
    }
    error
}
