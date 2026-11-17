use super::{
    acceptance::{accept, acceptance_mask},
    config::PrismaticRiemannConfig,
    curves::prepare_side,
    error::{HydraulicError, HydraulicResult, PrismaticRiemannRejection, checked},
    refinement::{Refinement, select_input, select_solution},
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
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

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
        + CertifiedUpperBracketAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = geometry.client();
    if input.owners.numel() == 0 {
        return checked(c, "surveyed empty Riemann admission", |a| {
            Samples::new(a, 0)?.finish(a)
        });
    }
    let mut retained = checked(c, "surveyed refinement allocation", |a| {
        Refinement::new(a, input.owners.numel())
    })?;
    let mut panels = config.initial_simpson_panels;
    let mut refinement = 0;
    loop {
        work.begin_refinement(panels, refinement)?;
        let initial_work = work.snapshot();
        let attempt = (|| {
            let pending_input = checked(c, "surveyed pending input", |a| retained.input(a, input))?;
            let bands = checked(c, "surveyed gather", |a| {
                geometry.gather(a, &pending_input.owners)
            })?;
            let left = prepare_side(
                geometry,
                &bands,
                &pending_input.left_area,
                &pending_input.left_discharge,
                config,
                panels,
                work,
            )?;
            let right = prepare_side(
                geometry,
                &bands,
                &pending_input.right_area,
                &pending_input.right_discharge,
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
            let mut solution = sample(&oracle)?;
            checked(c, "surveyed exact reflection", |a| {
                super::reflection::tighten(a, &pending_input, &mut solution)
            })?;
            let accepted = checked(c, "surveyed flux acceptance mask", |a| {
                acceptance_mask(a, &solution, config)
            })?;
            let rejected = checked(c, "surveyed retain accepted faces", |a| {
                retained.retain(a, &solution, &accepted)
            })?;
            if retained.pending().numel() == 0 {
                return Ok(());
            }
            let (remaining_solution, remaining_input) =
                checked(c, "surveyed rejected faces", |a| {
                    Ok((
                        select_solution(a, &solution, &rejected)?,
                        select_input(a, &pending_input, &rejected)?,
                    ))
                })?;
            let error = checked(c, "surveyed flux acceptance", |a| {
                accept(a, &remaining_solution, &remaining_input, config)
            }).err().ok_or_else(|| HydraulicError::Numerical {
                context: "Flux acceptance mask retains rows accepted by the diagnostic gate. Check flux acceptance predicates".into(),
            })?;
            Err(resolve_width_error(error, &remaining_solution)?)
        })();
        match attempt {
            Ok(()) => return retained.finish(),
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
                    return Err(retained.map_error(error)?);
                }
                let executed = work.snapshot();
                if executed.simpson_value_points == initial_work.simpson_value_points
                    && executed.simpson_derivative_points == initial_work.simpson_derivative_points
                {
                    return Err(retained.map_error(error)?);
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
                        return Err(retained.map_error(error)?);
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

pub(crate) fn resolve_width_error<R>(
    error: HydraulicError,
    solution: &SampledSolution<R>,
) -> HydraulicResult<HydraulicError>
where
    R: Runtime<DType = DType>,
{
    let limited = match &error {
        HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::FluxBudgetExceeded,
            query: Some(row),
            ..
        } => {
            solution
                .root_work_limited_rows
                .narrow(0, *row, 1)?
                .item::<f64>()?
                != 0.0
        }
        HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::FluxBudgetExceeded,
            query: None,
            ..
        } => solution.root_work_limited,
        _ => false,
    };
    Ok(resolve_acceptance(error, limited))
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
    if let HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::FluxBudgetExceeded,
        query,
        section_owner,
        ..
    } = &error
        && root_work_limited
    {
        return HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::WorkLimit,
            query: *query,
            section_owner: *section_owner,
            context: format!(
                "Certified root WorkLimit prevents accepted flux accuracy. Previous cause: {}. Review root work ceilings",
                describe(&error)
            ),
        };
    }
    error
}
