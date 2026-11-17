use super::{
    point_flux_admission::unresolved_positive,
    point_flux_types::{PointFluxStatus, ResolvedPointFluxes},
    point_root_session::{PointRootSession, invalid},
    wet_root_types::WetRootStatus,
};
use crate::channel::flow::surveyed::error::{
    HydraulicError, HydraulicResult, PrismaticRiemannRejection,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, IndexingOps, SortingOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Rejects the whole call when any face is not accepted.
/// The error names the failing face with the smallest original row.
pub(super) fn check<R>(
    session: &PointRootSession<'_, '_, R>,
    resolved: &ResolvedPointFluxes<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: CompareOps<R> + IndexingOps<R> + SortingOps<R> + RuntimeClient<R>,
{
    let sources = session.sources();
    let c = sources.geometry.client();
    let accepted = code::<R>(c, PointFluxStatus::Accepted as i64)?;
    let failing = c.nonzero(&c.ne(&resolved.status, &accepted)?)?.flatten()?;
    if failing.numel() == 0 {
        return Ok(());
    }
    let invocation = c.index_select(&resolved.invocation_owner, 0, &failing)?;
    let source = c.index_select(session.invocation_rows(), 0, &invocation)?;
    let originals = c
        .index_select(&sources.original_rows, 0, &source)?
        .try_to_vec::<i64>()?;
    let (first, original) = originals
        .iter()
        .copied()
        .enumerate()
        .min_by_key(|(_, original)| *original)
        .ok_or_else(|| invalid("rejected face set is empty"))?;
    let face = failing.narrow(0, first, 1)?;
    let invocation = invocation.narrow(0, first, 1)?;
    let owner = c
        .index_select(
            &sources.velocity.input().owners,
            0,
            &source.narrow(0, first, 1)?,
        )?
        .item::<i64>()?;
    let status = c.index_select(&resolved.status, 0, &face)?.item::<i64>()?;
    let rejected = |reason, context: String| -> HydraulicResult<HydraulicError> {
        Ok(HydraulicError::Rejected {
            reason,
            query: Some(index(original, "original row")?),
            section_owner: Some(index(owner, "section owner")?),
            context: format!("original row={original}, section owner={owner}: {context}"),
        })
    };
    let error = if status == PointFluxStatus::PendingCandidates as i64 {
        let selection = c
            .index_select(&resolved.selection, 0, &face)?
            .item::<i64>()?;
        let candidates = &resolved
            .selections
            .get(index(selection, "selection")?)
            .ok_or_else(|| invalid("rejected face names an absent selection"))?
            .candidates;
        let position = c
            .nonzero(&c.eq(&candidates.invocation_owner, &invocation)?)?
            .flatten()?;
        if position.numel() != 1 {
            return Err(invalid("rejected face lacks one candidate row"));
        }
        let wet = c
            .index_select(&candidates.wet_status, 0, &position)?
            .item::<i64>()?;
        if wet == WetRootStatus::CapExceeded as i64 {
            rejected(
                PrismaticRiemannRejection::CapExceeded,
                "the star state needs more water than the surveyed section holds. Check input units or extend the surveyed profile".into(),
            )?
        } else {
            rejected(
                PrismaticRiemannRejection::SamplingUnresolved,
                format!(
                    "candidate traces stay incomplete with wet root status={wet}. Inspect the root and wave certificates"
                ),
            )?
        }
    } else if status == PointFluxStatus::PendingPositiveArea as i64 {
        unresolved_positive(original, owner, "trace area")?
    } else if status == PointFluxStatus::FluxBudgetExceeded as i64 {
        let read = |tensor: &Tensor<R>| -> HydraulicResult<f64> {
            Ok(c.index_select(tensor, 0, &face)?.item::<f64>()?)
        };
        let config = sources.evaluation.config;
        rejected(
            PrismaticRiemannRejection::FluxBudgetExceeded,
            format!(
                "water_flux_m3_per_s bounds=[{:?}, {:?}], momentum_flux_m4_per_s2 bounds=[{:?}, {:?}], water_atol={:?}, momentum_atol={:?}, rtol={:?}: refine geometry, root, or quadrature bounds",
                read(resolved.joint_water.lower())?,
                read(resolved.joint_water.upper())?,
                read(resolved.joint_momentum.lower())?,
                read(resolved.joint_momentum.upper())?,
                config.water_flux_atol_m3_per_s,
                config.momentum_flux_atol_m4_per_s2,
                config.flux_rtol,
            ),
        )?
    } else if status == PointFluxStatus::WorkLimit as i64 {
        rejected(
            PrismaticRiemannRejection::WorkLimit,
            "a star or sonic fan continuation reached its root iteration or point ceiling. Inspect work counters before raising root.max_iterations or root.max_points".into(),
        )?
    } else if status == PointFluxStatus::RootBracketUnresolved as i64 {
        rejected(
            PrismaticRiemannRejection::RootBracketUnresolved,
            "the refined sonic root stays unresolved. Inspect the sonic bracket quantities".into(),
        )?
    } else if status == PointFluxStatus::WaveEnvelopeUnresolved as i64 {
        rejected(
            PrismaticRiemannRejection::SamplingUnresolved,
            "a possible wave branch lacks a certified signal speed, or an exact mirror face has water bounds that exclude 0, a trace that is neither star nor dry, or a pressure enclosure outside its momentum bound. Inspect the wave certificates and the selected trace"
                .into(),
        )?
    } else {
        return Err(invalid(
            "rejected face carries an unknown point flux status",
        ));
    };
    Err(error)
}

fn code<R>(c: &R::Client, value: i64) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R>,
{
    Ok(Tensor::from_slice(&[value], &[], c.device())?)
}

fn index(value: i64, name: &str) -> HydraulicResult<usize> {
    usize::try_from(value).map_err(|_| invalid(&format!("rejected face {name} is negative")))
}
