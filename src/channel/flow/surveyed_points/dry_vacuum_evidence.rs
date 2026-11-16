use super::{
    dry_vacuum_types::DryVacuumEdges,
    point_root_session::{PointRootSession, invalid},
    source_sonic_types::SonicSide,
    wave_curve_types,
    wet_root_state::selected_value as select_curve,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn owners<R>(
    a: &mut Arithmetic<'_, R>,
    session: &PointRootSession<'_, '_, R>,
    owners: &Tensor<R>,
) -> HydraulicResult<(Tensor<R>, Tensor<R>)>
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
    if owners.ndim() != 1 || owners.dtype() != DType::I64 || !owners.device().is_same(c.device()) {
        return Err(invalid(
            "dry vacuum owners need one-dimensional caller-device I64 metadata",
        ));
    }
    let zero = Tensor::zeros(&[], DType::I64, c.device())?;
    let count = Tensor::full_scalar(
        &[],
        DType::I64,
        session.invocation_rows.numel() as f64,
        c.device(),
    )?;
    if c.nonzero(&c.maximum(&c.lt(owners, &zero)?, &c.ge(owners, &count)?)?)?
        .numel()
        != 0
    {
        return Err(invalid(
            "dry vacuum owner lies outside the captured invocation",
        ));
    }
    if owners.numel() > 1 {
        let sorted = c.sort(owners, 0, false)?;
        let n = owners.numel() - 1;
        if c.nonzero(&c.eq(&sorted.narrow(0, 0, n)?, &sorted.narrow(0, 1, n)?)?)?
            .numel()
            != 0
        {
            return Err(invalid(
                "dry vacuum request contains duplicate invocation owners",
            ));
        }
    }
    let captured = Tensor::empty(owners.shape(), DType::I64, c.device())?;
    c.copy_into(&captured, owners)?;
    Ok((
        captured,
        c.index_select(&session.invocation_rows, 0, owners)?,
    ))
}

pub(super) fn edges<R>(
    a: &mut Arithmetic<'_, R>,
    session: &PointRootSession<'_, '_, R>,
    owners: &Tensor<R>,
    side: SonicSide,
) -> HydraulicResult<DryVacuumEdges<R>>
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
    let mut head = wave_curve_types::unavailable(a, session.invocation_rows.numel())?;
    let mut front = wave_curve_types::unavailable(a, session.invocation_rows.numel())?;
    for request in &session.sonic.requests {
        if request.binding.side != side {
            continue;
        }
        head = wave_curve_types::scatter_value(
            a,
            head,
            &request.binding.owners,
            &request.source_residual,
        )?;
        front = wave_curve_types::scatter_value(
            a,
            front,
            &request.binding.owners,
            &request.dry_residual,
        )?;
    }
    let mut head = select_curve(a, &head, owners)?;
    let mut front = select_curve(a, &front, owners)?;
    if side == SonicSide::Left {
        let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
        let z = a.expansion_point(&zero)?;
        head.retained = a.expansion_sub(&z, &head.retained)?;
        front.retained = a.expansion_sub(&z, &front.retained)?;
        let z = a.point(&zero)?;
        head.physical = a.sub(&z, &head.physical)?;
        front.physical = a.sub(&z, &front.physical)?;
    }
    #[cfg(test)]
    let head_exact_zero = c.mul(
        &head.available,
        &super::wet_star_endpoint::exact_zero(a, &head.retained)?,
    )?;
    #[cfg(test)]
    let front_exact_zero = c.mul(
        &front.available,
        &super::wet_star_endpoint::exact_zero(a, &front.retained)?,
    )?;
    Ok(DryVacuumEdges {
        head,
        front,
        #[cfg(test)]
        head_exact_zero,
        #[cfg(test)]
        front_exact_zero,
    })
}
