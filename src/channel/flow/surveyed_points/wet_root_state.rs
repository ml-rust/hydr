use super::{
    wave_curve_types::{CurveValue, scatter_value, unavailable},
    wet_root_types::{WetRootDomain, WetRootDomains, WetRootStatus},
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) struct Locator<R: Runtime<DType = DType>> {
    pub(super) output: WetRootDomains<R>,
    pub(super) pending: Tensor<R>,
    pub(super) f0: CurveValue<R>,
    pub(super) limit: usize,
}

impl<R> Locator<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn new(client: &R::Client, rows: Tensor<R>, limit: usize) -> HydraulicResult<Self> {
        checked::<R, _>(client, "wet root locator storage", |a| {
            let n = rows.numel();
            Ok(Self {
                output: WetRootDomains {
                    invocation_rows: rows,
                    status: Tensor::full_scalar(
                        &[n],
                        DType::I64,
                        WetRootStatus::Unresolved as i64 as f64,
                        client.device(),
                    )?,
                    domains: Vec::new(),
                    evidence: Vec::new(),
                    callback_points: 0,
                },
                pending: Tensor::ones(&[n], a.dtype(), client.device())?,
                f0: unavailable(a, n)?,
                limit,
            })
        })
    }

    pub(super) fn remaining(&self) -> HydraulicResult<usize> {
        self.limit
            .checked_sub(self.output.callback_points)
            .ok_or_else(|| HydraulicError::Numerical {
                context: "Wet root callback count exceeds its budget. Discard the locator result"
                    .into(),
            })
    }

    pub(super) fn add_points(&mut self, points: usize) -> HydraulicResult<()> {
        let used = self
            .output
            .callback_points
            .checked_add(points)
            .ok_or_else(|| HydraulicError::Numerical {
                context: "Wet root callback count overflows usize. Discard the locator result"
                    .into(),
            })?;
        if used > self.limit {
            return Err(HydraulicError::Numerical {
                context: format!(
                    "Wet root callbacks={used} exceed budget={}. Discard the locator result",
                    self.limit
                ),
            });
        }
        self.output.callback_points = used;
        Ok(())
    }

    pub(super) fn set_status(
        &mut self,
        client: &R::Client,
        owner: &Tensor<R>,
        status: WetRootStatus,
    ) -> HydraulicResult<()> {
        if owner.numel() == 0 {
            return Ok(());
        }
        let code = Tensor::full_scalar(
            owner.shape(),
            DType::I64,
            status as i64 as f64,
            client.device(),
        )?;
        self.output.status = client.index_put(&self.output.status, 0, owner, &code)?;
        self.pending = client.index_put(
            &self.pending,
            0,
            owner,
            &Tensor::zeros(owner.shape(), self.pending.dtype(), client.device())?,
        )?;
        Ok(())
    }

    pub(super) fn pending_rows(&self, client: &R::Client) -> HydraulicResult<Tensor<R>> {
        Ok(client.nonzero(&self.pending)?.flatten()?)
    }

    pub(super) fn retain_f0(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        value: &CurveValue<R>,
    ) -> HydraulicResult<()> {
        let empty = unavailable(a, self.output.invocation_rows.numel())?;
        let previous = std::mem::replace(&mut self.f0, empty);
        self.f0 = scatter_value(a, previous, owner, value)?;
        Ok(())
    }
}

pub(super) fn selected_value<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CurveValue<R>,
    owner: &Tensor<R>,
) -> HydraulicResult<CurveValue<R>>
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
    Ok(CurveValue {
        retained: super::potential_range::select_expansion(a, &value.retained, owner)?,
        physical: a.bounds(
            &c.index_select(value.physical.lower(), 0, owner)?,
            &c.index_select(value.physical.upper(), 0, owner)?,
        )?,
        available: c.index_select(&value.available, 0, owner)?,
    })
}

pub(super) fn exact_zero<R>(
    a: &Arithmetic<'_, R>,
    value: &CurveValue<R>,
) -> HydraulicResult<Tensor<R>>
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
    let zero = Tensor::zeros(value.available.shape(), a.dtype(), c.device())?;
    Ok(c.mul(
        &value.available,
        &c.mul(
            &c.eq(value.retained.high(), &zero)?,
            &c.mul(
                &c.eq(value.retained.low(), &zero)?,
                &c.eq(value.retained.abs_error(), &zero)?,
            )?,
        )?,
    )?)
}

pub(super) fn direct_domain<R>(
    a: &mut Arithmetic<'_, R>,
    invocation_owner: Tensor<R>,
    parent_rows: Tensor<R>,
    query: crate::channel::section::SupportedAreaQuery<R>,
    value: &CurveValue<R>,
) -> HydraulicResult<WetRootDomain<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(WetRootDomain {
        invocation_owner,
        parent_rows,
        query,
        lower_residual: a.bounds(value.physical.lower(), value.physical.upper())?,
        upper_residual: a.bounds(value.physical.lower(), value.physical.upper())?,
        lower_retained: a.expansion_bounds(
            value.retained.high(),
            value.retained.low(),
            value.retained.abs_error(),
        )?,
        upper_retained: a.expansion_bounds(
            value.retained.high(),
            value.retained.low(),
            value.retained.abs_error(),
        )?,
    })
}
