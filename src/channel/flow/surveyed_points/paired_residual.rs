use super::paired_sources::PairedSources;
use super::wave_curve_types::CurveValue;
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
    section::{PhysicalEndpoint, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

pub(super) fn endpoint<R>(
    parent: &mut PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    endpoint: PhysicalEndpoint,
) -> HydraulicResult<CurveValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let terms = super::paired_endpoint::prepare(
        parent,
        rows,
        query,
        endpoint,
        super::paired_endpoint::EndpointEvidenceMode::Discard,
    )?;
    let super::paired_endpoint::EndpointTerms {
        stage,
        far,
        local,
        far_sources,
        target_value,
        ..
    } = terms;
    let c = parent.geometry.client();
    let n = stage.original_rows.numel();
    checked(c, "paired cancellation before target potentials", |a| {
        let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
        let retained_du = &stage.velocity.difference;
        let mut retained = a.expansion_bounds(
            &c.where_cond(&stage.velocity.both_wet, retained_du.high(), &zero)?,
            &c.where_cond(&stage.velocity.both_wet, retained_du.low(), &zero)?,
            &c.where_cond(&stage.velocity.both_wet, retained_du.abs_error(), &zero)?,
        )?;
        for value in &far_sources {
            retained = a.expansion_sub(&retained, &value.retained)?;
        }
        for value in &local {
            retained = a.expansion_add(&retained, &value.retained)?;
        }
        for mask in &far {
            let term = a.expansion_bounds(
                &c.where_cond(mask, target_value.retained.high(), &zero)?,
                &c.where_cond(mask, target_value.retained.low(), &zero)?,
                &c.where_cond(mask, target_value.retained.abs_error(), &zero)?,
            )?;
            retained = a.expansion_add(&retained, &term)?;
        }
        let physical = a.expansion_interval(&retained)?;
        Ok(CurveValue {
            retained,
            physical,
            available: stage.velocity.both_wet,
        })
    })
}
