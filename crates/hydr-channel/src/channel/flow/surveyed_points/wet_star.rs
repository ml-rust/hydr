use super::{
    paired_sources::PairedSources,
    source_binding::SourcePanelLevel,
    wet_depth_types::{WetDepthDomain, WetDepthSession},
    wet_star_types::WetStarDomain,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::Interval, integrate::CertifiedSimpsonAlgorithms};

impl<R> WetDepthSession<'_, '_, R>
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
    #[cfg(test)]
    pub(super) fn certify_wet_star(&mut self) -> HydraulicResult<Vec<WetStarDomain<R>>> {
        self.wet_star_demand(None)
    }
    pub(super) fn certify_wet_star_selected(
        &mut self,
        demand: &Tensor<R>,
    ) -> HydraulicResult<Vec<WetStarDomain<R>>> {
        self.wet_star_demand(Some(demand))
    }
    fn wet_star_demand(
        &mut self,
        demand: Option<&Tensor<R>>,
    ) -> HydraulicResult<Vec<WetStarDomain<R>>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        if self.wet.terminal {
            return Err(super::point_root_session::invalid(
                "wet certificates remain sealed after an earlier numerical error",
            ));
        }
        self.check_domains()?;
        let c = self.sources.geometry.client();
        if let Some(mask) = demand {
            super::point_candidates_admission::check(self)?;
            checked::<R, _>(c, "wet star invocation demand admission", |a| {
                super::point_candidates_demand::admit(a, self, mask)
            })?;
        }
        let mut output = if demand.is_some() {
            Vec::new()
        } else {
            Vec::with_capacity(self.wet.domains.len())
        };
        for domain in &self.wet.domains {
            let mut selection = None;
            if let Some(mask) = demand {
                let active = c.index_select(mask, 0, &domain.invocation_owner)?;
                let rows = c.nonzero(&active)?;
                let rows = rows.reshape(&[rows.numel()])?;
                if rows.numel() == 0 {
                    continue;
                }
                if rows.numel() != domain.invocation_owner.numel() {
                    selection = Some(rows);
                }
            }
            if let [index] = domain.levels.policies() {
                let level = self.sources.panel_level(*index)?;
                output.push(star_group(
                    self.sources,
                    domain,
                    selection.as_ref(),
                    &level,
                )?);
                continue;
            }
            // Each output domain holds the rows of one panel level.
            let levels = match &selection {
                Some(rows) => domain.levels.selected(c, rows)?,
                None => domain.levels.clone(),
            };
            for index in levels.policies() {
                let group = levels.rows_at(c, *index)?;
                if group.numel() == 0 {
                    continue;
                }
                let rows = match &selection {
                    Some(rows) => c.index_select(rows, 0, &group)?,
                    None => group,
                };
                let level = self.sources.panel_level(*index)?;
                output.push(star_group(self.sources, domain, Some(&rows), &level)?);
            }
        }
        Ok(output)
    }
}

fn star_group<R>(
    sources: &mut PairedSources<'_, R>,
    domain: &WetDepthDomain<R>,
    rows: Option<&Tensor<R>>,
    level: &SourcePanelLevel,
) -> HydraulicResult<WetStarDomain<R>>
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
    let Some(rows) = rows else {
        return super::wet_candidates_star::evaluate(
            sources,
            &domain.invocation_owner,
            &domain.parent_rows,
            &domain.query,
            level,
        );
    };
    let c = sources.geometry.client();
    let (owners, parent, query) =
        checked::<R, _>(c, "wet star immutable selected operands", |a| {
            Ok((
                c.index_select(&domain.invocation_owner, 0, rows)?,
                c.index_select(&domain.parent_rows, 0, rows)?,
                domain.query.select(a, rows)?,
            ))
        })?;
    super::wet_candidates_star::evaluate(sources, &owners, &parent, &query, level)
}

pub(super) fn take_pair<T>(values: Vec<T>) -> HydraulicResult<[T; 2]> {
    values.try_into().map_err(|_: Vec<T>| {
        crate::channel::flow::surveyed::error::HydraulicError::Numerical {
            context: "Wet star side count differs from two. Rebuild the paired certificate".into(),
        }
    })
}

pub(super) fn ordered_bounds<R>(
    a: &mut Arithmetic<'_, R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    available: &Tensor<R>,
) -> HydraulicResult<(Interval<R>, Tensor<R>)>
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
    let valid = c.le(lower, upper)?;
    let active = c.mul(available, &valid)?;
    let zero = Tensor::zeros(lower.shape(), a.dtype(), c.device())?;
    Ok((
        a.bounds(
            &c.where_cond(&active, lower, &zero)?,
            &c.where_cond(&active, upper, &zero)?,
        )?,
        valid,
    ))
}

pub(super) fn intersection<R>(
    a: &mut Arithmetic<'_, R>,
    first: &Interval<R>,
    second: &Interval<R>,
) -> HydraulicResult<(Interval<R>, Tensor<R>)>
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
    let lower = c.maximum(first.lower(), second.lower())?;
    let upper = c.minimum(first.upper(), second.upper())?;
    let valid = c.le(&lower, &upper)?;
    let zero = Tensor::zeros(lower.shape(), a.dtype(), c.device())?;
    Ok((
        a.bounds(
            &c.where_cond(&valid, &lower, &zero)?,
            &c.where_cond(&valid, &upper, &zero)?,
        )?,
        valid,
    ))
}

#[cfg(test)]
#[path = "wet_star_tests.rs"]
mod tests;
