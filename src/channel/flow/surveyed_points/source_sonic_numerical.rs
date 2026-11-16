use super::{
    paired_source_panels::SourcePanelValues, paired_sources::PairedSources,
    potential_range::select_expansion, root_invocation::RootInvocation,
    source_binding::SourcePanelLevel, source_sonic_types::SourceSonicBinding,
};
use crate::channel::flow::surveyed::{
    equation_ops::select_interval,
    error::{Arithmetic, HydraulicResult, checked},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};
use std::sync::Arc;

struct Proof<R: Runtime<DType = DType>> {
    binding: SourceSonicBinding<R>,
    witness: SourcePanelLevel,
    levels: Tensor<R>,
    epsilon: CertifiedExpansion<R>,
    source: Interval<R>,
    policies: Vec<usize>,
}

/// One immutable numerical generation binds the outgoing invariant to its source and panel policies.
pub(super) struct SourceSonicNumericalEvidence<R: Runtime<DType = DType>>(Arc<Proof<R>>);
impl<R: Runtime<DType = DType>> Clone for SourceSonicNumericalEvidence<R> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}
impl<R: Runtime<DType = DType>> SourceSonicNumericalEvidence<R> {
    pub(super) fn policies(&self) -> &[usize] {
        &self.0.policies
    }
    pub(super) fn epsilon(&self) -> &CertifiedExpansion<R> {
        &self.0.epsilon
    }
    pub(super) fn levels(&self) -> &Tensor<R> {
        &self.0.levels
    }
    pub(super) fn source_physical(&self) -> &Interval<R> {
        &self.0.source
    }
}
impl<R> SourceSonicNumericalEvidence<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn issue(
        sources: &PairedSources<'_, R>,
        binding: &SourceSonicBinding<R>,
        values: &SourcePanelValues<R>,
    ) -> HydraulicResult<Self> {
        let potential = values.checked_side(sources, binding.side.index(), &binding.rows)?;
        let c = sources.geometry.client();
        checked::<R, _>(c, "source sonic authenticated numerical generation", |a| {
            let zero = Tensor::zeros(binding.rows.shape(), a.dtype(), c.device())?;
            if c.nonzero(&c.eq(&potential.available, &zero)?)?.numel() != 0 {
                return Err(super::point_root_session::invalid(
                    "source sonic source potential is unavailable",
                ));
            }
            let selected = sources.select_sources(a, &binding.rows)?;
            let epsilon = if binding.side.index() == 0 {
                a.expansion_add(&selected.velocity.left, &potential.retained)?
            } else {
                a.expansion_sub(&potential.retained, &selected.velocity.right)?
            };
            let owners = copy::<R>(c, &binding.owners)?;
            let parents = copy::<R>(c, &binding.rows)?;
            Ok(Self(Arc::new(Proof {
                binding: binding.selected(owners, parents),
                witness: values.level.clone(),
                levels: Tensor::full_scalar(
                    binding.rows.shape(),
                    DType::I64,
                    values.level.index() as f64,
                    c.device(),
                )?,
                epsilon,
                policies: vec![values.level.index()],
                source: a.bounds(
                    &copy::<R>(c, potential.physical.lower())?,
                    &copy::<R>(c, potential.physical.upper())?,
                )?,
            })))
        })
    }

    pub(super) fn check(
        &self,
        sources: &PairedSources<'_, R>,
        invocation: &RootInvocation,
        binding: &SourceSonicBinding<R>,
    ) -> HydraulicResult<()> {
        sources.check_root_witness(self.0.witness.witness())?;
        binding.check(invocation, binding.side, &self.0.binding)?;
        equal::<R>(
            sources.geometry.client(),
            &binding.owners,
            &self.0.binding.owners,
        )?;
        equal::<R>(
            sources.geometry.client(),
            &binding.rows,
            &self.0.binding.rows,
        )
    }

    pub(super) fn check_history(
        &self,
        sources: &PairedSources<'_, R>,
        levels: &Tensor<R>,
        refresh: Option<&Tensor<R>>,
    ) -> HydraulicResult<()> {
        let c = sources.geometry.client();
        let Some(rows) = refresh else {
            return equal::<R>(c, self.levels(), levels);
        };
        let old = c.index_select(levels, 0, rows)?;
        let expected = c.index_put(levels, 0, rows, &c.add_scalar(&old, 1.0)?)?;
        equal::<R>(c, self.levels(), &expected)
    }

    pub(super) fn check_rows(
        &self,
        sources: &PairedSources<'_, R>,
        side: super::source_sonic_types::SonicSide,
        owners: &Tensor<R>,
        parents: &Tensor<R>,
    ) -> HydraulicResult<()> {
        self.check_parents(sources, side, parents)?;
        equal::<R>(sources.geometry.client(), owners, &self.0.binding.owners)
    }

    pub(super) fn check_parents(
        &self,
        sources: &PairedSources<'_, R>,
        side: super::source_sonic_types::SonicSide,
        parents: &Tensor<R>,
    ) -> HydraulicResult<()> {
        sources.check_root_witness(self.0.witness.witness())?;
        if self.0.binding.side != side {
            return Err(super::point_root_session::invalid(
                "source sonic numerical evidence belongs to another side",
            ));
        }
        equal::<R>(sources.geometry.client(), parents, &self.0.binding.rows)
    }

    pub(super) fn owned(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<Self> {
        let c = a.client();
        Ok(Self(Arc::new(Proof {
            binding: self.0.binding.selected(
                copy::<R>(c, &self.0.binding.owners)?,
                copy::<R>(c, &self.0.binding.rows)?,
            ),
            witness: self.0.witness.clone(),
            policies: self.0.policies.clone(),
            levels: copy::<R>(c, self.levels())?,
            epsilon: a.expansion_bounds(
                &copy::<R>(c, self.epsilon().high())?,
                &copy::<R>(c, self.epsilon().low())?,
                &copy::<R>(c, self.epsilon().abs_error())?,
            )?,
            source: a.bounds(
                &copy::<R>(c, self.source_physical().lower())?,
                &copy::<R>(c, self.source_physical().upper())?,
            )?,
        })))
    }

    pub(super) fn selected(
        &self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let owners = c.index_select(&self.0.binding.owners, 0, rows)?;
        let parents = c.index_select(&self.0.binding.rows, 0, rows)?;
        Ok(Self(Arc::new(Proof {
            binding: self.0.binding.selected(owners, parents),
            witness: self.0.witness.clone(),
            levels: c.index_select(&self.0.levels, 0, rows)?,
            epsilon: select_expansion(a, &self.0.epsilon, rows)?,
            source: select_interval(a, &self.0.source, rows)?,
            policies: self.0.policies.clone(),
        })))
    }

    pub(super) fn refreshed(
        &self,
        sources: &PairedSources<'_, R>,
        rows: &Tensor<R>,
        values: &SourcePanelValues<R>,
    ) -> HydraulicResult<Self> {
        let c = sources.geometry.client();
        checked::<R, _>(c, "source sonic checked policy replacement", |a| {
            let owners = c.index_select(&self.0.binding.owners, 0, rows)?;
            let parents = c.index_select(&self.0.binding.rows, 0, rows)?;
            let binding = self.0.binding.selected(owners, parents);
            let replacement = Self::issue(sources, &binding, values)?;
            let prior = c.index_select(&self.0.levels, 0, rows)?;
            let expected = Tensor::full_scalar(
                prior.shape(),
                DType::I64,
                values.level.index().checked_sub(1).ok_or_else(|| {
                    super::point_root_session::invalid(
                        "source sonic policy replacement requires a positive panel level",
                    )
                })? as f64,
                c.device(),
            )?;
            equal::<R>(c, &prior, &expected)?;
            let previous = select_interval(a, &self.0.source, rows)?;
            let (intersection, valid) =
                super::wet_star::intersection(a, &previous, replacement.source_physical())?;
            let zero = Tensor::zeros(valid.shape(), a.dtype(), c.device())?;
            if c.nonzero(&c.eq(&valid, &zero)?)?.numel() != 0 {
                return Err(super::point_root_session::invalid(
                    "source potential panel levels have disjoint physical certificates",
                ));
            }
            let mut policies = self.0.policies.clone();
            if !policies.contains(&values.level.index()) {
                policies.push(values.level.index());
            }
            let old = &self.0.epsilon;
            let new = replacement.epsilon();
            Ok(Self(Arc::new(Proof {
                binding: self.0.binding.clone(),
                witness: self.0.witness.clone(),
                policies,
                levels: c.index_put(&self.0.levels, 0, rows, replacement.levels())?,
                epsilon: a.expansion_bounds(
                    &c.index_put(old.high(), 0, rows, new.high())?,
                    &c.index_put(old.low(), 0, rows, new.low())?,
                    &c.index_put(old.abs_error(), 0, rows, new.abs_error())?,
                )?,
                source: a.bounds(
                    &c.index_put(self.0.source.lower(), 0, rows, intersection.lower())?,
                    &c.index_put(self.0.source.upper(), 0, rows, intersection.upper())?,
                )?,
            })))
        })
    }
}

fn copy<R>(c: &R::Client, value: &Tensor<R>) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + RuntimeClient<R>,
{
    let owned = Tensor::empty(value.shape(), value.dtype(), c.device())?;
    c.copy_into(&owned, value)?;
    Ok(owned)
}
fn equal<R>(c: &R::Client, actual: &Tensor<R>, expected: &Tensor<R>) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    if actual.shape() != expected.shape()
        || actual.dtype() != expected.dtype()
        || !actual.device().is_same(expected.device())
        || c.nonzero(&c.ne(actual, expected)?)?.numel() != 0
    {
        return Err(super::point_root_session::invalid(
            "source sonic numerical evidence has foreign owner, parent, or panel-level identities",
        ));
    }
    Ok(())
}
