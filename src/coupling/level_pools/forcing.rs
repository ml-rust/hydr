use super::{CoupledLevelPoolError, CoupledPoolInflow, CoupledPoolInflowValue};
use crate::storage::LevelPoolSources;

/// Complete local source rates and prescribed external inputs over an explicit interval.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolForcingInterval {
    /// Earliest covered time in seconds.
    pub start_s: f64,
    /// Latest covered time in seconds, including the original requested endpoint.
    pub end_s: f64,
    /// One checked rain, bottom-loss, and evaporation record per ordered pool.
    pub sources: Option<Vec<LevelPoolSources>>,
    /// Complete original external inputs, resolved into immutable binding order during admission.
    pub inflows: Vec<CoupledPoolInflow>,
}

impl CoupledLevelPoolForcingInterval {
    pub(super) fn check(
        &self,
        accepted: f64,
        requested: f64,
        count: usize,
        enabled: bool,
    ) -> Result<(), CoupledLevelPoolError> {
        check_coverage(
            (self.start_s, self.end_s),
            (accepted, requested),
            self.sources.as_ref().map(Vec::len),
            count,
            enabled,
            self.requires_exact_span(),
        )
    }

    pub(super) fn requires_exact_span(&self) -> bool {
        self.inflows
            .iter()
            .any(|inflow| matches!(inflow.value, CoupledPoolInflowValue::IntervalVolume { .. }))
    }
    pub(super) fn same_bits(&self, other: &Self) -> bool {
        let source_bits = match (&self.sources, &other.sources) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b).all(|(a, b)| {
                        hydr_structures::internal::bits(
                            &[
                                a.rain_rate_m_per_s,
                                a.potential_evaporation_rate_m_per_s,
                                a.bottom_loss_rate_m3_per_s,
                            ],
                            &[
                                b.rain_rate_m_per_s,
                                b.potential_evaporation_rate_m_per_s,
                                b.bottom_loss_rate_m3_per_s,
                            ],
                        )
                    })
            }
            _ => false,
        };
        self.start_s.to_bits() == other.start_s.to_bits()
            && self.end_s.to_bits() == other.end_s.to_bits()
            && source_bits
            && self.inflows.len() == other.inflows.len()
            && self.inflows.iter().zip(&other.inflows).all(|(a, b)| {
                a.boundary_id == b.boundary_id
                    && a.input_id == b.input_id
                    && a.value.same_bits(&b.value)
            })
    }
}

pub(super) fn check_coverage(
    coverage: (f64, f64),
    actual: (f64, f64),
    source_count: Option<usize>,
    count: usize,
    enabled: bool,
    exact_span: bool,
) -> Result<(), CoupledLevelPoolError> {
    let (start_s, end_s) = coverage;
    let (accepted, requested) = actual;
    if source_count.is_some() != enabled {
        return Err(CoupledLevelPoolError::input(
            "forcing source presence differs from setup policies. Supply complete rates exactly when local sources are enabled",
        ));
    }
    if !start_s.is_finite()
        || !end_s.is_finite()
        || !requested.is_finite()
        || start_s > accepted
        || accepted >= requested
        || requested > end_s
        || source_count.is_some_and(|records| records != count)
    {
        return Err(CoupledLevelPoolError::input(format!(
            "forcing interval=[{},{}], accepted={accepted}, requested={requested}, source records={:?}, pools={count}: supply finite coverage and complete records for every enabled process",
            start_s, end_s, source_count
        )));
    }
    if exact_span
        && (start_s.to_bits() != accepted.to_bits()
            || end_s.to_bits() != requested.to_bits()
            || !(end_s - start_s).is_finite()
            || end_s - start_s <= 0.0)
    {
        return Err(CoupledLevelPoolError::input(format!(
            "integrated inflow span=[{},{}], accepted={accepted}, requested={requested}: recompute complete forcing for this exact positive span",
            start_s, end_s,
        )));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(super) struct BoundPoolForcing<R: numr::runtime::Runtime<DType = numr::dtype::DType>> {
    pub start_s: f64,
    pub end_s: f64,
    pub sources: Option<Vec<LevelPoolSources>>,
    pub inflows: super::inflows::BoundInflows<R>,
}

impl<R: numr::runtime::Runtime<DType = numr::dtype::DType>> BoundPoolForcing<R> {
    pub fn check(
        &self,
        accepted: f64,
        requested: f64,
        count: usize,
        enabled: bool,
    ) -> Result<(), CoupledLevelPoolError> {
        check_coverage(
            (self.start_s, self.end_s),
            (accepted, requested),
            self.sources.as_ref().map(Vec::len),
            count,
            enabled,
            self.inflows.has_interval_volumes(),
        )
    }
}

impl<R: numr::runtime::Runtime<DType = numr::dtype::DType>> BoundPoolForcing<R>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::ShapeOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    pub fn project(
        &self,
        client: &R::Client,
        context: &super::inflows::InflowContext<R>,
    ) -> Result<CoupledLevelPoolForcingInterval, CoupledLevelPoolError> {
        Ok(CoupledLevelPoolForcingInterval {
            start_s: self.start_s,
            end_s: self.end_s,
            sources: self.sources.clone(),
            inflows: context.project_inputs(client, &self.inflows)?,
        })
    }

    pub fn restore(
        &self,
        client: &R::Client,
        context: &super::inflows::InflowContext<R>,
    ) -> Result<Self, CoupledLevelPoolError> {
        Ok(Self {
            start_s: self.start_s,
            end_s: self.end_s,
            sources: self.sources.clone(),
            inflows: context.restore_inputs(client, &self.inflows)?,
        })
    }
}
