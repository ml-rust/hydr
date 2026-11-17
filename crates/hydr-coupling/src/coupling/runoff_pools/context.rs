use super::{RunoffPoolError, RunoffPoolForcingInterval, RunoffPoolSetup};
use crate::runoff::{Subarea, SubcatchmentInput};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Device;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Context<R: Runtime<DType = DType>> {
    pub client: R::Client,
    pub setup: RunoffPoolSetup,
    pub initial_time: f64,
    pub minimum: f64,
    pub maximum: f64,
    order: Tensor<R>,
    excluded: Option<Tensor<R>>,
    zero: Tensor<R>,
    maximum_value: Tensor<R>,
    lanes: usize,
}
impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub fn new(
        client: R::Client,
        setup: RunoffPoolSetup,
        initial_time: f64,
    ) -> Result<Self, RunoffPoolError> {
        if !initial_time.is_finite() || initial_time < 0.0 {
            return Err(RunoffPoolError::input(
                "common original epoch must be finite and nonnegative",
            ));
        }
        let c = &setup.config;
        if !c.minimum_step_s.is_finite()
            || c.minimum_step_s <= 0.0
            || !c.maximum_step_s.is_finite()
            || c.maximum_step_s < c.minimum_step_s
            || !c.mass_absolute_m3.is_finite()
            || c.mass_absolute_m3 < 0.0
            || !c.mass_relative.is_finite()
            || c.mass_relative < 0.0
            || (c.mass_absolute_m3 == 0.0 && c.mass_relative == 0.0)
        {
            return Err(RunoffPoolError::input(
                "require finite positive common duration limits and a positive physical mass budget",
            ));
        }
        let minimum = c.minimum_step_s.max(setup.pools.config.minimum_step_s);
        let maximum = c.maximum_step_s.min(setup.pools.config.maximum_step_s);
        if minimum > maximum {
            return Err(RunoffPoolError::input(
                "parent and pool duration ranges do not overlap",
            ));
        }
        let lanes = setup
            .subcatchments
            .len()
            .checked_mul(3)
            .ok_or_else(|| RunoffPoolError::input("native outlet lane count overflows"))?;
        if lanes == 0 || setup.pools.external_inflows.is_empty() {
            return Err(RunoffPoolError::input(
                "require represented runoff and nonempty receiving boundaries",
            ));
        }
        let catchments: BTreeMap<_, _> = setup
            .subcatchments
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.as_str(), i))
            .collect();
        let boundaries: BTreeSet<_> = setup
            .pools
            .external_inflows
            .iter()
            .map(|p| p.boundary_id.as_str())
            .collect();
        let mut potential = BTreeSet::new();
        for (i, definition) in setup.subcatchments.iter().enumerate() {
            for area in Subarea::ALL {
                if definition.parameters.area_m2(area) > 0.0
                    && definition.parameters.donor_fraction(area) < 1.0
                {
                    potential.insert(i * 3 + area.index());
                }
            }
        }
        let mut paired = BTreeMap::new();
        let mut used = BTreeSet::new();
        for outlet in &setup.outlets {
            text(&outlet.free_outlet_assertion, "free outlet evidence")?;
            text(&outlet.nonoverlap_assertion, "nonoverlap evidence")?;
            let index = *catchments
                .get(outlet.subcatchment_id.as_str())
                .ok_or_else(|| {
                    RunoffPoolError::input(format!(
                        "unknown outlet subcatchment={}",
                        outlet.subcatchment_id
                    ))
                })?;
            let lane = index * 3 + outlet.subarea.index();
            if !potential.contains(&lane)
                || !boundaries.contains(outlet.boundary_id.as_str())
                || !used.insert(lane)
                || paired.insert(outlet.boundary_id.as_str(), lane).is_some()
            {
                return Err(RunoffPoolError::input(
                    "outlet bindings require unique potential native lanes and unique known receiving boundaries",
                ));
            }
        }
        if used != potential || paired.len() != boundaries.len() {
            return Err(RunoffPoolError::input(
                "pair every potential native outlet and every receiving boundary exactly once",
            ));
        }
        let order = setup
            .pools
            .external_inflows
            .iter()
            .map(|p| {
                paired.get(p.boundary_id.as_str()).copied().ok_or_else(|| {
                    RunoffPoolError::input("receiving boundary lacks a native outlet")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let indices = |v: Vec<usize>| -> Result<Tensor<R>, RunoffPoolError> {
            let values = v
                .into_iter()
                .map(|i| {
                    i64::try_from(i).map_err(|_| {
                        RunoffPoolError::input("native outlet index exceeds runtime support")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Tensor::from_slice(
                &values,
                &[values.len()],
                client.device(),
            )?)
        };
        let order = indices(order)?;
        let unused = (0..lanes)
            .filter(|i| !potential.contains(i))
            .collect::<Vec<_>>();
        let excluded = if unused.is_empty() {
            None
        } else {
            Some(indices(unused)?)
        };
        let zero = Tensor::<R>::zeros(&[], DType::F64, client.device())?;
        let maximum_value = Tensor::<R>::full_scalar(&[], DType::F64, f64::MAX, client.device())?;
        Ok(Self {
            client,
            setup,
            initial_time,
            minimum,
            maximum,
            order,
            excluded,
            zero,
            maximum_value,
            lanes,
        })
    }
    pub fn check_forcing(
        &self,
        forcing: &RunoffPoolForcingInterval,
        start: f64,
        requested: f64,
    ) -> Result<(), RunoffPoolError> {
        if !forcing.start_s.is_finite()
            || !forcing.end_s.is_finite()
            || !requested.is_finite()
            || forcing.start_s > start
            || start >= requested
            || requested > forcing.end_s
            || forcing.subcatchments.len() != self.setup.subcatchments.len()
            || forcing.pool_sources.is_some() != self.setup.pools.source_policies.is_some()
            || forcing
                .pool_sources
                .as_ref()
                .is_some_and(|s| s.len() != self.setup.pools.pools.len())
        {
            return Err(RunoffPoolError::input(format!(
                "forcing coverage=[{},{}], accepted={start}, requested={requested}: supply complete finite coverage and enabled child inputs",
                forcing.start_s, forcing.end_s
            )));
        }
        if forcing.subcatchments.iter().any(|m| {
            !m.rain_rate_m_per_s.is_finite()
                || m.rain_rate_m_per_s < 0.0
                || !m.potential_evaporation_m_per_s.is_finite()
                || m.potential_evaporation_m_per_s < 0.0
        }) || forcing.pool_sources.as_ref().is_some_and(|s| {
            s.iter().any(|s| {
                [
                    s.rain_rate_m_per_s,
                    s.potential_evaporation_rate_m_per_s,
                    s.bottom_loss_rate_m3_per_s,
                ]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            })
        }) {
            return Err(RunoffPoolError::input(
                "meteorology and local pool sources require finite nonnegative original rates",
            ));
        }
        Ok(())
    }
    pub fn runoff_inputs(&self, forcing: &RunoffPoolForcingInterval) -> Vec<SubcatchmentInput> {
        forcing
            .subcatchments
            .iter()
            .map(|m| SubcatchmentInput {
                rain_rate_m_per_s: m.rain_rate_m_per_s,
                potential_evaporation_m_per_s: m.potential_evaporation_m_per_s,
                external_runon_volumes_m3: [0.0; 3],
            })
            .collect()
    }
    pub fn route(&self, outlets: &Tensor<R>) -> Result<Tensor<R>, RunoffPoolError> {
        let c = &self.client;
        if outlets.shape() != [self.lanes]
            || outlets.dtype() != DType::F64
            || !outlets.device().is_same(c.device())
        {
            return Err(RunoffPoolError::input(
                "original native outlets require complete F64 lane shape on the configured device",
            ));
        }
        let valid = c.mul(
            &c.ge(outlets, &self.zero)?,
            &c.le(outlets, &self.maximum_value)?,
        )?;
        if c.min(&valid, &[0], false)?.item::<f64>()? != 1.0 {
            return Err(RunoffPoolError::numerical(
                "native outlets contain negative or nonfinite water",
            ));
        }
        if let Some(excluded) = &self.excluded {
            let values = c.index_select(outlets, 0, excluded)?;
            if c.max(&c.abs(&values)?, &[0], false)?.item::<f64>()? != 0.0 {
                return Err(RunoffPoolError::numerical(
                    "absent or fully routed native lanes export unbound water",
                ));
            }
        }
        Ok(c.index_select(outlets, 0, &self.order)?)
    }
    pub fn check_pair(
        &self,
        outlets: &Tensor<R>,
        inflows: &Tensor<R>,
        owned: bool,
    ) -> Result<(), RunoffPoolError> {
        let routed = self.route(outlets)?;
        if routed.shape() != inflows.shape()
            || inflows.dtype() != DType::F64
            || !inflows.device().is_same(self.client.device())
        {
            return Err(RunoffPoolError::input(
                "retained receiving inputs differ from complete native outlet layout",
            ));
        }
        if owned {
            let a = routed.try_to_vec::<f64>()?;
            let b = inflows.try_to_vec::<f64>()?;
            if !hydr_structures::internal::bits(&a, &b) {
                return Err(RunoffPoolError::input(
                    "retained receiving inputs differ from original native outlet bits",
                ));
            }
        // Native binding retains the gathered operands. Owned import also checks signed-zero bits.
        } else if self
            .client
            .min(&self.client.eq(&routed, inflows)?, &[0], false)?
            .item::<f64>()?
            != 1.0
        {
            return Err(RunoffPoolError::numerical(
                "receiving inputs differ from original native outlet operands",
            ));
        }
        Ok(())
    }
}
fn text(value: &str, field: &str) -> Result<(), RunoffPoolError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(RunoffPoolError::input(format!(
            "{field}={value:?}: supply explicit physical evidence"
        )));
    }
    Ok(())
}
