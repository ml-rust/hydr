//! Exact interval admission without implicit input slicing or physical composition.

use super::{
    AdmittedInterval, CellRainAssignment, RainSuppressionReceipt, SupportError, SupportErrorKind,
    SupportIntervalInput, SupportPlan,
};
use crate::contract::{BoundaryForcing, ForcingInterval, ForcingSchedule};
use std::collections::BTreeMap;
use std::sync::Arc;

impl SupportPlan {
    /// Bind one exact physical input interval and compile explicit rainfall suppression.
    pub fn admit_interval(
        &self,
        input: SupportIntervalInput,
    ) -> Result<AdmittedInterval, SupportError> {
        let start = input.start;
        let end = input.end;
        if !start.is_finite()
            || !end.is_finite()
            || start < 0.0
            || end <= start
            || !(end - start).is_finite()
        {
            return Err(err(
                SupportErrorKind::Coverage,
                format!(
                    "interval=[{start},{end}): require finite nonnegative start and finite positive duration"
                ),
            ));
        }
        span(
            start,
            end,
            input.surface.start,
            input.surface.end,
            "surface observations",
        )?;
        if input.surface.input_identity != self.inner.input.surface_input_identity {
            return Err(SupportError::identity(format!(
                "surface identity={:?},expected={:?}: supply bound observations",
                input.surface.input_identity, self.inner.input.surface_input_identity
            )));
        }
        for request in &input.subcatchments {
            span(
                start,
                end,
                request.start,
                request.end,
                &request.contribution_id,
            )?;
        }
        for request in &input.boundaries {
            span(start, end, request.start, request.end, &request.boundary_id)?;
        }
        let mut models = BTreeMap::new();
        for request in input.subcatchments {
            if let Some(previous) = models.insert(request.contribution_id.clone(), request) {
                return Err(err(
                    SupportErrorKind::Coverage,
                    format!("duplicate model input={:?}", previous.contribution_id),
                ));
            }
        }
        let mut active_inputs = Vec::with_capacity(self.inner.active_contributions.len());
        for index in &self.inner.active_contributions {
            let source = &self.inner.input.contributions[*index];
            let request = models.remove(&source.id).ok_or_else(|| {
                err(
                    SupportErrorKind::Coverage,
                    format!(
                        "missing active input={:?}: supply exact interval without fallback",
                        source.id
                    ),
                )
            })?;
            if request.input_identity != source.input_identity {
                return Err(SupportError::identity(format!(
                    "model={:?},identity={:?},expected={:?}",
                    source.id, request.input_identity, source.input_identity
                )));
            }
            let forcing = request.input;
            if forcing
                .external_runon_volumes_m3
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
                || !forcing.rain_rate_m_per_s.is_finite()
                || forcing.rain_rate_m_per_s < 0.0
                || !forcing.potential_evaporation_m_per_s.is_finite()
                || forcing.potential_evaporation_m_per_s < 0.0
            {
                return Err(err(
                    SupportErrorKind::Coverage,
                    format!(
                        "model={:?},input={forcing:?}: require finite nonnegative physical inputs",
                        source.id
                    ),
                ));
            }
            if forcing.external_runon_volumes_m3.iter().any(|v| *v > 0.0) {
                return Err(err(
                    SupportErrorKind::Composition,
                    format!(
                        "model={:?}: external runon requires donor provenance and a paired coordinator. Supply zero runon in this ownership unit",
                        source.id
                    ),
                ));
            }
            active_inputs.push(forcing);
        }
        if !models.is_empty() {
            return Err(err(
                SupportErrorKind::Coverage,
                format!(
                    "unknown or suppressed model requests={:?}: provide active sources only",
                    models.keys().collect::<Vec<_>>()
                ),
            ));
        }
        let mut values = vec![None; self.inner.input.boundaries.len()];
        for request in input.boundaries {
            let index = *self
                .inner
                .boundary_indices
                .get(&request.boundary_id)
                .ok_or_else(|| {
                    err(
                        SupportErrorKind::Identity,
                        format!("unknown boundary={:?}", request.boundary_id),
                    )
                })?;
            let binding = &self.inner.input.boundaries[index];
            if values[index].is_some() {
                return Err(err(
                    SupportErrorKind::Coverage,
                    format!("duplicate boundary={:?}", binding.id),
                ));
            }
            if request.input_identity != binding.input_identity {
                return Err(SupportError::identity(format!(
                    "boundary={:?},identity={:?},expected={:?}",
                    binding.id, request.input_identity, binding.input_identity
                )));
            }
            if request.value.face() != binding.face {
                return Err(err(
                    SupportErrorKind::Destination,
                    format!(
                        "boundary={:?},face={:?},expected={:?}",
                        binding.id,
                        request.value.face(),
                        binding.face
                    ),
                ));
            }
            let admitted = if binding.contribution.is_some() {
                matches!(
                    request.value,
                    BoundaryForcing::Flux(_) | BoundaryForcing::Discharge(_)
                )
            } else {
                match &request.value {
                    BoundaryForcing::Flux(v) => v.water_rate >= 0.0,
                    BoundaryForcing::Discharge(v) => v.outward_rate >= 0.0,
                    _ => true,
                }
            };
            if !admitted {
                return Err(err(
                    SupportErrorKind::Composition,
                    format!(
                        "boundary={:?},value={:?}: hydrological roots require complete Flux/Discharge. Incoming rates require a selected source",
                        binding.id, request.value
                    ),
                ));
            }
            values[index] = Some(request.value);
        }
        let boundaries = values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                value.ok_or_else(|| {
                    err(
                        SupportErrorKind::Coverage,
                        format!("missing boundary={:?}", self.inner.input.boundaries[i].id),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let surface = input.surface;
        // Check raw observations before suppression so inactive or invalid data cannot disappear.
        let raw = ForcingInterval {
            start,
            end,
            rain: surface.rain,
            potential_evaporation: surface.potential_evaporation,
            withdrawal: surface.withdrawal,
            boundaries,
        };
        ForcingSchedule::new(
            &self.inner.domain,
            &self.inner.boundaries,
            vec![raw.clone()],
        )?;
        let mut forcing = raw;
        let mut suppression = Vec::new();
        for cell in &self.inner.input.surface_cells {
            if let CellRainAssignment::Suppressed {
                selected_contribution_id,
            } = &cell.rain
            {
                let observed = forcing.rain[cell.cell];
                forcing.rain[cell.cell] = 0.0;
                suppression.push(RainSuppressionReceipt {
                    cell: cell.cell,
                    atoms: cell.atoms.clone(),
                    observed_rain_m_per_s: observed,
                    injected_rain_m_per_s: 0.0,
                    selected_contribution_id: selected_contribution_id.clone(),
                });
            }
        }
        Ok(AdmittedInterval {
            plan: Arc::clone(&self.inner),
            forcing,
            active_inputs,
            suppression,
        })
    }
}
fn span(
    start: f64,
    end: f64,
    actual_start: f64,
    actual_end: f64,
    id: &str,
) -> Result<(), SupportError> {
    if start != actual_start || end != actual_end {
        return Err(err(
            SupportErrorKind::Coverage,
            format!(
                "input={id:?},span=[{actual_start},{actual_end}),requested=[{start},{end}): supply a fresh exact interval without volume slicing"
            ),
        ));
    }
    Ok(())
}
fn err(kind: SupportErrorKind, context: impl Into<String>) -> SupportError {
    SupportError::new(kind, context)
}
