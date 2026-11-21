//! Exact interval admission without implicit input slicing or physical composition.

use super::{
    AdmittedInterval, CellRainAssignment, RainSuppressionReceipt, SupportError, SupportErrorKind,
    SupportIntervalInput, SupportPlan,
};
use crate::contract::{BoundaryForcing, ForcingInterval, ForcingSchedule};
use std::sync::Arc;

impl SupportPlan {
    /// Bind one exact physical input interval and compile explicit rainfall suppression.
    pub fn admit_interval(
        &self,
        input: SupportIntervalInput,
    ) -> Result<AdmittedInterval, SupportError> {
        let start = input.start;
        let end = input.end;
        interval(start, end)?;
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
        let active_inputs = self.select_active(input.subcatchments)?;
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
/// Rejects an interval without a finite nonnegative start and a finite positive duration.
pub(super) fn interval(start: f64, end: f64) -> Result<(), SupportError> {
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
    Ok(())
}

pub(super) fn span(
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
