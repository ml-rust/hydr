//! Real surface forcing construction with immutable plan-instance checks.

use super::{AdmittedInterval, CellRainAssignment, SupportError, SupportErrorKind, SupportPlan};
use crate::contract::ForcingSchedule;
use std::sync::Arc;

impl SupportPlan {
    /// Construct the actual checked surface schedule from this plan's admitted packets.
    pub fn surface_schedule(
        &self,
        packets: &[AdmittedInterval],
    ) -> Result<ForcingSchedule, SupportError> {
        for packet in packets {
            self.check_packet(packet)?;
        }
        Ok(ForcingSchedule::new(
            &self.inner.domain,
            &self.inner.boundaries,
            packets.iter().map(|p| p.forcing.clone()).collect(),
        )?)
    }

    /// Check that no suppressed cell receives rain in any interval of `schedule`.
    ///
    /// A suppressed cell's whole footprint belongs to its selected source. Rain there would
    /// count the same water twice.
    ///
    /// # Errors
    /// Returns [`SupportError::Admission`] of kind `Ownership`, naming `suppressed-rain`, the
    /// cell, the interval, and the owner, for nonzero rain on a suppressed cell. Returns kind
    /// `Coverage` for a rain array whose length differs from the plan's grid.
    pub fn check_surface_schedule(&self, schedule: &ForcingSchedule) -> Result<(), SupportError> {
        let cells = self.inner.domain.active().len();
        for interval in schedule.intervals() {
            if interval.rain.len() != cells {
                return Err(SupportError::new(
                    SupportErrorKind::Coverage,
                    format!(
                        "interval=[{},{}),rain cells={},grid cells={cells}: supply forcing on the plan's grid",
                        interval.start,
                        interval.end,
                        interval.rain.len()
                    ),
                ));
            }
            for support in &self.inner.input.surface_cells {
                let CellRainAssignment::Suppressed {
                    selected_contribution_id,
                } = &support.rain
                else {
                    continue;
                };
                let rain = interval.rain.get(support.cell).copied().unwrap_or(0.0);
                if rain != 0.0 {
                    return Err(SupportError::new(
                        SupportErrorKind::Ownership,
                        format!(
                            "suppressed-rain: cell={},interval=[{},{}),rain={rain},owner={selected_contribution_id:?}: supply zero rain on cells whose support belongs to a selected source",
                            support.cell, interval.start, interval.end
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn check_packet(&self, packet: &AdmittedInterval) -> Result<(), SupportError> {
        if !Arc::ptr_eq(&self.inner, &packet.plan) {
            return Err(SupportError::identity(
                "packet belongs to another immutable plan instance. Admit fresh inputs through this plan",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{Result, fixture};
    use super::*;
    use crate::contract::{BoundaryConditions, ForcingInterval};

    fn schedule(
        fixture: &super::super::fixture::Fixture,
        rains: &[[f64; 2]],
    ) -> Result<ForcingSchedule> {
        let intervals = rains
            .iter()
            .zip(0_u32..)
            .map(|(rain, index)| ForcingInterval {
                start: f64::from(index),
                end: f64::from(index + 1),
                rain: rain.to_vec(),
                potential_evaporation: vec![0.0; 2],
                withdrawal: vec![0.0; 2],
                boundaries: vec![],
            })
            .collect();
        Ok(ForcingSchedule::new(
            &fixture.domain,
            &BoundaryConditions::impermeable_walls(),
            intervals,
        )?)
    }

    #[test]
    fn schedule_with_zero_rain_on_suppressed_cells_passes() -> Result {
        let fixture = fixture()?;
        let plan = fixture.plan()?;
        plan.check_surface_schedule(&schedule(&fixture, &[[0.0, 1e-5], [0.0, 0.0]])?)?;
        Ok(())
    }

    #[test]
    fn rain_on_a_suppressed_cell_in_any_interval_rejects() -> Result {
        let fixture = fixture()?;
        let plan = fixture.plan()?;
        let late = schedule(&fixture, &[[0.0, 1e-5], [1e-6, 0.0]])?;
        let Err(error) = plan.check_surface_schedule(&late) else {
            return Err("rain on suppressed cell 0 passed".into());
        };
        assert_eq!(error.kind(), Some(SupportErrorKind::Ownership));
        assert!(
            error.to_string().contains("suppressed-rain: cell=0"),
            "{error}"
        );
        Ok(())
    }
}
