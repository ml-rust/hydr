//! Prescribed piecewise-linear and step geometry schedules for hydraulic structures.
//!
//! Each schedule targets one geometric parameter or blockage field of one structure. Knot times
//! join the coordinator's event set so no interval straddles a knot. Each interval evaluates its
//! midpoint geometry on the linear path between knots, or the active step value.

mod evaluation;
mod events;
mod knot;
mod schedule;

pub use evaluation::{evaluate_element_interval, track_value_bits};
pub use events::{active_geometry_epochs, next_geometry_event_after};
pub use knot::{
    DiscreteGeometryKnot, ScalarGeometryParameter, StructureGeometryKnot, StructureGeometryTrack,
};
pub use schedule::{StructureGeometrySchedule, check_schedules};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{
        DirectionPolicy, StructureCondition, StructureElement, StructureLaw,
        SuppressedRectangularWeirParameters, WeirApplicability,
    };

    fn weir_element() -> Result<StructureElement, crate::structures::StructureError> {
        Ok(StructureElement {
            id: "w1".into(),
            endpoint_a: "a".into(),
            endpoint_b: "b".into(),
            datum: "d".into(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
                2.0,
                1.5,
                1.7,
                WeirApplicability {
                    coefficient_provenance: "Test".into(),
                    free_nappe_and_approach_assertion: "Test".into(),
                    submergence_correction_assertion: "Test".into(),
                    symmetric_reversal: true,
                },
            )?),
            condition: StructureCondition::AsBuilt,
        })
    }

    #[test]
    fn linear_and_step_schedules_evaluate_midpoint_and_events()
    -> Result<(), crate::structures::StructureError> {
        let weir = weir_element()?;
        let breach = StructureGeometrySchedule {
            schedule_id: "breach".into(),
            structure_id: "w1".into(),
            track: StructureGeometryTrack::Scalar {
                parameter: ScalarGeometryParameter::WeirCrestElevation,
                knots: vec![
                    StructureGeometryKnot::continuous(0.0, 1.5),
                    StructureGeometryKnot::continuous(10.0, 0.5),
                ],
            },
            evidence: "Linear breach".into(),
        };
        let shut = StructureGeometrySchedule {
            schedule_id: "close".into(),
            structure_id: "w1".into(),
            track: StructureGeometryTrack::Closed {
                knots: vec![DiscreteGeometryKnot::step(5.0, false, true)],
            },
            evidence: "Bulkhead at 5 s".into(),
        };
        let schedules = vec![breach, shut];
        check_schedules(std::slice::from_ref(&weir), &schedules)?;
        assert_eq!(next_geometry_event_after(&schedules, 0.0), Some(5.0));
        assert_eq!(next_geometry_event_after(&schedules, 5.0), Some(10.0));
        assert_eq!(next_geometry_event_after(&schedules, 10.0), None);
        assert_eq!(active_geometry_epochs(&schedules, 0.0), vec![1, 0]);
        assert_eq!(active_geometry_epochs(&schedules, 5.0), vec![1, 1]);
        assert_eq!(active_geometry_epochs(&schedules, 10.0), vec![2, 1]);

        let refs: Vec<&StructureGeometrySchedule> = schedules.iter().collect();
        let before = evaluate_element_interval(&weir, &refs, 0.0, 4.0)?;
        assert!(matches!(
            &before.law,
            StructureLaw::SuppressedRectangularWeir(p) if p.crest_m() == 1.3
        ));
        assert_eq!(before.condition, StructureCondition::AsBuilt);

        let after = evaluate_element_interval(&weir, &refs, 5.0, 6.0)?;
        assert!(matches!(after.condition, StructureCondition::Closed { .. }));
        Ok(())
    }
}
