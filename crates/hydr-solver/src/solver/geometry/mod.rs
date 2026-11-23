//! Surface bed elevation epochs with bit-identical depth and momentum conservation.

mod apply;
mod schedule;

pub use schedule::{
    SurfaceBedEpoch, SurfaceBedSchedule, SurfaceBedUpdate, active_bed_at,
    active_surface_bed_epochs, bed_digest_bits, check_surface_bed_schedules,
    next_surface_bed_event_after,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::CartesianGrid;
    use crate::model::{CellState, Domain, FlowState, SpatialReference};
    use crate::soil::{LandCell, LandModel, SoilState};
    use crate::solver::{
        BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, Solver, SolverConfig,
        SolverSetup,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    #[test]
    fn bed_step_conserves_depth_and_starts_flow_from_lake_at_rest()
    -> Result<(), Box<dyn std::error::Error>> {
        let client = CpuClient::new(CpuDevice::new());
        let domain = Domain::new(
            CartesianGrid::new(2, 1, 2.0, 2.0)?,
            vec![0.0, 0.0],
            vec![0.02, 0.02],
            vec![true, true],
            SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 2.0,
                rain: vec![0.0; 2],
                potential_evaporation: vec![0.0; 2],
                withdrawal: vec![0.0; 2],
                boundaries: Vec::new(),
            }],
        )?;
        let land = LandModel::new(&domain, vec![LandCell::Impervious; 2])?;
        let soil = SoilState::initialized(&land)?;
        let flow = FlowState::new(
            &domain,
            vec![
                CellState {
                    h: 1.0,
                    qx: 0.0,
                    qy: 0.0,
                },
                CellState {
                    h: 1.0,
                    qx: 0.0,
                    qy: 0.0,
                },
            ],
        )?;
        let setup = SolverSetup {
            domain: domain.clone(),
            land,
            boundaries,
            forcing,
            config: SolverConfig::default(),
        };
        let mut solver = Solver::<CpuRuntime>::new(
            client,
            setup,
            InitialState {
                time: 0.0,
                flow,
                soil,
            },
        )?;
        let schedule = SurfaceBedSchedule {
            schedule_id: "lift-left".into(),
            epochs: vec![SurfaceBedEpoch {
                time_s: 0.5,
                updates: vec![SurfaceBedUpdate {
                    cell: 0,
                    bed_elevation_m: 0.25,
                }],
                evidence: "Uplift step on cell 0".into(),
            }],
        };
        check_surface_bed_schedules(&domain, std::slice::from_ref(&schedule))?;
        assert_eq!(
            next_surface_bed_event_after(std::slice::from_ref(&schedule), 0.0),
            Some(0.5)
        );
        assert_eq!(
            active_surface_bed_epochs(std::slice::from_ref(&schedule), 0.5),
            vec![1]
        );

        let before = solver.state()?;
        let before_storage = solver.ledger().storage;
        let bed = active_bed_at(domain.bed(), std::slice::from_ref(&schedule), 0.5);
        solver = solver.coupling_with_bed(bed)?;
        let after = solver.state()?;
        assert_eq!(solver.ledger().storage.to_bits(), before_storage.to_bits());
        for (a, b) in before.cells().iter().zip(after.cells()) {
            assert_eq!(a.h.to_bits(), b.h.to_bits());
            assert_eq!(a.qx.to_bits(), b.qx.to_bits());
            assert_eq!(a.qy.to_bits(), b.qy.to_bits());
        }
        // Step forward: lifted cell 0 (stage 1.25) must now flow into cell 1 (stage 1.0).
        solver.advance_to(0.1, 64)?;
        let moved = solver.state()?;
        assert!(moved.cells()[0].h < 1.0 && moved.cells()[1].h > 1.0);
        assert!(solver.ledger().residual.abs() <= 1e-12);
        Ok(())
    }
}
