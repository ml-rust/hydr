//! Actual runtime execution and arithmetic admission through public components.

#[path = "support/runtime.rs"]
mod runtime;

use hydr::soil::{
    ModifiedGreenAmptBatch, ModifiedGreenAmptParameters, NativeSoilInput, NativeSoilState,
};
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

macro_rules! dry_soil_execution {
    ($runtime:ty, $client:expr) => {{
        let parameters = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
        let initial = NativeSoilState {
            imd: 0.2,
            f_ft: 0.01,
            fu_ft: 0.02,
            saturated: true,
            t_s: 5.0,
        };
        let batch = ModifiedGreenAmptBatch::<$runtime>::from_native_states(
            $client,
            &[parameters],
            &[initial],
        )?;
        let trial = batch.trial_native(
            60.0,
            &[NativeSoilInput {
                incoming_rate_ft_per_s: 0.0,
                ponded_depth_ft: 0.0,
            }],
        )?;
        assert_eq!(trial.outputs()?[0].native_rate_ft_per_s, 0.0);
        assert_eq!(
            trial.into_next().native_states()?,
            vec![NativeSoilState {
                t_s: -55.0,
                ..initial
            }]
        );
        assert_eq!(batch.native_states()?, vec![initial]);
    }};
}

#[test]
fn selected_cpu_client_executes_soil_trial_without_mutating_accepted_state() -> Result {
    dry_soil_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_soil_trial_without_mutating_accepted_state() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    dry_soil_execution!(CudaRuntime, client);
    Ok(())
}

macro_rules! still_water_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::grid::CartesianGrid;
        use hydr::model::{CellState, Domain, FlowState, SpatialReference};
        use hydr::soil::{LandModel, SoilState};
        use hydr::solver::{
            BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, Solver,
            SolverConfig, SolverSetup,
        };

        let grid = CartesianGrid::new(2, 2, 1.0, 1.0)?;
        let count = grid.cell_count();
        let reference = SpatialReference::new(
            "synthetic metric plane".into(),
            "synthetic datum".into(),
            [0.0, 0.0],
        )?;
        let domain = Domain::new(
            grid,
            vec![0.0; count],
            vec![0.0; count],
            vec![true; count],
            reference,
        )?;
        let cells = vec![
            CellState {
                h: 1.0,
                qx: 0.0,
                qy: 0.0
            };
            count
        ];
        let flow = FlowState::new(&domain, cells.clone())?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 2.0,
                rain: vec![0.0; count],
                potential_evaporation: vec![0.0; count],
                withdrawal: vec![0.0; count],
                boundaries: vec![],
            }],
        )?;
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let mut solver = Solver::<$runtime>::new(
            $client,
            SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            InitialState {
                time: 0.0,
                flow,
                soil,
            },
        )?;
        assert_eq!(solver.advance_to(0.5, 1_000)?.accepted_time, 0.5);
        let checkpoint = solver.snapshot();
        let checkpoint_data = checkpoint.data()?;
        assert_eq!(solver.state()?.cells(), cells.as_slice());
        assert_eq!(solver.advance_to(1.0, 1_000)?.accepted_time, 1.0);
        let continued = solver.snapshot().data()?;
        let continued_state = solver.state()?;
        assert_eq!(continued_state.cells(), cells.as_slice());
        assert_eq!(solver.ledger().storage, 4.0);
        assert_eq!(solver.ledger().residual, 0.0);
        solver.restore(&checkpoint)?;
        assert_eq!(solver.snapshot().data()?, checkpoint_data);
        assert_eq!(solver.advance_to(1.0, 1_000)?.accepted_time, 1.0);
        assert_eq!(solver.snapshot().data()?, continued);
        assert_eq!(solver.state()?, continued_state);
        assert_eq!(solver.ledger().storage, 4.0);
        assert_eq!(solver.ledger().residual, 0.0);
    }};
}

#[test]
fn selected_cpu_client_preserves_still_water_and_trusted_restart() -> Result {
    still_water_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_preserves_still_water_and_trusted_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    still_water_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_actuator_rejects_backend_defined_fused_arithmetic() -> Result {
    use hydr::structures::*;
    use numr::dtype::DType;
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    use numr::runtime::{FusedMulAddContract, Runtime};

    assert_eq!(
        CudaRuntime::fused_mul_add_contract(DType::F64),
        FusedMulAddContract::BackendDefined
    );
    let context = StructureBatch::<CudaRuntime>::new(
        CudaClient::new(CudaDevice::new(0))?,
        vec![StructureElement {
            id: "gate".into(),
            endpoint_a: "pool".into(),
            endpoint_b: "river".into(),
            datum: "Analytical runtime datum".into(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
                1.0,
                0.0,
                1.0,
                1.0,
                0.5,
                GateApplicability {
                    coefficient_provenance: "Analytical gate fixture".into(),
                    hydrostatic_quiescent_pool_assertion: "Quiescent analytical endpoints".into(),
                    free_submerged_aperture_applicability: "Declared analytical aperture envelope"
                        .into(),
                    unobstructed_aperture_assertion: "Both endpoints expose the aperture".into(),
                    symmetric_reversal: true,
                },
            )?),
        }],
    )?;
    let result = context.with_gate_actuators(
        vec![GateActuatorDefinition {
            actuator_id: "actuator".into(),
            structure_id: "gate".into(),
            opening_rate_m_per_s: 0.25,
            closing_rate_m_per_s: 0.25,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-10,
            },
            rate_provenance: "Explicit analytical rates".into(),
            ideal_response_applicability: "Ideal finite rate response".into(),
        }],
        vec![GateActuatorInitialState {
            actuator_id: "actuator".into(),
            opening_m: 0.0,
            target_opening_m: 0.5,
        }],
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: vec![],
            end_s: 4.0,
        },
    );
    let error = result
        .err()
        .ok_or("CUDA actuator admitted backend-defined fused arithmetic")?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    assert!(error.to_string().contains("fused arithmetic contracts"));
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_client_rejects_f64_profile() -> Result {
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    assert!(!WgpuRuntime::computes_float_dtype(DType::F64));
    let client = WgpuClient::new(WgpuDevice::new(0))?;
    let parameters = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
    let error = ModifiedGreenAmptBatch::<WgpuRuntime>::initialized(client, &[parameters])
        .err()
        .ok_or("WGPU admitted unsupported F64 profile")?;
    assert!(error.to_string().contains("F64"));
    Ok(())
}
