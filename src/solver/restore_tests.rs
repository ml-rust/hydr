use super::*;
use crate::grid::CartesianGrid;
use crate::model::SpatialReference;
use crate::soil::{LandCell, ModifiedGreenAmptParameters};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

fn engine(client: CpuClient) -> Result<Solver<CpuRuntime>, Box<dyn std::error::Error>> {
    let domain = Domain::new(
        CartesianGrid::new(1, 1, 1.0, 1.0)?,
        vec![0.0],
        vec![0.0],
        vec![true],
        SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
    )?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![super::super::ForcingInterval {
            start: 0.0,
            end: 1.0,
            rain: vec![1e-6],
            potential_evaporation: vec![0.0],
            withdrawal: vec![0.0],
            boundaries: Vec::new(),
        }],
    )?;
    let parameters = ModifiedGreenAmptParameters::from_native(0.5, 1e-5, 0.3)?;
    let land = LandModel::new(&domain, vec![LandCell::ModifiedGreenAmpt(parameters)])?;
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(
        &domain,
        vec![CellState {
            h: 0.02,
            qx: 0.0,
            qy: 0.0,
        }],
    )?;
    Ok(Solver::<CpuRuntime>::new(
        client,
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
    )?)
}

#[test]
fn trusted_restore_retains_destination_client_with_soil_and_receipts()
-> Result<(), Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let source_config = ParallelismConfig::new(Some(1), Some(3));
    let destination_config = ParallelismConfig::new(Some(1), Some(7));
    let mut source = engine(CpuClient::new(device.clone()).with_parallelism(source_config))?;
    let mut destination = engine(CpuClient::new(device).with_parallelism(destination_config))?;
    source.advance_to(0.01, 100)?;
    destination.advance_to(0.02, 100)?;
    let snapshot = source.snapshot();
    let context = Arc::clone(&destination.context);
    destination.restore(&snapshot)?;
    assert!(Arc::ptr_eq(&context, &destination.context));
    assert_eq!(source.context.client.parallelism(), source_config);
    assert_eq!(destination.context.client.parallelism(), destination_config);
    assert_eq!(destination.snapshot().data()?, snapshot.data()?);
    let Some(super::super::receipts::StepReceipt::Device(receipt)) = &destination.accepted.last
    else {
        return Err("accepted step must retain device receipts".into());
    };
    assert_eq!(
        receipt.sources.soil.next.soil_state()?,
        destination.soil_state()?
    );
    source.advance_to(0.03, 100)?;
    destination.advance_to(0.03, 100)?;
    assert_eq!(destination.snapshot().data()?, source.snapshot().data()?);
    assert_eq!(destination.context.client.parallelism(), destination_config);
    Ok(())
}
