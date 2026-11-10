//! Compact native evidence errors retain their original physical lane.

use super::*;
use crate::runoff::{
    Subarea, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentGeometry,
    SubcatchmentParameters, SubcatchmentRouting,
};
use crate::soil::ModifiedGreenAmptParameters;
use crate::tensor::subcatchment::TensorSubcatchmentContext;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

#[test]
fn compact_soil_evidence_rejects_first_tied_lane_and_retains_subarea_mapping()
-> Result<(), Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device).with_parallelism(ParallelismConfig::new(Some(1), None));
    let parameters = SubcatchmentParameters::new(
        SubcatchmentGeometry {
            total_area_m2: 1.0,
            width_m: 1.0,
            slope: 0.01,
            impervious_fraction: 0.5,
            zero_storage_impervious_fraction: 0.5,
            impervious_manning: 0.0,
            pervious_manning: 0.0,
            impervious_depression_m: 0.0,
            pervious_depression_m: 0.0,
        },
        Some(ModifiedGreenAmptParameters::from_native(
            0.125,
            1.0 / 1024.0,
            1.0,
        )?),
        SubcatchmentRouting::Direct,
    )?;
    let context = TensorSubcatchmentContext::<CpuRuntime>::new(
        client,
        (0..3)
            .map(|i| SubcatchmentDefinition {
                id: format!("catchment-{i}"),
                parameters: parameters.clone(),
            })
            .collect(),
        SubcatchmentConfig::default(),
    )?;
    let group = context
        .phase1
        .as_ref()
        .ok_or("Missing direct runoff phase")?;
    assert_eq!(group.soil_slots, vec![2, 5, 8]);
    let zero = group.reservoir.tensor(&[0.0, 0.0, 0.0])?;
    let evidence = TensorSubcatchmentSoilStep {
        infiltration_depth: group.reservoir.tensor(&[0.0, -1.0, -2.0])?,
        adjustment_depth: zero.clone(),
        native_rate: zero.clone(),
        native_available_depth: zero.clone(),
        native_infiltration_depth: zero.clone(),
        native_excess_depth: zero.clone(),
        native_roundoff_bound: zero.clone(),
        bridge_roundoff_bound: zero.clone(),
        total_roundoff_bound: zero,
    };
    let error = check_evidence(group, Some(&evidence))
        .err()
        .ok_or("Missing compact evidence rejection")?;
    assert_eq!(error.element(), Some(5));
    let physical = group.map_reservoir_error(error);
    assert_eq!(physical.catchment(), Some(1));
    assert_eq!(physical.subarea(), Some(Subarea::Pervious));
    Ok(())
}
