use super::hydraulic_geometry::HydraulicGeometry;
use crate::channel::section::{CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

#[test]
fn facade_retains_configured_client_parallelism() -> Result<(), Box<dyn std::error::Error>> {
    let original = CpuClient::new(CpuDevice::new());
    let original_config = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let configured = original.with_parallelism(config);
    let section = CrossSectionDefinition {
        id: "rectangle".into(),
        vertical_datum: "bed".into(),
        points: [(-1.0, 2.0), (-1.0, 0.0), (1.0, 0.0), (1.0, 2.0)]
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: 2.0,
        profile_provenance: "analytical rectangle".into(),
    };
    let batch = CrossSectionBatch::<CpuRuntime>::new(configured.clone(), vec![section])?
        .certify_prismatic()?;
    let geometry = HydraulicGeometry::from_batch(&batch);
    assert_eq!(geometry.client().parallelism(), config);
    assert_eq!(configured.parallelism(), config);
    assert_eq!(original.parallelism(), original_config);
    Ok(())
}
