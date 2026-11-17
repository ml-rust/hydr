//! First-lane numerical diagnostics using scalar runtime reads.

use super::TensorSubcatchmentContext;
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;
use crate::runoff::{
    ReservoirConfig, ReservoirElement, ReservoirParameters, Subarea, SubcatchmentConfig,
    SubcatchmentDefinition, SubcatchmentGeometry, SubcatchmentParameters, SubcatchmentRouting,
};
use crate::soil::ModifiedGreenAmptParameters;
use hydr_soil::internal::TensorSoilContext;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

type Result = std::result::Result<(), Box<dyn std::error::Error>>;

fn client() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    (device, client)
}
fn strided(device: &CpuDevice, values: &[f64]) -> numr::error::Result<Tensor<CpuRuntime>> {
    let padded: Vec<_> = values.iter().flat_map(|value| [-99.0, *value]).collect();
    Ok(Tensor::from_slice(&padded, &[values.len(), 2], device)?
        .narrow(1, 1, 1)?
        .squeeze(Some(1)))
}

#[test]
fn reservoir_and_soil_diagnostics_keep_first_tied_lane_and_strided_field_value() -> Result {
    let (device, client) = client();
    for count in [3, 33, 257] {
        let first = if count == 3 { 1 } else { 17 };
        let mut flags = vec![0.0; count];
        flags[first] = 1.0;
        flags[count - 1] = 1.0;
        let mask = strided(&device, &flags)?;
        let field = strided(
            &device,
            &(0..count).map(|i| i as f64 + 10.0).collect::<Vec<_>>(),
        )?;
        let parameters = ReservoirParameters::from_si(1.0, 1.0, 1.0, 0.01, 0.0, 0.0)?;
        let reservoirs = TensorReservoirContext::<CpuRuntime>::new(
            client.clone(),
            (0..count)
                .map(|i| ReservoirElement {
                    id: format!("reservoir-{i}"),
                    parameters,
                })
                .collect(),
            ReservoirConfig::default(),
        )?;
        let error = reservoirs
            .reject_mask(
                &mask,
                "ordered scalar field",
                &[("sample", &field)],
                &[("dt", 0.5)],
                "Declared action",
            )
            .err()
            .ok_or("Missing reservoir rejection")?;
        assert_eq!(error.element(), Some(first));
        let message = error.to_string();
        assert!(message.contains(&format!(", sample={:.17e}", first as f64 + 10.0)));
        assert!(message.contains("Declared action"));
        let soils = TensorSoilContext::<CpuRuntime>::new(
            client.clone(),
            &vec![ModifiedGreenAmptParameters::from_native(0.125, 1.0 / 1024.0, 1.0)?; count],
        )?;
        let error = soils
            .reject_mask(
                &mask,
                "ordered scalar field",
                &[("sample", &field)],
                &[("dt", 0.5)],
                "Declared action",
            )
            .err()
            .ok_or("Missing soil rejection")?;
        assert_eq!(error.cell(), Some(first));
        let message = error.to_string();
        assert!(message.contains(&format!(", sample={:.17e}", first as f64 + 10.0)));
        assert!(message.contains("Declared action"));
    }
    Ok(())
}

#[test]
fn runoff_diagnostics_preserve_flat_subarea_and_catchment_mapping() -> Result {
    let (device, client) = client();
    let parameters = SubcatchmentParameters::new(
        SubcatchmentGeometry {
            total_area_m2: 1.0,
            width_m: 1.0,
            slope: 0.01,
            impervious_fraction: 1.0,
            zero_storage_impervious_fraction: 1.0,
            impervious_manning: 0.0,
            pervious_manning: 0.0,
            impervious_depression_m: 0.0,
            pervious_depression_m: 0.0,
        },
        None,
        SubcatchmentRouting::Direct,
    )?;
    let definitions = (0..3)
        .map(|i| SubcatchmentDefinition {
            id: format!("catchment-{i}"),
            parameters: parameters.clone(),
        })
        .collect();
    let context = TensorSubcatchmentContext::<CpuRuntime>::new(
        client,
        definitions,
        SubcatchmentConfig::default(),
    )?;
    let mask = strided(&device, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])?;
    let field = strided(
        &device,
        &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
    )?;
    let error = context
        .reject_mask(
            &mask,
            "flat scalar field",
            &[("sample", &field)],
            &[],
            "Declared action",
        )
        .err()
        .ok_or("Missing flat runoff rejection")?;
    assert_eq!(error.catchment(), Some(1));
    assert_eq!(error.subarea(), Some(Subarea::ImperviousStorage));
    assert!(
        error
            .to_string()
            .contains(&format!(", sample={:.17e}", 14.0))
    );
    let mask = strided(&device, &[0.0, 1.0, 1.0])?;
    let field = strided(&device, &[10.0, 20.0, 30.0])?;
    let error = context
        .reject_mask(
            &mask,
            "catchment scalar field",
            &[("sample", &field)],
            &[],
            "Declared action",
        )
        .err()
        .ok_or("Missing catchment runoff rejection")?;
    assert_eq!(error.catchment(), Some(1));
    assert_eq!(error.subarea(), None);
    assert!(
        error
            .to_string()
            .contains(&format!(", sample={:.17e}", 20.0))
    );
    Ok(())
}
