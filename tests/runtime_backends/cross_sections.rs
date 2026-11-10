use super::{Result, cross_section_profiles::*, runtime};

macro_rules! cross_section_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::channel::{CrossSectionBatch, CrossSectionError};

        let client = $client;
        let definitions = all();
        let batch = CrossSectionBatch::<$runtime>::new(client.clone(), definitions.clone())?;
        let stages = [1.0, 1.0, 1.0, 1.5];
        let query = batch.evaluate(&stages)?;
        let original = query.properties()?;
        let expected = [
            rectangle_expected(1.0),
            trapezoid_expected(1.0),
            triangle_expected(1.0),
            bench_expected(1.5),
        ];
        for ((actual, expected), definition) in original.iter().zip(expected).zip(&definitions) {
            assert_eq!(actual.section_id, definition.id);
            assert_wet(actual, expected)?;
        }
        let unsupported = batch
            .evaluate(&[4.125, 1.0, 1.0, 1.5])
            .err()
            .ok_or("Unsupported bank stage was accepted")?;
        assert!(matches!(
            unsupported,
            CrossSectionError::UnsupportedStage { section: 0, .. }
        ));
        let unrepresentable = batch
            .evaluate(&[f64::from_bits(1), 1.0, 1.0, 1.5])
            .err()
            .ok_or("Positive unresolved section became dry")?;
        assert!(matches!(
            unrepresentable,
            CrossSectionError::Numerical {
                section: Some(0),
                ..
            }
        ));
        assert_eq!(batch.definitions(), definitions.as_slice());
        let repeated = batch.evaluate(&stages)?.properties()?;
        let retained = query.properties()?;
        for ((a, b), c) in original.iter().zip(&repeated).zip(&retained) {
            assert_same(a, b);
            assert_same(a, c);
        }
        let dry = batch.evaluate(&[-1.0; 4])?.properties()?;
        for output in &dry {
            assert_dry(output, -1.0);
        }
        for exponent in [-100, 100] {
            let scale = 2.0_f64.powi(exponent);
            let scaled = definitions
                .iter()
                .cloned()
                .map(|mut profile| {
                    for point in &mut profile.points {
                        point.station_m *= scale;
                        point.elevation_m *= scale;
                    }
                    profile.maximum_stage_m *= scale;
                    profile
                })
                .collect();
            let scaled_batch = CrossSectionBatch::<$runtime>::new(client.clone(), scaled)?;
            let outputs = scaled_batch
                .evaluate(&stages.map(|h| h * scale))?
                .properties()?;
            for (actual, expected) in outputs.iter().zip(expected) {
                assert_scaled_wet(actual, expected, scale)?;
            }
        }
    }};
}

#[test]
fn selected_cpu_client_evaluates_sections_and_rejects_unresolved_positive_geometry() -> Result {
    cross_section_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_evaluates_sections_and_rejects_unresolved_positive_geometry() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    cross_section_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_client_rejects_cross_section_f64_without_fallback() -> Result {
    use hydr::channel::{CrossSectionBatch, CrossSectionError};
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

    let client = WgpuClient::new(WgpuDevice::new(0))?;
    let error = CrossSectionBatch::<WgpuRuntime>::new(client, vec![rectangle()])
        .err()
        .ok_or("WGPU accepted the unsupported section F64 profile")?;
    assert!(error.to_string().contains("F64"));
    assert!(matches!(error, CrossSectionError::InvalidInput { .. }));
    Ok(())
}
