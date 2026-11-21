//! Culvert inlet-control headwater on configured runtime clients.

#[cfg(feature = "cuda")]
use super::culvert_inlet_lanes::{WIDTHS, lanes, layout, outputs, reference_lanes};

/// The headwater query over every frozen inlet row and each layout width, as `Debug` text.
#[cfg(feature = "cuda")]
fn culvert_inlet_outputs<R>(client: &R::Client) -> super::Result<Vec<(String, Vec<String>)>>
where
    R: numr::runtime::Runtime<DType = numr::dtype::DType>,
    R::Client: numr::ops::TensorOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::NextRepresentableOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::CorrectlyRoundedOps<R>
        + numr::runtime::RuntimeClient<R>
        + solvr::optimize::CertifiedBisectionAlgorithms<R>,
{
    let mut found = vec![(
        "reference rows".to_string(),
        outputs::<R>(client, &reference_lanes()?)?,
    )];
    let spaced = lanes()?;
    for width in WIDTHS {
        let batch: Vec<_> = layout(width, spaced.len())
            .into_iter()
            .filter_map(|i| spaced.get(i).copied())
            .collect();
        found.push((format!("width {width}"), outputs::<R>(client, &batch)?));
    }
    Ok(found)
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_culvert_inlets_bit_for_bit() -> super::Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = culvert_inlet_outputs::<super::runtime::Runtime>(&super::runtime::client())?;
    let cuda = culvert_inlet_outputs::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)?;
    if cpu.len() != cuda.len() {
        return Err(format!("CPU ran {} batches, CUDA {}", cpu.len(), cuda.len()).into());
    }
    for ((what, cpu_rows), (_, cuda_rows)) in cpu.iter().zip(&cuda) {
        super::same_bits(cpu_rows, cuda_rows).map_err(|error| format!("{what}: {error}"))?;
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_culvert_inlet_f64_without_fallback() -> super::Result {
    use hydr::structures::{
        CulvertBarrel, CulvertInlet, CulvertInletBatch, CulvertInletRow, StructureInputErrorKind,
    };
    use numr::runtime::wgpu::WgpuRuntime;
    let rows = vec![CulvertInletRow {
        barrel: CulvertBarrel::circular(1.0)?,
        inlet: CulvertInlet::hds5(1)?,
        slope: 0.01,
        gravity_m_per_s2: 9.81,
    }];
    let error = super::f64_rejection(
        CulvertInletBatch::<WgpuRuntime>::new(super::wgpu_client()?, rows),
        "culvert inlet",
    )?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    Ok(())
}
