//! Culvert structure laws on configured runtime clients.

#[cfg(feature = "cuda")]
use super::culvert_law_lanes::{WIDTHS, layout, layout_lanes, outputs};

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_culvert_laws_bit_for_bit() -> super::Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = super::runtime::client();
    let cuda = CudaClient::new(CudaDevice::new(0))?;
    let lanes = layout_lanes()?;
    let mut batches = vec![lanes.clone()];
    for width in WIDTHS {
        batches.push(
            layout(width, lanes.len())
                .into_iter()
                .filter_map(|i| lanes.get(i).cloned())
                .collect(),
        );
    }
    for (index, batch) in batches.iter().enumerate() {
        let on_cpu = outputs::<super::runtime::Runtime>(&cpu, batch)?;
        let on_cuda = outputs::<CudaRuntime>(&cuda, batch)?;
        super::same_bits(&on_cpu, &on_cuda).map_err(|error| format!("batch {index}: {error}"))?;
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn selected_wgpu_rejects_culvert_law_f64_without_fallback() -> super::Result {
    use super::support::{Culvert, elements};
    use hydr::structures::{CulvertBarrel, CulvertInlet, StructureBatch, StructureInputErrorKind};
    use numr::runtime::wgpu::WgpuRuntime;
    let culvert = Culvert::both(
        CulvertBarrel::circular(1.0)?,
        [0.0, 0.0],
        CulvertInlet::hds5(1)?,
        0.5,
    );
    let error = super::f64_rejection(
        StructureBatch::<WgpuRuntime>::with_culverts(super::wgpu_client()?, elements(&[culvert])?),
        "culvert law",
    )?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    Ok(())
}
