//! Culvert inlet-control headwater on configured runtime clients.

#[cfg(feature = "cuda")]
use super::culvert_inlet_lanes::{WIDTHS, lanes, layout, outputs, reference_lanes};

/// The headwater query over every frozen inlet row and each layout width, as `Debug` text.
///
/// Arc rows of codes 30 to 47 follow, mixed with circle and box rows.
#[cfg(feature = "cuda")]
fn culvert_inlet_outputs<R>(client: &R::Client) -> super::Result<Vec<(String, Vec<String>)>>
where
    R: numr::runtime::Runtime<DType = numr::dtype::DType>,
    R::Client: hydr::structures::CulvertClient<R>,
{
    use super::culvert_shapes::{lane_outputs as run, lanes as arcs};
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
    let lanes = arcs::inlet_lanes()?;
    found.push((
        "arc rows".to_string(),
        run::inlet_outputs::<R>(client, &lanes)?,
    ));
    for width in arcs::WIDTHS {
        let batch = run::picked(width, &lanes);
        found.push((
            format!("arc width {width}"),
            run::inlet_outputs::<R>(client, &batch)?,
        ));
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
    let row = |barrel, code| -> super::Result<CulvertInletRow> {
        Ok(CulvertInletRow {
            barrel,
            inlet: CulvertInlet::hds5(code)?,
            slope: 0.01,
            gravity_m_per_s2: 9.81,
            face_blockage: hydr::structures::BlockageFraction::NONE,
        })
    };
    let mut rows = vec![row(CulvertBarrel::circular(1.0)?, 1)?];
    // One code of each arc shape, each in its own batch.
    let arcs = super::culvert_shapes::lanes::arc_barrels()?;
    for (barrel, code) in arcs.into_iter().zip([30, 33, 36, 45]) {
        rows.push(row(barrel, code)?);
    }
    for row in rows {
        let error = super::f64_rejection(
            CulvertInletBatch::<WgpuRuntime>::new(super::wgpu_client()?, vec![row]),
            "culvert inlet",
        )?;
        assert_eq!(
            error.input_kind(),
            Some(StructureInputErrorKind::Definition),
            "{:?}",
            row.barrel.shape()
        );
    }
    Ok(())
}
