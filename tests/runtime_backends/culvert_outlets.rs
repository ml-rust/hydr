//! Culvert outlet-control headwater and jumps on configured runtime clients.

#[cfg(feature = "cuda")]
use super::culvert_outlet_lanes::{WIDTHS, layout, layout_lanes, outputs, reference_lanes};

/// The headwater and jump queries over every frozen outlet row and each layout width, as
/// `Debug` text.
///
/// Arc rows of every shape follow, one per outlet regime, mixed with circle and box rows.
#[cfg(feature = "cuda")]
fn culvert_outlet_outputs<R>(client: &R::Client) -> super::Result<Vec<(String, Vec<String>)>>
where
    R: numr::runtime::Runtime<DType = numr::dtype::DType>,
    R::Client: hydr::structures::CulvertClient<R>,
{
    use super::culvert_shapes::{lane_outputs as run, lanes as arcs};
    let mut found = vec![(
        "reference rows".to_string(),
        outputs::<R>(client, &reference_lanes()?)?,
    )];
    let spaced = layout_lanes()?;
    for width in WIDTHS {
        let batch: Vec<_> = layout(width, spaced.len())
            .into_iter()
            .filter_map(|i| spaced.get(i).copied())
            .collect();
        found.push((format!("width {width}"), outputs::<R>(client, &batch)?));
    }
    let lanes = arcs::outlet_lanes()?;
    found.push((
        "arc rows".to_string(),
        run::outlet_outputs::<R>(client, &lanes)?,
    ));
    for width in arcs::WIDTHS {
        let batch = run::picked(width, &lanes);
        found.push((
            format!("arc width {width}"),
            run::outlet_outputs::<R>(client, &batch)?,
        ));
    }
    Ok(found)
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_culvert_outlets_bit_for_bit() -> super::Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = culvert_outlet_outputs::<super::runtime::Runtime>(&super::runtime::client())?;
    let cuda = culvert_outlet_outputs::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)?;
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
fn selected_wgpu_rejects_culvert_outlet_f64_without_fallback() -> super::Result {
    use hydr::structures::{
        CulvertBarrel, CulvertOutletBatch, CulvertOutletRow, StructureInputErrorKind,
    };
    use numr::runtime::wgpu::WgpuRuntime;
    let row = |barrel| CulvertOutletRow {
        barrel,
        length_m: 40.0,
        manning_n: 0.013,
        entrance_loss: 0.5,
        slope: 0.001,
        gravity_m_per_s2: 9.81,
        face_blockage: hydr::structures::BlockageFraction::NONE,
    };
    let arcs = super::culvert_shapes::lanes::arc_barrels()?;
    // The circle, then each arc shape alone.
    let barrels = std::iter::once(CulvertBarrel::circular(1.0)?).chain(arcs);
    for barrel in barrels {
        let error = super::f64_rejection(
            CulvertOutletBatch::<WgpuRuntime>::new(super::wgpu_client()?, vec![row(barrel)]),
            "culvert outlet",
        )?;
        assert_eq!(
            error.input_kind(),
            Some(StructureInputErrorKind::Definition),
            "{:?}",
            barrel.shape()
        );
    }
    Ok(())
}
