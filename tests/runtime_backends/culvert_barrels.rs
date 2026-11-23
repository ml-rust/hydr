//! Culvert barrel geometry and depth queries on configured runtime clients.

#[cfg(feature = "cuda")]
use super::culvert_lanes::{QUERIES, Query, WIDTHS, lanes, layout, outputs};
#[cfg(feature = "cuda")]
use super::culvert_lanes::{dense_lanes, full_outputs, reference_lanes};

/// Every query over the reference rows, the dense sweep and each layout width, as `Debug` text.
///
/// Arc rows of every shape follow, mixed with circle and box rows: point, range and full
/// geometry, and critical and normal depth through the construction certificates.
#[cfg(feature = "cuda")]
fn culvert_barrel_outputs<R>(client: &R::Client) -> super::Result<Vec<(String, Vec<String>)>>
where
    R: numr::runtime::Runtime<DType = numr::dtype::DType>,
    R::Client: hydr::structures::CulvertClient<R>,
{
    use super::culvert_shapes::{lane_outputs as run, lanes as arcs};
    let mut found = Vec::new();
    for query in QUERIES {
        let rows = reference_lanes(query)?;
        found.push((
            format!("{query:?} reference"),
            outputs::<R>(client, query, &rows)?,
        ));
        let spaced = lanes(query)?;
        for width in WIDTHS {
            let batch: Vec<_> = layout(width, spaced.len())
                .into_iter()
                .map(|i| spaced[i])
                .collect();
            let what = format!("{query:?} width {width}");
            found.push((what, outputs::<R>(client, query, &batch)?));
        }
    }
    let dense = dense_lanes()?;
    let what = "Geometry dense".to_string();
    found.push((what, outputs::<R>(client, Query::Geometry, &dense)?));
    let full = reference_lanes(Query::Geometry)?;
    found.push((
        "full geometry".to_string(),
        full_outputs::<R>(client, &full)?,
    ));
    for query in arcs::QUERIES {
        let lanes = arcs::barrel_lanes(query)?;
        found.push((
            format!("arc {query:?}"),
            run::barrel_outputs::<R>(client, query, &lanes)?,
        ));
        for width in arcs::WIDTHS {
            let batch = run::picked(width, &lanes);
            found.push((
                format!("arc {query:?} width {width}"),
                run::barrel_outputs::<R>(client, query, &batch)?,
            ));
        }
    }
    Ok(found)
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_matches_culvert_barrels_bit_for_bit() -> super::Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let cpu = culvert_barrel_outputs::<super::runtime::Runtime>(&super::runtime::client())?;
    let cuda = culvert_barrel_outputs::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)?;
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
fn selected_wgpu_rejects_culvert_barrel_f64_without_fallback() -> super::Result {
    use hydr::structures::{CulvertBarrel, CulvertBarrelBatch, StructureInputErrorKind};
    use numr::runtime::wgpu::WgpuRuntime;
    let closed = vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.0, 1.0)?,
    ];
    let arcs = super::culvert_shapes::lanes::arc_barrels()?;
    // Each arc shape alone, so no circle or box row can carry the rejection for it.
    let batches = std::iter::once(closed).chain(arcs.into_iter().map(|arc| vec![arc]));
    for barrels in batches {
        let shape = barrels.last().map(|barrel| barrel.shape());
        let error = super::f64_rejection(
            CulvertBarrelBatch::<WgpuRuntime>::new(super::wgpu_client()?, barrels),
            "culvert barrel",
        )?;
        assert_eq!(
            error.input_kind(),
            Some(StructureInputErrorKind::Definition),
            "{shape:?}"
        );
    }
    Ok(())
}
