//! Culvert barrel geometry and depth queries on configured runtime clients.

#[cfg(feature = "cuda")]
use super::culvert_lanes::{QUERIES, Query, WIDTHS, lanes, layout, outputs};
#[cfg(feature = "cuda")]
use super::culvert_lanes::{dense_lanes, full_outputs, reference_lanes};

/// Every query over the reference rows, the dense sweep and each layout width, as `Debug` text.
#[cfg(feature = "cuda")]
fn culvert_barrel_outputs<R>(client: &R::Client) -> super::Result<Vec<(String, Vec<String>)>>
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
    let barrels = vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.0, 1.0)?,
    ];
    let error = super::f64_rejection(
        CulvertBarrelBatch::<WgpuRuntime>::new(super::wgpu_client()?, barrels),
        "culvert barrel",
    )?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    Ok(())
}
